from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from tts_data_attribution.cli import training as training_cli
from tts_data_attribution.cli.main import main
from tts_data_attribution.dataset import Utterance, UtteranceDataset
from tts_data_attribution.experiment import (
    ExperimentManifest,
    Plan,
    TrainingParameter,
    TrainingRunManifest,
)


def training_manifest(resolved: bool = False) -> TrainingRunManifest:
    manifest = TrainingRunManifest(
        dtype="bfloat16",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        learning_rate=2e-5,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        weight_decay=0.01,
        epochs=3,
        batch_size=1,
        seed=5678,
    )
    if not resolved:
        return manifest
    return replace(
        manifest,
        parameters=(
            TrainingParameter("weight", (2, 2), "float32"),
            TrainingParameter("bias", (2,), "float32"),
        ),
        parameter_groups=(("weight", "bias"),),
    )


def configure_arguments() -> list[str]:
    return [
        "training",
        "configure",
        "study",
        "--lora-rank",
        "8",
        "--lora-alpha",
        "16",
        "--learning-rate",
        "2e-5",
        "--epochs",
        "3",
        "--batch-size",
        "1",
        "--seed",
        "5678",
    ]


def start_arguments(*selection: str) -> list[str]:
    return [
        "training",
        "start",
        "study",
        "run-20260820T120000000000Z",
        *selection,
        "--device",
        "cpu",
    ]


def create_initialized_experiment(root: Path) -> tuple[Path, Plan]:
    directory = root / "experiments" / "study"
    directory.mkdir(parents=True)
    (directory / "manifest.yaml").write_text("manifest", encoding="utf-8")
    plan = Plan(
        references={"speaker": "reference"},
        training_pool=["train-0", "train-1"],
        validation_pool=["validation-0"],
        query_pool=[],
        subsets={
            "subset-0000": ["train-0"],
            "subset-0001": ["train-1"],
        },
    )
    plan.to_json(directory / "plan.json")
    return directory, plan


def create_ready_experiment(
    root: Path,
    *,
    resolved: bool = False,
) -> tuple[Path, Plan, str]:
    directory, plan = create_initialized_experiment(root)
    ExperimentManifest(
        dataset="dailytalk",
        data_root=Path("dataset"),
        model="qwen3-tts",
        model_path=Path("model"),
        training_pool_size=2,
        validation_pool_size=1,
        query_pool_size=0,
        subset_count=2,
        subset_size=1,
        speaker_count=1,
        seed=1234,
    ).to_yaml(directory / "manifest.yaml")
    run_name = "run-20260820T120000000000Z"
    run_directory = directory / "training-runs" / run_name
    run_directory.mkdir(parents=True)
    training_manifest(resolved).to_yaml(run_directory / "manifest.yaml")
    UtteranceDataset(
        Utterance(
            id=identifier,
            speaker="speaker",
            dialogue=identifier,
            text_ids=[1],
            audio_codes=[[2] * 16],
        )
        for identifier in [*plan.training_pool, *plan.validation_pool]
    ).to_jsonl(directory / "sampled_utterances_encoded.jsonl")
    torch.save({"speaker": torch.ones(4)}, directory / "speaker_embeddings.pt")
    return directory, plan, run_name


def write_complete_checkpoint(path: Path, metrics: dict[str, int | float]) -> None:
    (path / "adapter").mkdir(parents=True)
    (path / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def install_fake_training(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[list[str]], list[Path]]:
    trained_ids: list[list[str]] = []
    checkpoints: list[Path] = []

    def load_model(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(talker=nn.Linear(2, 2))

    def apply_lora(model: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
        return model

    def train(*args: Any, **kwargs: Any) -> list[dict[str, int | float]]:
        loader = cast(DataLoader, args[1])
        dataset = cast(UtteranceDataset, loader.dataset)
        trained_ids.append([utterance.id for utterance in dataset])
        metrics: dict[str, int | float] = {
            "epoch": 3,
            "step": len(loader) * 3,
            "training_loss": 1.0,
            "validation_loss": 2.0,
        }
        callback = args[8]
        callback(metrics)
        return [metrics]

    def save_checkpoint(
        directory: str | Path,
        adapter: object,
        optimizer: object,
        metrics: dict[str, int | float],
    ) -> None:
        path = Path(directory)
        checkpoints.append(path)
        write_complete_checkpoint(path, metrics)

    monkeypatch.setattr(training_cli, "load_model", load_model)
    monkeypatch.setattr(training_cli, "apply_lora", apply_lora)
    monkeypatch.setattr(training_cli, "train", train)
    monkeypatch.setattr(training_cli, "save_lora_checkpoint", save_checkpoint)
    return trained_ids, checkpoints


def test_training_configure_creates_one_configuration_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, _ = create_initialized_experiment(tmp_path)

    assert main(configure_arguments()) == 0

    [run_directory] = (directory / "training-runs").iterdir()
    assert re.fullmatch(r"run-\d{8}T\d{12}Z", run_directory.name)
    assert (
        TrainingRunManifest.from_yaml(run_directory / "manifest.yaml")
        == training_manifest()
    )
    assert sorted(path.name for path in run_directory.iterdir()) == ["manifest.yaml"]


def test_training_configure_requires_an_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(configure_arguments()) == 1

    assert "run experiment init first" in capsys.readouterr().err


def test_training_configure_validates_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, _ = create_initialized_experiment(tmp_path)
    command = configure_arguments()
    command[command.index("8", command.index("--lora-rank"))] = "0"

    assert main(command) == 1

    assert "LoRA rank must be positive" in capsys.readouterr().err
    assert not (directory / "training-runs").exists()


def test_training_start_trains_the_pool_and_resolves_the_run_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, plan, run_name = create_ready_experiment(tmp_path)
    trained_ids, checkpoints = install_fake_training(monkeypatch)

    assert main(start_arguments("--training-pool")) == 0

    run = directory / "training-runs" / run_name
    model = run / "training-pool"
    assert trained_ids == [plan.training_pool]
    assert [path.resolve() for path in checkpoints] == [
        model / "checkpoints" / "epoch-0003"
    ]
    assert (model / "metrics.jsonl").is_file()
    assert (checkpoints[0] / "metrics.json").is_file()
    assert not (run / "target").exists()
    assert not (run / "best-checkpoint.json").exists()
    assert not (checkpoints[0] / "metadata.json").exists()
    assert not (checkpoints[0] / "training-run.yaml").exists()
    assert TrainingRunManifest.from_yaml(run / "manifest.yaml") == training_manifest(
        True
    )


def test_training_start_trains_a_subset_inside_the_same_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, plan, run_name = create_ready_experiment(tmp_path, resolved=True)
    trained_ids, checkpoints = install_fake_training(monkeypatch)

    assert main(start_arguments("--subset", "subset-0001")) == 0

    model = directory / "training-runs" / run_name / "subsets" / "subset-0001"
    assert trained_ids == [plan.subsets["subset-0001"]]
    assert [path.resolve() for path in checkpoints] == [
        model / "checkpoints" / "epoch-0003"
    ]


def test_training_start_requires_the_pool_before_a_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    create_ready_experiment(tmp_path)

    assert main(start_arguments("--subset", "subset-0000")) == 1

    assert (
        "training-pool model must be started before subsets" in capsys.readouterr().err
    )


def test_training_start_rejects_an_unknown_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    create_ready_experiment(tmp_path, resolved=True)

    assert main(start_arguments("--subset", "missing")) == 1

    assert "unknown training subset" in capsys.readouterr().err


def test_training_start_refuses_an_existing_model_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, _, run_name = create_ready_experiment(tmp_path)
    (directory / "training-runs" / run_name / "training-pool").mkdir()

    assert main(start_arguments("--training-pool")) == 1

    assert "already started" in capsys.readouterr().err


@pytest.mark.parametrize("run_name", ["..", "../outside", "/tmp/outside"])
def test_training_start_rejects_non_flat_run_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    command = start_arguments("--training-pool")
    command[3] = run_name

    assert main(command) == 1

    assert (
        "training run name must be a single path component" in capsys.readouterr().err
    )


def test_training_start_requires_a_configured_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    create_initialized_experiment(tmp_path)

    assert main(start_arguments("--training-pool")) == 1

    assert "training run manifest" in capsys.readouterr().err


def test_training_start_requires_embeddings_for_encoded_speakers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, _, _ = create_ready_experiment(tmp_path)
    torch.save({}, directory / "speaker_embeddings.pt")

    assert main(start_arguments("--training-pool")) == 1

    assert "speaker embeddings missing" in capsys.readouterr().err
