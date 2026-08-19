from __future__ import annotations

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
from tts_data_attribution.experiment import ExperimentManifest, Plan, TrainingConfig


def training_config() -> TrainingConfig:
    return TrainingConfig(
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


def create_initialized_experiment(root: Path) -> Path:
    directory = root / "experiments" / "study"
    directory.mkdir(parents=True)
    (directory / "manifest.yaml").write_text("manifest", encoding="utf-8")
    (directory / "plan.json").write_text("{}", encoding="utf-8")
    return directory


def create_ready_experiment(root: Path) -> tuple[Path, Plan]:
    directory = root / "experiments" / "study"
    directory.mkdir(parents=True)
    ExperimentManifest(
        dataset="dailytalk",
        data_root=Path("dataset"),
        model="qwen3-tts",
        model_path=Path("model"),
        training_pool_size=2,
        validation_pool_size=1,
        subset_count=2,
        subset_size=1,
        speaker_count=1,
        seed=1234,
    ).to_yaml(directory / "manifest.yaml")
    plan = Plan(
        references={"speaker": "reference"},
        training_pool=["train-0", "train-1"],
        validation_pool=["validation-0"],
        subsets={
            "subset-0000": ["train-0"],
            "subset-0001": ["train-1"],
        },
    )
    plan.to_json(directory / "plan.json")
    training_config().to_yaml(directory / "training.yaml")
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
    return directory, plan


def write_complete_checkpoint(path: Path) -> None:
    (path / "adapter").mkdir(parents=True)
    (path / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "metadata.json").write_text("{}", encoding="utf-8")


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
        return [{"epoch": 1, "step": len(loader)}]

    def save_lora_checkpoint(
        directory: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        path = Path(directory)
        checkpoints.append(path)
        write_complete_checkpoint(path)

    monkeypatch.setattr(training_cli, "load_model", load_model)
    monkeypatch.setattr(training_cli, "apply_lora", apply_lora)
    monkeypatch.setattr(training_cli, "train", train)
    monkeypatch.setattr(
        training_cli,
        "save_lora_checkpoint",
        save_lora_checkpoint,
    )
    return trained_ids, checkpoints


def test_training_configure_writes_valid_canonical_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = create_initialized_experiment(tmp_path)

    assert main(configure_arguments()) == 0

    assert TrainingConfig.from_yaml(directory / "training.yaml") == training_config()


def test_training_configure_refuses_an_existing_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = create_initialized_experiment(tmp_path)
    (directory / "training.yaml").write_text("existing", encoding="utf-8")

    assert main(configure_arguments()) == 1

    assert "already configured" in capsys.readouterr().err
    assert (directory / "training.yaml").read_text(encoding="utf-8") == "existing"


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
    directory = create_initialized_experiment(tmp_path)
    arguments = configure_arguments()
    arguments[arguments.index("--lora-rank") + 1] = "0"

    assert main(arguments) == 1

    assert "LoRA rank must be positive" in capsys.readouterr().err
    assert not (directory / "training.yaml").exists()


def test_training_start_selects_the_training_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _, plan = create_ready_experiment(tmp_path)
    trained_ids, checkpoints = install_fake_training(monkeypatch)

    assert (
        main(["training", "start", "study", "--training-pool", "--device", "cpu"]) == 0
    )

    assert trained_ids == [plan.training_pool]
    assert checkpoints == [Path("experiments/study/checkpoints/training-pool")]


def test_training_start_selects_one_named_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _, plan = create_ready_experiment(tmp_path)
    trained_ids, checkpoints = install_fake_training(monkeypatch)

    assert (
        main(
            ["training", "start", "study", "--subset", "subset-0001", "--device", "cpu"]
        )
        == 0
    )

    assert trained_ids == [plan.subsets["subset-0001"]]
    assert checkpoints == [Path("experiments/study/checkpoints/subsets/subset-0001")]


def test_training_start_all_skips_complete_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, plan = create_ready_experiment(tmp_path)
    complete = directory / "checkpoints" / "subsets" / "subset-0000"
    write_complete_checkpoint(complete)
    trained_ids, checkpoints = install_fake_training(monkeypatch)

    assert (
        main(
            [
                "training",
                "start",
                "study",
                "--all-training-sets",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    assert trained_ids == [plan.training_pool, plan.subsets["subset-0001"]]
    assert checkpoints == [
        Path("experiments/study/checkpoints/training-pool"),
        Path("experiments/study/checkpoints/subsets/subset-0001"),
    ]
    assert "checkpoint already complete" in capsys.readouterr().out


def test_training_start_refuses_an_existing_single_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, _ = create_ready_experiment(tmp_path)
    write_complete_checkpoint(directory / "checkpoints" / "training-pool")

    assert (
        main(
            [
                "training",
                "start",
                "study",
                "--training-pool",
                "--device",
                "cpu",
            ]
        )
        == 1
    )

    assert "checkpoint already exists" in capsys.readouterr().err


def test_training_start_requires_training_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, _ = create_ready_experiment(tmp_path)
    (directory / "training.yaml").unlink()

    assert (
        main(
            [
                "training",
                "start",
                "study",
                "--training-pool",
                "--device",
                "cpu",
            ]
        )
        == 1
    )

    assert "training config" in capsys.readouterr().err


def test_training_start_rejects_unknown_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    create_ready_experiment(tmp_path)

    assert (
        main(
            [
                "training",
                "start",
                "study",
                "--subset",
                "missing",
                "--device",
                "cpu",
            ]
        )
        == 1
    )

    assert "unknown training subset" in capsys.readouterr().err


def test_training_start_requires_embeddings_for_encoded_speakers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    directory, plan = create_ready_experiment(tmp_path)
    UtteranceDataset(
        Utterance(
            id=identifier,
            speaker="unplanned-speaker",
            dialogue=identifier,
            text_ids=[1],
            audio_codes=[[2] * 16],
        )
        for identifier in [*plan.training_pool, *plan.validation_pool]
    ).to_jsonl(directory / "sampled_utterances_encoded.jsonl")

    assert (
        main(
            [
                "training",
                "start",
                "study",
                "--training-pool",
                "--device",
                "cpu",
            ]
        )
        == 1
    )

    assert "unplanned-speaker" in capsys.readouterr().err
