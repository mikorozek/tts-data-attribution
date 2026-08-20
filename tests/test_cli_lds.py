from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from tts_data_attribution.cli import lds as lds_cli
from tts_data_attribution.cli.main import main
from tts_data_attribution.dataset import Utterance, UtteranceDataset
from tts_data_attribution.experiment import (
    ExperimentManifest,
    Plan,
    TrainingRunManifest,
)


class TinyTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.subset_index = 0


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.talker = TinyTalker()


def training_manifest(training_set: str) -> TrainingRunManifest:
    return TrainingRunManifest(
        training_set=training_set,
        dtype="bfloat16",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        learning_rate=6e-5,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        weight_decay=0.01,
        epochs=3,
        batch_size=2,
        seed=7,
    )


def create_complete_target(run_directory: Path) -> None:
    target = run_directory / "target"
    adapter = target / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    torch.save({}, target / "optimizer.pt")
    (target / "metadata.json").write_text("{}", encoding="utf-8")


def create_lds_experiment(root: Path) -> Path:
    experiment = root / "experiments" / "study"
    experiment.mkdir(parents=True)
    ExperimentManifest(
        dataset="dailytalk",
        data_root=Path("dataset"),
        model="qwen3-tts",
        model_path=Path("model"),
        training_pool_size=3,
        validation_pool_size=0,
        query_pool_size=2,
        subset_count=3,
        subset_size=1,
        speaker_count=1,
        seed=1,
    ).to_yaml(experiment / "manifest.yaml")
    plan = Plan(
        references={"speaker": "reference"},
        training_pool=["train-0", "train-1", "train-2"],
        validation_pool=[],
        query_pool=["query-0", "query-1"],
        subsets={
            "subset-0000": ["train-0"],
            "subset-0001": ["train-1"],
            "subset-0002": ["train-2"],
        },
    )
    plan.to_json(experiment / "plan.json")
    UtteranceDataset(
        [
            Utterance(
                id=identifier,
                speaker="speaker",
                dialogue=identifier,
                text_ids=[index],
                audio_codes=[[2] * 16],
            )
            for index, identifier in enumerate([*plan.query_pool, *plan.training_pool])
        ]
    ).to_jsonl(experiment / "sampled_utterances_encoded.jsonl")
    torch.save({"speaker": torch.ones(4)}, experiment / "speaker_embeddings.pt")

    full_run_name = "training-pool-20260820T120000000000Z"
    full_run = experiment / "training-runs" / full_run_name
    full_run.mkdir(parents=True)
    training_manifest("training-pool").to_yaml(full_run / "manifest.yaml")

    for index, subset_id in enumerate(plan.subsets):
        run = experiment / "training-runs" / f"{subset_id}-20260820T12000{index}000000Z"
        run.mkdir(parents=True)
        training_manifest(subset_id).to_yaml(run / "manifest.yaml")
        create_complete_target(run)

    incomplete = experiment / "training-runs" / "subset-0000-20260820T130000000000Z"
    incomplete.mkdir(parents=True)
    (incomplete / "manifest.yaml").write_text("invalid: [", encoding="utf-8")

    projection = experiment / "trackstar" / "projections" / "two-sided-4"
    projection.mkdir(parents=True)
    (projection / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "two-sided",
                "training_run": full_run_name,
                "output_dimension": 4,
                "seed": 13,
                "parameters": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "task_weight": 0.5,
            "training_ids": plan.training_pool,
            "query_ids": plan.query_pool,
            "attributions": torch.tensor([[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]]),
        },
        projection / "attributions.pt",
    )
    return experiment


def arguments() -> list[str]:
    return [
        "lds",
        "compute",
        "study",
        "--projection",
        "two-sided-4",
        "--device",
        "cpu",
    ]


def install_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loss_table = torch.tensor(
        [[3.0, 1.0], [2.0, 3.0], [1.0, 2.0]],
    )

    def fake_evaluator(
        model_path: Path,
        adapter_directories: list[Path],
        query_utterances: UtteranceDataset,
        speaker_embeddings: dict[str, torch.Tensor],
        **options: object,
    ) -> torch.Tensor:
        assert model_path == Path("model")
        assert len(adapter_directories) == 3
        assert len(query_utterances) == 2
        assert "speaker" in speaker_embeddings
        callback = options["adapter_evaluated"]
        assert callable(callback)
        for completed in range(1, 4):
            callback(completed, 3)
        return -loss_table

    monkeypatch.setitem(
        lds_cli.EXPERIMENT_ADAPTER_RESPONSE_EVALUATORS,
        "qwen3-tts",
        lambda: fake_evaluator,
    )


def test_lds_compute_discovers_completed_subsets_and_saves_detailed_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment = create_lds_experiment(tmp_path)
    install_fake_model(monkeypatch)

    assert main(arguments()) == 0

    result = torch.load(
        experiment / "trackstar" / "projections" / "two-sided-4" / "lds.pt",
        weights_only=True,
    )
    assert result["projection_name"] == "two-sided-4"
    assert result["response"] == "negative_objective"
    assert result["configuration"] == {
        "correlation": "spearman",
        "aggregation": "mean",
        "uncertainty": "subset-bootstrap",
        "bootstrap_samples": 1000,
        "confidence_level": 0.95,
        "seed": 0,
    }
    assert result["training_ids"] == ["train-0", "train-1", "train-2"]
    assert result["query_ids"] == ["query-0", "query-1"]
    assert result["subset_ids"] == ["subset-0000", "subset-0001", "subset-0002"]
    assert len(result["training_run_names"]) == 3
    torch.testing.assert_close(result["membership"], torch.eye(3, dtype=torch.bool))
    torch.testing.assert_close(
        result["query_losses"],
        torch.tensor([[3.0, 1.0], [2.0, 3.0], [1.0, 2.0]]),
    )
    torch.testing.assert_close(
        result["observed_responses"],
        torch.tensor([[-3.0, -1.0], [-2.0, -3.0], [-1.0, -2.0]]),
    )
    torch.testing.assert_close(
        result["predicted_responses"],
        torch.tensor([[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result["per_query_correlations"], torch.ones(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["per_query_lds"], torch.ones(2, dtype=torch.float64)
    )
    assert result["mean_lds"] == pytest.approx(1.0)
    assert result["aggregate_lds"] == pytest.approx(1.0)
    torch.testing.assert_close(
        result["bootstrap_aggregate_lds"],
        torch.ones(1000, dtype=torch.float64),
    )
    assert result["confidence_interval"] == pytest.approx((1.0, 1.0))


def test_lds_compute_rejects_multiple_completed_runs_for_one_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment = create_lds_experiment(tmp_path)
    duplicate = experiment / "training-runs" / "subset-0000-20260820T140000000000Z"
    duplicate.mkdir(parents=True)
    training_manifest("subset-0000").to_yaml(duplicate / "manifest.yaml")
    create_complete_target(duplicate)

    assert main(arguments()) == 1

    assert "multiple completed runs for subset-0000" in capsys.readouterr().err
    assert not (
        experiment / "trackstar" / "projections" / "two-sided-4" / "lds.pt"
    ).exists()


def test_lds_compute_requires_at_least_two_completed_subsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment = create_lds_experiment(tmp_path)
    for run in (experiment / "training-runs").iterdir():
        if not (run / "target" / "optimizer.pt").is_file():
            continue
        manifest = TrainingRunManifest.from_yaml(run / "manifest.yaml")
        if manifest.training_set in {"subset-0001", "subset-0002"}:
            for path in (run / "target").rglob("*"):
                if path.is_file():
                    path.unlink()

    assert main(arguments()) == 1

    assert "at least two completed subset" in capsys.readouterr().err


def test_lds_compute_rejects_a_mismatched_subset_training_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment = create_lds_experiment(tmp_path)
    run = next(
        run
        for run in (experiment / "training-runs").iterdir()
        if run.name.startswith("subset-0001-")
    )
    reference = training_manifest("subset-0001")
    mismatched = replace(reference, seed=reference.seed + 1)
    mismatched.to_yaml(run / "manifest.yaml")

    assert main(arguments()) == 1

    assert "differs from the projection training run" in capsys.readouterr().err
    assert not (
        experiment / "trackstar" / "projections" / "two-sided-4" / "lds.pt"
    ).exists()


@pytest.mark.parametrize("unsafe_name", ["../study", "/tmp/study"])
def test_lds_compute_rejects_an_unsafe_experiment_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    unsafe_name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    command = arguments()
    command[command.index("study")] = unsafe_name

    assert main(command) == 1

    assert "experiment name must be a single path component" in capsys.readouterr().err


def test_lds_compute_rejects_an_unsafe_persisted_training_run_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment = create_lds_experiment(tmp_path)
    projection_manifest = (
        experiment / "trackstar" / "projections" / "two-sided-4" / "manifest.yaml"
    )
    manifest = yaml.safe_load(projection_manifest.read_text(encoding="utf-8"))
    manifest["training_run"] = "../outside"
    projection_manifest.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    assert main(arguments()) == 1

    assert "single path component" in capsys.readouterr().err
