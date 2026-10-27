from __future__ import annotations

import json
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
    TrainingParameter,
    TrainingRunManifest,
)


def write_checkpoint(path: Path, epoch: int = 2) -> None:
    (path / "adapter").mkdir(parents=True)
    (path / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "step": epoch,
                "training_loss": 1.0,
                "validation_loss": 2.0,
            }
        ),
        encoding="utf-8",
    )


def create_lds_experiment(
    root: Path, method: str = "trackstar-based"
) -> tuple[Path, Path]:
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
        Utterance(
            id=identifier,
            speaker="speaker",
            dialogue=identifier,
            text_ids=[1],
            audio_codes=[[2] * 16],
        )
        for identifier in plan.query_pool
    ).to_jsonl(experiment / "sampled_utterances_encoded.jsonl")
    torch.save({"speaker": torch.ones(4)}, experiment / "speaker_embeddings.pt")

    run = experiment / "training-runs" / "run-1"
    run.mkdir(parents=True)
    TrainingRunManifest(
        dtype="float32",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        learning_rate=2e-5,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        weight_decay=0.01,
        epochs=2,
        batch_size=2,
        seed=7,
        parameters=(TrainingParameter("layer", (2, 2), "float32"),),
        parameter_groups=(("layer",),),
    ).to_yaml(run / "manifest.yaml")
    write_checkpoint(run / "training-pool" / "checkpoints" / "epoch-0002")
    for subset_id in plan.subsets:
        write_checkpoint(run / "subsets" / subset_id / "checkpoints" / "epoch-0002")

    projection = run / "projections" / "two-sided-4"
    projection.mkdir(parents=True)
    (projection / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "two-sided",
                "training_checkpoint": "training-pool/checkpoints/epoch-0002",
                "output_dimension": 4,
                "seed": 13,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    attribution_path = projection / method / "attributions.pt"
    attribution_path.parent.mkdir()
    torch.save(
        torch.tensor([[3.0, 0.0], [2.0, 1.0], [1.0, 2.0]]),
        attribution_path,
    )
    return run, attribution_path


def install_fake_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    run: Path,
) -> None:
    def evaluate(
        experiment_directory: Path,
        experiment: ExperimentManifest,
        checkpoint_paths: list[Path],
        query_ids: list[str],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        assert experiment_directory.name == "study"
        assert experiment.model_path == Path("model")
        assert [path.parents[1].name for path in checkpoint_paths] == [
            "subset-0000",
            "subset-0001",
            "subset-0002",
        ]
        assert query_ids == ["query-0", "query-1"]
        assert batch_size == 2
        assert dtype == torch.float32
        assert device == torch.device("cpu")
        return torch.tensor([[3.0, 2.0], [2.0, 1.0], [1.0, 0.0]])

    monkeypatch.setattr(lds_cli, "_evaluate_subset_responses", evaluate)


def arguments(path: Path) -> list[str]:
    return ["lds", "compute", str(path), "--device", "cpu"]


@pytest.mark.parametrize("method", ["trackstar-based", "projected-cosine"])
def test_lds_compute_saves_only_the_scalar_next_to_the_attributions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    run, attribution_path = create_lds_experiment(tmp_path, method)
    install_fake_evaluator(monkeypatch, run)

    assert main(arguments(attribution_path)) == 0

    result = torch.load(attribution_path.parent / "lds.pt", weights_only=True)
    assert isinstance(result, float)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_lds_uses_the_projection_checkpoint_epoch_for_every_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    run, attribution_path = create_lds_experiment(tmp_path)
    (
        run / "subsets" / "subset-0001" / "checkpoints" / "epoch-0002" / "metrics.json"
    ).unlink()

    assert main(arguments(attribution_path)) == 1

    assert "subset checkpoints are missing for: subset-0001" in capsys.readouterr().err


def test_lds_rejects_a_subset_checkpoint_from_another_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    run, attribution_path = create_lds_experiment(tmp_path)
    metrics_path = (
        run / "subsets" / "subset-0001" / "checkpoints" / "epoch-0002" / "metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["epoch"] = 1
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    assert main(arguments(attribution_path)) == 1

    assert "checkpoint metrics are invalid for subset-0001" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (torch.ones(2, 2), "dimensions or dtype"),
        (torch.ones(3, 2, dtype=torch.int64), "dimensions or dtype"),
        (torch.tensor([[float("nan"), 0.0], [1.0, 2.0], [2.0, 1.0]]), "must be finite"),
    ],
)
def test_lds_rejects_invalid_attribution_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    replacement: torch.Tensor,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    _, attribution_path = create_lds_experiment(tmp_path)
    torch.save(replacement, attribution_path)

    assert main(arguments(attribution_path)) == 1

    assert message in capsys.readouterr().err


def test_lds_rejects_an_attribution_outside_the_method_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _, attribution_path = create_lds_experiment(tmp_path)
    misplaced = attribution_path.parents[2] / "attributions.pt"
    misplaced.write_bytes(attribution_path.read_bytes())

    assert main(arguments(misplaced)) == 1

    assert "must belong to a training-run projection method" in capsys.readouterr().err


def test_lds_refuses_to_replace_an_existing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    run, attribution_path = create_lds_experiment(tmp_path)
    install_fake_evaluator(monkeypatch, run)
    output = attribution_path.parent / "lds.pt"
    torch.save(0.5, output)

    assert main(arguments(attribution_path)) == 1

    assert "already exists" in capsys.readouterr().err
    assert torch.load(output, weights_only=True) == 0.5


def test_lds_rejects_an_invalid_subset_response_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _, attribution_path = create_lds_experiment(tmp_path)
    monkeypatch.setattr(
        lds_cli,
        "_evaluate_subset_responses",
        lambda *args, **kwargs: torch.ones(2, 2),
    )

    assert main(arguments(attribution_path)) == 1

    assert "observed response dimensions" in capsys.readouterr().err


def test_lds_requires_the_attribution_path() -> None:
    with pytest.raises(SystemExit) as error:
        main(["lds", "compute"])

    assert error.value.code == 2


def test_lds_evaluates_each_subset_with_the_qwen_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_directory = tmp_path / "study"
    experiment_directory.mkdir()
    queries = UtteranceDataset(
        Utterance(
            id=f"query-{index}",
            speaker="speaker",
            dialogue=f"dialogue-{index}",
            text_ids=[index],
            audio_codes=[[2] * 16],
        )
        for index in range(2)
    )
    queries.to_jsonl(experiment_directory / "sampled_utterances_encoded.jsonl")
    torch.save(
        {"speaker": torch.ones(4)},
        experiment_directory / "speaker_embeddings.pt",
    )
    experiment = ExperimentManifest(
        dataset="dailytalk",
        data_root=Path("dataset"),
        model="qwen3-tts",
        model_path=Path("model"),
        training_pool_size=3,
        validation_pool_size=0,
        query_pool_size=2,
        subset_count=2,
        subset_size=1,
        speaker_count=1,
        seed=1,
    )

    class TinyTalker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter_index = 0

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.talker = TinyTalker()

    loaded_models = []

    def fake_load_model(*args: object, **kwargs: object) -> TinyModel:
        model = TinyModel()
        loaded_models.append(model)
        return model

    class FakePeftModel:
        @staticmethod
        def from_pretrained(
            talker: TinyTalker,
            path: Path,
            *,
            is_trainable: bool,
        ) -> TinyTalker:
            assert not is_trainable
            talker.adapter_index = int(path.parents[2].name[-1])
            return talker

    def fake_collate(
        utterances: list[Utterance],
        speaker_embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert "speaker" in speaker_embeddings
        return {
            "query_indices": torch.tensor(
                [utterance.text_ids[0] for utterance in utterances]
            )
        }

    losses = torch.tensor([[3.0, 1.0], [2.0, 4.0]])

    def fake_objective(
        model: TinyModel,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        assert not model.training
        assert not torch.is_grad_enabled()
        return losses[model.talker.adapter_index, batch["query_indices"]]

    monkeypatch.setattr(lds_cli, "load_model", fake_load_model)
    monkeypatch.setattr(lds_cli, "PeftModel", FakePeftModel)
    monkeypatch.setattr(lds_cli, "collate", fake_collate)
    monkeypatch.setattr(lds_cli, "objective", fake_objective)
    checkpoints = [
        tmp_path
        / "run"
        / "subsets"
        / f"subset-000{index}"
        / "checkpoints"
        / "epoch-0002"
        for index in range(2)
    ]

    responses = lds_cli._evaluate_subset_responses(
        experiment_directory,
        experiment,
        checkpoints,
        ["query-0", "query-1"],
        batch_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(responses, -losses)
    assert len(loaded_models) == 2
