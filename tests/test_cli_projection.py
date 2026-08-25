from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.optim import AdamW

from tts_data_attribution.cli import projection as projection_cli
from tts_data_attribution.cli.main import main
from tts_data_attribution.dataset import Utterance, UtteranceDataset
from tts_data_attribution.experiment import (
    ExperimentManifest,
    Plan,
    TrainingParameter,
    TrainingRunManifest,
)
from tts_data_attribution.trackstar import TwoSidedRandomProjection


class TinyTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(2, 3))
        self.lora_B = nn.Parameter(torch.randn(4, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs @ self.lora_A.T) @ self.lora_B.T


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.talker = TinyTalker()


def create_checkpoint(root: Path) -> tuple[Path, Path]:
    experiment = root / "experiments" / "study"
    experiment.mkdir(parents=True)
    ExperimentManifest(
        dataset="dailytalk",
        data_root=Path("dataset"),
        model="qwen3-tts",
        model_path=Path("model"),
        training_pool_size=2,
        validation_pool_size=0,
        query_pool_size=1,
        subset_count=1,
        subset_size=1,
        speaker_count=1,
        seed=1,
    ).to_yaml(experiment / "manifest.yaml")
    plan = Plan(
        references={"speaker": "reference"},
        training_pool=["train-0", "train-1"],
        validation_pool=[],
        query_pool=["query-0"],
        subsets={"subset-0000": ["train-0"]},
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
        for identifier in [*plan.training_pool, *plan.query_pool]
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
        epochs=3,
        batch_size=1,
        seed=7,
        parameters=(
            TrainingParameter("lora_A", (2, 3), "float32"),
            TrainingParameter("lora_B", (4, 2), "float32"),
        ),
        parameter_groups=(("lora_A", "lora_B"),),
    ).to_yaml(run / "manifest.yaml")

    checkpoint = run / "training-pool" / "checkpoints" / "epoch-0002"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    talker = TinyTalker()
    optimizer = AdamW(talker.parameters(), lr=2e-5)
    optimizer.zero_grad(set_to_none=True)
    talker(torch.ones(1, 3)).square().mean().backward()
    optimizer.step()
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    (checkpoint / "metrics.json").write_text(
        json.dumps(
            {
                "epoch": 2,
                "step": 1,
                "training_loss": 1.0,
                "validation_loss": 2.0,
            }
        ),
        encoding="utf-8",
    )
    return experiment, run


def init_arguments(
    checkpoint: str = "training-pool/checkpoints/epoch-0002",
) -> list[str]:
    return [
        "projection",
        "init",
        "study",
        "run-1",
        "two-sided-4",
        "--training-checkpoint",
        checkpoint,
        "--output-dimension",
        "4",
        "--seed",
        "13",
    ]


def test_projection_init_uses_the_run_layout_and_saves_random_matrices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _, run = create_checkpoint(tmp_path)

    assert main(init_arguments()) == 0

    directory = run / "projections" / "two-sided-4"
    assert yaml.safe_load((directory / "manifest.yaml").read_text()) == {
        "type": "two-sided",
        "training_checkpoint": "training-pool/checkpoints/epoch-0002",
        "output_dimension": 4,
        "seed": 13,
    }
    matrices = torch.load(directory / "matrices.pt", weights_only=True)
    expected = TwoSidedRandomProjection(
        ((2, 3), (4, 2)),
        output_dimension=4,
        seed=13,
        device="cpu",
    )
    for actual, expected_matrix in zip(
        matrices["left_matrices"], expected.left_matrices, strict=True
    ):
        torch.testing.assert_close(actual, expected_matrix)
    for actual, expected_matrix in zip(
        matrices["right_matrices"], expected.right_matrices, strict=True
    ):
        torch.testing.assert_close(actual, expected_matrix)


@pytest.mark.parametrize(
    ("selection", "filename", "identifiers"),
    [
        ("--training-pool", "training-pool.pt", ["train-0", "train-1"]),
        ("--query-pool", "query-pool.pt", ["query-0"]),
    ],
)
def test_projection_apply_projects_an_experiment_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    filename: str,
    identifiers: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _, run = create_checkpoint(tmp_path)
    assert main(init_arguments()) == 0

    class FakePeftModel:
        @staticmethod
        def from_pretrained(
            model: nn.Module,
            path: Path,
            *,
            is_trainable: bool,
        ) -> nn.Module:
            assert path.name == "adapter"
            assert is_trainable
            return model

    def fake_collate(
        utterances: list[Utterance],
        speaker_embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert "speaker" in speaker_embeddings
        return {
            "inputs": torch.tensor(
                [[float(index + 1), 1.0, 2.0] for index, _ in enumerate(utterances)]
            )
        }

    def fake_objective(
        model: TinyModel,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return model.talker(batch["inputs"]).square().mean(dim=1)

    monkeypatch.setattr(
        projection_cli, "load_model", lambda *args, **kwargs: TinyModel()
    )
    monkeypatch.setattr(projection_cli, "PeftModel", FakePeftModel)
    monkeypatch.setattr(projection_cli, "collate", fake_collate)
    monkeypatch.setattr(projection_cli, "objective", fake_objective)

    assert (
        main(
            [
                "projection",
                "apply",
                "study",
                "run-1",
                "two-sided-4",
                selection,
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    output = torch.load(
        run / "projections" / "two-sided-4" / "projected" / filename,
        weights_only=True,
    )
    assert output["ids"] == identifiers
    assert output["projected_gradients"].shape == (len(identifiers), 4)
    assert torch.isfinite(output["projected_gradients"]).all()

    output_path = run / "projections" / "two-sided-4" / "projected" / filename
    original_bytes = output_path.read_bytes()
    assert (
        main(
            [
                "projection",
                "apply",
                "study",
                "run-1",
                "two-sided-4",
                selection,
                "--device",
                "cpu",
            ]
        )
        == 1
    )
    assert output_path.read_bytes() == original_bytes


def test_projection_init_refuses_to_replace_an_existing_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _, run = create_checkpoint(tmp_path)
    directory = run / "projections" / "two-sided-4"
    directory.mkdir(parents=True)

    assert main(init_arguments()) == 1

    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "checkpoint",
    [
        "subsets/subset-0000/checkpoints/epoch-0002",
        "../outside",
        "/tmp/outside",
        "training-pool/checkpoints/latest",
    ],
)
def test_projection_init_requires_an_explicit_pool_epoch_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    checkpoint: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    create_checkpoint(tmp_path)

    assert main(init_arguments(checkpoint)) == 1

    error = capsys.readouterr().err
    assert "training-pool/checkpoints/epoch-NNNN" in error or "epoch name" in error


def test_projection_init_requires_a_complete_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _, run = create_checkpoint(tmp_path)
    (run / "training-pool" / "checkpoints" / "epoch-0002" / "optimizer.pt").unlink()

    assert main(init_arguments()) == 1

    assert "training checkpoint is incomplete" in capsys.readouterr().err


def test_projection_init_requires_a_resolved_parameter_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _, run = create_checkpoint(tmp_path)
    manifest = TrainingRunManifest.from_yaml(run / "manifest.yaml")
    TrainingRunManifest(
        dtype=manifest.dtype,
        lora_rank=manifest.lora_rank,
        lora_alpha=manifest.lora_alpha,
        lora_dropout=manifest.lora_dropout,
        learning_rate=manifest.learning_rate,
        adam_betas=manifest.adam_betas,
        adam_epsilon=manifest.adam_epsilon,
        weight_decay=manifest.weight_decay,
        epochs=manifest.epochs,
        batch_size=manifest.batch_size,
        seed=manifest.seed,
    ).to_yaml(run / "manifest.yaml")

    assert main(init_arguments()) == 1

    assert "parameter layout is missing" in capsys.readouterr().err


def test_projection_init_validates_dimension_and_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    create_checkpoint(tmp_path)
    arguments = init_arguments()
    arguments[arguments.index("4")] = "8"

    assert main(arguments) == 1
    assert "output dimension must be a square" in capsys.readouterr().err

    arguments = init_arguments()
    arguments[arguments.index("13")] = "-1"
    assert main(arguments) == 1
    assert "seed must not be negative" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lr", 3e-5, "lr differs"),
        ("betas", (0.8, 0.99), "betas differs"),
        ("eps", 1e-6, "eps differs"),
        ("weight_decay", 0.2, "weight_decay differs"),
        ("maximize", True, "maximize differs"),
        ("foreach", False, "foreach differs"),
        ("decoupled_weight_decay", False, "decoupled_weight_decay differs"),
    ],
)
def test_optimizer_checkpoint_must_match_the_run_recipe(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    _, run = create_checkpoint(tmp_path)
    state_path = run / "training-pool/checkpoints/epoch-0002/optimizer.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["param_groups"][0][field] = value
    torch.save(state, state_path)
    talker = TinyTalker()
    parameters = tuple(talker.named_parameters())
    manifest = TrainingRunManifest.from_yaml(run / "manifest.yaml")

    with pytest.raises(ValueError, match=message):
        projection_cli._load_adamw_optimizer(parameters, manifest, state_path, 1)


def test_optimizer_checkpoint_must_match_parameter_order_and_step(
    tmp_path: Path,
) -> None:
    _, run = create_checkpoint(tmp_path)
    state_path = run / "training-pool/checkpoints/epoch-0002/optimizer.pt"
    manifest = TrainingRunManifest.from_yaml(run / "manifest.yaml")
    talker = TinyTalker()
    parameters = tuple(talker.named_parameters())
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["param_groups"][0]["params"] = [1, 0]
    torch.save(state, state_path)

    with pytest.raises(ValueError, match="parameter order differs"):
        projection_cli._load_adamw_optimizer(parameters, manifest, state_path, 1)

    state["param_groups"][0]["params"] = [0, 1]
    state["state"][0]["step"] = torch.tensor(2.0)
    torch.save(state, state_path)
    with pytest.raises(ValueError, match="step differs from checkpoint metrics"):
        projection_cli._load_adamw_optimizer(parameters, manifest, state_path, 1)


@pytest.mark.parametrize(
    "metrics",
    [
        [],
        {
            "epoch": 2,
            "step": 1,
            "training_loss": 1.0,
            "validation_loss": 2.0,
            "extra": 3,
        },
        {
            "epoch": 2,
            "step": 0,
            "training_loss": 1.0,
            "validation_loss": 2.0,
        },
        {
            "epoch": 4,
            "step": 1,
            "training_loss": 1.0,
            "validation_loss": 2.0,
        },
        {
            "epoch": 2,
            "step": 1,
            "training_loss": float("nan"),
            "validation_loss": 2.0,
        },
    ],
)
def test_projection_init_rejects_invalid_checkpoint_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    metrics: object,
) -> None:
    monkeypatch.chdir(tmp_path)
    _, run = create_checkpoint(tmp_path)
    metrics_path = run / "training-pool/checkpoints/epoch-0002/metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    assert main(init_arguments()) == 1

    assert "cannot read training checkpoint" in capsys.readouterr().err


def test_optimizer_checkpoint_rejects_non_floating_moments(tmp_path: Path) -> None:
    _, run = create_checkpoint(tmp_path)
    state_path = run / "training-pool/checkpoints/epoch-0002/optimizer.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["state"][0]["exp_avg_sq"] = torch.ones_like(
        state["state"][0]["exp_avg_sq"],
        dtype=torch.int64,
    )
    torch.save(state, state_path)
    talker = TinyTalker()
    manifest = TrainingRunManifest.from_yaml(run / "manifest.yaml")

    with pytest.raises(ValueError, match="exp_avg_sq is invalid"):
        projection_cli._load_adamw_optimizer(
            tuple(talker.named_parameters()),
            manifest,
            state_path,
            1,
        )
