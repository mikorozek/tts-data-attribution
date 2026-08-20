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


def create_training_target(
    root: Path,
    training_set: str = "training-pool",
) -> tuple[Path, list[dict[str, object]]]:
    experiment = root / "experiments" / "study"
    experiment.mkdir(parents=True)
    ExperimentManifest(
        dataset="dailytalk",
        data_root=Path("dataset"),
        model="qwen3-tts",
        model_path=Path("model"),
        training_pool_size=2,
        validation_pool_size=0,
        subset_count=1,
        subset_size=1,
        speaker_count=1,
        seed=1,
    ).to_yaml(experiment / "manifest.yaml")
    plan = Plan(
        references={"speaker": "reference"},
        training_pool=["train-0", "train-1"],
        validation_pool=[],
        subsets={"subset-0000": ["train-0"]},
    )
    plan.to_json(experiment / "plan.json")
    UtteranceDataset(
        [
            Utterance(
                id=identifier,
                speaker="speaker",
                dialogue=identifier,
                text_ids=[1],
                audio_codes=[[2] * 16],
            )
            for identifier in plan.training_pool
        ]
    ).to_jsonl(experiment / "sampled_utterances_encoded.jsonl")
    torch.save({"speaker": torch.ones(4)}, experiment / "speaker_embeddings.pt")
    run_name = f"{training_set}-20260820T120000000000Z"
    run = experiment / "training-runs" / run_name
    target = run / "target"
    (target / "adapter").mkdir(parents=True)
    TrainingRunManifest(
        training_set=training_set,
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
        seed=7,
    ).to_yaml(run / "manifest.yaml")
    (target / "adapter" / "adapter_config.json").write_text("{}")
    (target / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    talker = TinyTalker()
    optimizer = AdamW(talker.parameters(), lr=2e-5)
    optimizer.zero_grad(set_to_none=True)
    talker(torch.ones(1, 3)).square().mean().backward()
    optimizer.step()
    torch.save(optimizer.state_dict(), target / "optimizer.pt")
    parameters: list[dict[str, object]] = [
        {"name": "lora_A", "shape": [2, 3], "dtype": "float32"},
        {"name": "lora_B", "shape": [4, 2], "dtype": "float32"},
    ]
    metadata = {
        "format_version": 1,
        "epoch": 3,
        "step": 1,
        "parameter_groups": [[parameter["name"] for parameter in parameters]],
        "parameters": parameters,
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return experiment, parameters


def init_arguments() -> list[str]:
    return [
        "projection",
        "init",
        "study",
        "two-sided-4",
        "--training-run",
        "training-pool-20260820T120000000000Z",
        "--output-dimension",
        "4",
        "--seed",
        "13",
    ]


def test_projection_init_saves_the_parameter_layout_and_random_matrices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment, parameters = create_training_target(tmp_path)

    assert main(init_arguments()) == 0

    directory = experiment / "trackstar" / "projections" / "two-sided-4"
    assert yaml.safe_load((directory / "manifest.yaml").read_text()) == {
        "type": "two-sided",
        "training_run": "training-pool-20260820T120000000000Z",
        "output_dimension": 4,
        "seed": 13,
        "parameters": [
            {"name": parameter["name"], "shape": parameter["shape"]}
            for parameter in parameters
        ],
    }
    matrices = torch.load(directory / "matrices.pt", weights_only=True)
    expected = TwoSidedRandomProjection(
        ((2, 3), (4, 2)),
        output_dimension=4,
        seed=13,
        device="cpu",
    )
    assert matrices.keys() == {"left_matrices", "right_matrices"}
    for actual, expected_matrix in zip(
        matrices["left_matrices"],
        expected.left_matrices,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected_matrix)
    for actual, expected_matrix in zip(
        matrices["right_matrices"],
        expected.right_matrices,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected_matrix)


def test_projection_apply_projects_the_training_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment, _ = create_training_target(tmp_path)
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
                [
                    [float(index + 1), 1.0, 2.0]
                    for index, _ in enumerate(utterances)
                ]
            )
        }

    def fake_objective(
        model: TinyModel,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return model.talker(batch["inputs"]).square().mean(dim=1)

    monkeypatch.setattr(
        projection_cli,
        "load_model",
        lambda *args, **kwargs: TinyModel(),
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
                "two-sided-4",
                "--training-pool",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    output = torch.load(
        experiment
        / "trackstar"
        / "projections"
        / "two-sided-4"
        / "projected"
        / "training-pool.pt",
        weights_only=True,
    )
    assert output["ids"] == ["train-0", "train-1"]
    assert output["projected_gradients"].shape == (2, 4)
    assert torch.isfinite(output["projected_gradients"]).all()


def test_projection_init_refuses_to_replace_an_existing_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment, _ = create_training_target(tmp_path)
    directory = experiment / "trackstar" / "projections" / "two-sided-4"
    directory.mkdir(parents=True)
    marker = directory / "marker"
    marker.write_text("existing")

    assert main(init_arguments()) == 1

    assert "already exists" in capsys.readouterr().err
    assert marker.read_text() == "existing"


def test_projection_init_rejects_a_subset_training_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment, _ = create_training_target(tmp_path, "subset-0000")
    arguments = init_arguments()
    arguments[arguments.index("training-pool-20260820T120000000000Z")] = (
        "subset-0000-20260820T120000000000Z"
    )

    assert main(arguments) == 1

    assert "requires a training-pool run" in capsys.readouterr().err
    assert not (experiment / "trackstar").exists()


def test_projection_init_requires_a_complete_training_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment, _ = create_training_target(tmp_path)
    target = next((experiment / "training-runs").iterdir()) / "target"
    (target / "optimizer.pt").unlink()

    assert main(init_arguments()) == 1

    assert "training target is incomplete" in capsys.readouterr().err
    assert not (experiment / "trackstar").exists()



@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("two-sided-4", "../projection", "projection name"),
        (
            "training-pool-20260820T120000000000Z",
            "../training-run",
            "training run name",
        ),
    ],
)
def test_projection_init_rejects_non_flat_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argument: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    arguments = init_arguments()
    arguments[arguments.index(argument)] = value

    assert main(arguments) == 1

    assert message in capsys.readouterr().err


def test_projection_init_validates_projection_dimension_and_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    create_training_target(tmp_path)
    arguments = init_arguments()
    arguments[arguments.index("4")] = "8"

    assert main(arguments) == 1
    assert "output dimension must be a square" in capsys.readouterr().err

    arguments = init_arguments()
    arguments[arguments.index("13")] = "-1"
    assert main(arguments) == 1
    assert "seed must not be negative" in capsys.readouterr().err
