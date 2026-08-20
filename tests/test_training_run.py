from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tts_data_attribution.experiment import TrainingRunManifest


def manifest() -> TrainingRunManifest:
    return TrainingRunManifest(
        training_set="training-pool",
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


def test_training_run_manifest_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"

    manifest().to_yaml(path)

    assert TrainingRunManifest.from_yaml(path) == manifest()


def test_example_training_run_manifest_is_valid() -> None:
    example = Path(__file__).parents[1] / "training-run.example.yaml"

    assert TrainingRunManifest.from_yaml(example) == manifest()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_set", "", "training set"),
        ("dtype", "float16", "dtype"),
        ("lora_rank", 0, "rank"),
        ("lora_alpha", 0, "alpha"),
        ("lora_dropout", 1.0, "dropout"),
        ("learning_rate", 0.0, "learning rate"),
        ("adam_betas", (0.9, 1.0), "betas"),
        ("adam_epsilon", 0.0, "epsilon"),
        ("weight_decay", -0.1, "weight decay"),
        ("epochs", 0, "epochs"),
        ("batch_size", 0, "batch size"),
        ("seed", -1, "seed"),
    ],
)
def test_training_run_manifest_rejects_invalid_values(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(manifest(), **{field: value})


def test_training_run_manifest_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain exactly"):
        TrainingRunManifest.from_yaml(path)



def test_training_run_manifest_rejects_wrong_scalar_types(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    manifest().to_yaml(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("rank: 8", "rank: 8.5"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lora rank must be an integer"):
        TrainingRunManifest.from_yaml(path)
