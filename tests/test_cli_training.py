from __future__ import annotations

from pathlib import Path

import pytest

from tts_data_attribution.cli.main import main
from tts_data_attribution.experiment import TrainingConfig


def configure_arguments(root: Path) -> list[str]:
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
        "--root",
        str(root),
    ]


def create_experiment(root: Path) -> Path:
    directory = root / "study"
    directory.mkdir()
    (directory / "manifest.yaml").write_text("manifest", encoding="utf-8")
    (directory / "plan.json").write_text("{}", encoding="utf-8")
    return directory


def test_training_configure_writes_valid_canonical_config(tmp_path: Path) -> None:
    directory = create_experiment(tmp_path)

    assert main(configure_arguments(tmp_path)) == 0

    assert TrainingConfig.from_yaml(directory / "training.yaml") == TrainingConfig(
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


def test_training_configure_refuses_an_existing_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = create_experiment(tmp_path)
    (directory / "training.yaml").write_text("existing", encoding="utf-8")

    assert main(configure_arguments(tmp_path)) == 1

    assert "already configured" in capsys.readouterr().err
    assert (directory / "training.yaml").read_text(encoding="utf-8") == "existing"


def test_training_configure_requires_an_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(configure_arguments(tmp_path)) == 1

    assert "run experiment init first" in capsys.readouterr().err


def test_training_configure_validates_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = create_experiment(tmp_path)
    arguments = configure_arguments(tmp_path)
    arguments[arguments.index("--lora-rank") + 1] = "0"

    assert main(arguments) == 1

    assert "LoRA rank must be positive" in capsys.readouterr().err
    assert not (directory / "training.yaml").exists()
