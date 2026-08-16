from __future__ import annotations

from pathlib import Path

import pytest

from tts_data_attribution.experiment import ExperimentConfig


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    config = ExperimentConfig(
        dataset=Path("data/processed/dailytalk_qwen3tts.jsonl"),
        model="qwen3-tts",
        model_path=Path("artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254"),
    )

    config.to_yaml(path)

    assert ExperimentConfig.from_yaml(path) == config
    assert path.read_text(encoding="utf-8") == (
        "dataset: data/processed/dailytalk_qwen3tts.jsonl\n"
        "model: qwen3-tts\n"
        "model_path: artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254\n"
    )


def test_missing_key_raises_key_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("dataset: a.jsonl\nmodel: qwen3-tts\n", encoding="utf-8")

    with pytest.raises(KeyError):
        ExperimentConfig.from_yaml(path)
