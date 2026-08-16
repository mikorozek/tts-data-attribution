from __future__ import annotations

from pathlib import Path

import pytest

from tts_data_attribution.experiment import ExperimentManifest


def manifest() -> ExperimentManifest:
    return ExperimentManifest(
        dataset=Path("data/processed/dailytalk_qwen3tts.jsonl"),
        audio_root=Path("data/raw/dailytalk"),
        model="qwen3-tts",
        model_path=Path("artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254"),
        training_pool_size=2000,
        subset_count=50,
        subset_size=1000,
        speaker_count=2,
        seed=1234,
    )


def test_manifest_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"

    manifest().to_yaml(path)

    assert ExperimentManifest.from_yaml(path) == manifest()
    assert path.read_text(encoding="utf-8") == (
        "audio_root: data/raw/dailytalk\n"
        "dataset: data/processed/dailytalk_qwen3tts.jsonl\n"
        "model: qwen3-tts\n"
        "model_path: artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254\n"
        "seed: 1234\n"
        "speaker_count: 2\n"
        "subset_count: 50\n"
        "subset_size: 1000\n"
        "training_pool_size: 2000\n"
    )


def test_missing_key_raises_key_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("dataset: a.jsonl\nmodel: qwen3-tts\n", encoding="utf-8")

    with pytest.raises(KeyError):
        ExperimentManifest.from_yaml(path)
