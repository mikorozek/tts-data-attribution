from __future__ import annotations

from pathlib import Path

import pytest

from tts_data_attribution.experiment import ExperimentManifest


def manifest() -> ExperimentManifest:
    return ExperimentManifest(
        dataset="dailytalk",
        data_root=Path("data/raw/dailytalk"),
        training_pool_size=2000,
        validation_pool_size=200,
        query_pool_size=100,
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
        "data_root: data/raw/dailytalk\n"
        "dataset: dailytalk\n"
        "query_pool_size: 100\n"
        "seed: 1234\n"
        "speaker_count: 2\n"
        "subset_count: 50\n"
        "subset_size: 1000\n"
        "training_pool_size: 2000\n"
        "validation_pool_size: 200\n"
    )


def test_missing_key_raises_key_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("dataset: dailytalk\n", encoding="utf-8")

    with pytest.raises(KeyError):
        ExperimentManifest.from_yaml(path)
