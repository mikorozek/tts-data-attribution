from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from tts_data_attribution.experiment import ExperimentManifest, Plan


@dataclass(frozen=True)
class SourceRecord:
    id: str
    speaker: str
    dialogue: str


def dataset(
    per_speaker: int = 12, speakers: tuple[str, ...] = ("0", "1")
) -> list[SourceRecord]:
    return [
        SourceRecord(
            id=f"{speaker}-{index}",
            speaker=speaker,
            dialogue=str(index),
        )
        for speaker in speakers
        for index in range(per_speaker)
    ]


def manifest(**overrides: object) -> ExperimentManifest:
    values: dict[str, Any] = dict(
        dataset="dailytalk",
        data_root=Path("raw"),
        model="qwen3-tts",
        model_path=Path("model"),
        training_pool_size=8,
        validation_pool_size=2,
        subset_count=3,
        subset_size=4,
        speaker_count=2,
        seed=7,
    )
    values.update(overrides)
    return ExperimentManifest(**values)


def test_plan_has_the_requested_shape() -> None:
    plan = Plan.sample(manifest(), dataset())

    assert sorted(plan.references) == ["0", "1"]
    assert len(plan.training_pool) == 8
    assert len(plan.validation_pool) == 2
    assert len(plan.subsets) == 3
    assert all(len(subset) == 4 for subset in plan.subsets)


def test_references_are_one_per_speaker_and_never_in_the_pool() -> None:
    plan = Plan.sample(manifest(), dataset())

    for speaker, reference in plan.references.items():
        assert reference.startswith(f"{speaker}-")
        assert reference not in plan.training_pool
        assert reference not in plan.validation_pool


def test_pools_and_references_are_dialogue_disjoint() -> None:
    plan = Plan.sample(manifest(), dataset())
    groups = [
        {identifier.split("-")[1] for identifier in plan.references.values()},
        {identifier.split("-")[1] for identifier in plan.training_pool},
        {identifier.split("-")[1] for identifier in plan.validation_pool},
    ]

    for index, group in enumerate(groups):
        assert all(group.isdisjoint(other) for other in groups[index + 1 :])


def test_subsets_are_drawn_from_the_pool_without_repeats() -> None:
    plan = Plan.sample(manifest(), dataset())

    for subset in plan.subsets:
        assert set(subset) <= set(plan.training_pool)
        assert len(set(subset)) == len(subset)


def test_only_the_first_speakers_take_part() -> None:
    plan = Plan.sample(
        manifest(speaker_count=1, training_pool_size=5),
        dataset(speakers=("0", "1", "2")),
    )

    assert list(plan.references) == ["0"]
    assert all(utterance_id.startswith("0-") for utterance_id in plan.training_pool)


def test_same_manifest_gives_the_same_plan() -> None:
    assert Plan.sample(manifest(), dataset()) == Plan.sample(manifest(), dataset())


def test_another_seed_gives_another_plan() -> None:
    assert Plan.sample(manifest(), dataset()) != Plan.sample(
        manifest(seed=8), dataset()
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"speaker_count": 3}, "speaker_count 3 exceeds the 2 speakers"),
        ({"training_pool_size": 25}, "training_pool_size 25 exceeds"),
        ({"validation_pool_size": 25}, "validation_pool_size 25 exceeds"),
        ({"subset_size": 9}, "subset_size 9 exceeds training_pool_size 8"),
    ],
)
def test_impossible_sampling_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Plan.sample(manifest(**overrides), dataset())


def test_plan_round_trips_through_json(tmp_path: Path) -> None:
    plan = Plan.sample(manifest(), dataset())
    path = tmp_path / "plan.json"

    plan.to_json(path)

    assert Plan.from_json(path) == plan
    assert replace(plan, subsets=[]) != plan
