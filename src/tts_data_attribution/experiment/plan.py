from __future__ import annotations

import json
import random
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Self

from .manifest import ExperimentManifest


class _SourceRecord(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def speaker(self) -> str: ...

    @property
    def dialogue(self) -> str: ...


def _sample_dialogue_disjoint_pool(
    rng: random.Random,
    records: Sequence[_SourceRecord],
    size: int,
    name: str,
) -> tuple[list[str], set[str]]:
    if size < 0:
        raise ValueError(f"{name}_size must not be negative")
    if size > len(records):
        raise ValueError(
            f"{name}_size {size} exceeds the {len(records)} candidate utterances"
        )
    selected = rng.sample(records, size)
    return (
        sorted(record.id for record in selected),
        {record.dialogue for record in selected},
    )


@dataclass(frozen=True)
class Plan:
    references: dict[str, str]
    training_pool: list[str]
    validation_pool: list[str]
    subsets: dict[str, list[str]]

    def __post_init__(self) -> None:
        if not isinstance(self.references, dict) or not all(
            isinstance(speaker, str) and isinstance(identifier, str)
            for speaker, identifier in self.references.items()
        ):
            raise ValueError("references must map speaker names to utterance IDs")
        for name, identifiers in (
            ("training_pool", self.training_pool),
            ("validation_pool", self.validation_pool),
        ):
            if not isinstance(identifiers, list) or not all(
                isinstance(identifier, str) for identifier in identifiers
            ):
                raise ValueError(f"{name} must be a list of utterance IDs")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{name} utterance IDs must be unique")
        if set(self.training_pool) & set(self.validation_pool):
            raise ValueError("training and validation pools must be disjoint")
        if not isinstance(self.subsets, dict):
            raise ValueError("subsets must map names to utterance ID lists")
        expected_subset_names = {
            f"subset-{index:04d}" for index in range(len(self.subsets))
        }
        if set(self.subsets) != expected_subset_names:
            raise ValueError("subset names must be consecutive stable IDs")
        for name, identifiers in self.subsets.items():
            if (
                not isinstance(name, str)
                or not isinstance(identifiers, list)
                or not all(isinstance(identifier, str) for identifier in identifiers)
            ):
                raise ValueError("subsets must map names to utterance ID lists")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"subset {name} utterance IDs must be unique")
            if not set(identifiers) <= set(self.training_pool):
                raise ValueError(f"subset {name} must contain only training pool IDs")

    @classmethod
    def sample(
        cls, manifest: ExperimentManifest, dataset: Collection[_SourceRecord]
    ) -> Self:
        rng = random.Random(manifest.seed)
        records = sorted(dataset, key=lambda utterance: utterance.id)
        speakers = sorted({utterance.speaker for utterance in records})
        if manifest.speaker_count > len(speakers):
            raise ValueError(
                f"speaker_count {manifest.speaker_count} exceeds the {len(speakers)} "
                "speakers in the dataset"
            )
        if manifest.subset_size > manifest.training_pool_size:
            raise ValueError(
                f"subset_size {manifest.subset_size} exceeds training_pool_size "
                f"{manifest.training_pool_size}"
            )
        chosen_speakers = speakers[: manifest.speaker_count]
        by_speaker = {
            speaker: [u for u in records if u.speaker == speaker]
            for speaker in chosen_speakers
        }
        chosen_references = {
            speaker: rng.choice(utterances)
            for speaker, utterances in by_speaker.items()
        }
        references = {
            speaker: utterance.id for speaker, utterance in chosen_references.items()
        }
        reference_dialogues = {
            utterance.dialogue for utterance in chosen_references.values()
        }
        candidates = [
            utterance
            for utterance in records
            if utterance.speaker in chosen_speakers
            and utterance.dialogue not in reference_dialogues
        ]
        validation_pool, validation_dialogues = _sample_dialogue_disjoint_pool(
            rng,
            candidates,
            manifest.validation_pool_size,
            "validation_pool",
        )
        candidates = [
            utterance
            for utterance in candidates
            if utterance.dialogue not in validation_dialogues
        ]
        training_pool, _ = _sample_dialogue_disjoint_pool(
            rng,
            candidates,
            manifest.training_pool_size,
            "training_pool",
        )
        subsets = {
            f"subset-{index:04d}": sorted(
                rng.sample(training_pool, manifest.subset_size)
            )
            for index in range(manifest.subset_count)
        }
        return cls(
            references=references,
            training_pool=training_pool,
            validation_pool=validation_pool,
            subsets=subsets,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
