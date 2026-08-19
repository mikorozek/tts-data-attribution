from __future__ import annotations

import json
import random
from collections.abc import Collection
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
    records: list[_SourceRecord],
    size: int,
    name: str,
) -> tuple[list[str], set[str]]:
    if size < 0:
        raise ValueError(f"{name}_size must not be negative")
    if size == 0:
        return [], set()
    by_dialogue: dict[str, list[str]] = {}
    for record in records:
        by_dialogue.setdefault(record.dialogue, []).append(record.id)
    dialogues = sorted(by_dialogue)
    rng.shuffle(dialogues)
    selected_dialogues: set[str] = set()
    candidate_count = 0
    for dialogue in dialogues:
        selected_dialogues.add(dialogue)
        candidate_count += len(by_dialogue[dialogue])
        if candidate_count >= size:
            break
    if candidate_count < size:
        raise ValueError(
            f"{name}_size {size} exceeds the {candidate_count} candidate "
            "utterances available in dialogue-disjoint groups"
        )
    candidates = [
        identifier
        for dialogue in sorted(selected_dialogues)
        for identifier in by_dialogue[dialogue]
    ]
    return sorted(rng.sample(candidates, size)), selected_dialogues


@dataclass(frozen=True)
class Plan:
    references: dict[str, str]
    training_pool: list[str]
    validation_pool: list[str]
    subsets: list[list[str]]

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
        subsets = [
            sorted(rng.sample(training_pool, manifest.subset_size))
            for _ in range(manifest.subset_count)
        ]
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
