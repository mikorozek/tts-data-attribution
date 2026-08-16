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


@dataclass(frozen=True)
class Plan:
    references: dict[str, str]
    training_pool: list[str]
    subsets: list[list[str]]

    @classmethod
    def sample(
        cls, manifest: ExperimentManifest, dataset: Collection[_SourceRecord]
    ) -> Self:
        rng = random.Random(manifest.seed)
        speakers = sorted({utterance.speaker for utterance in dataset})
        if manifest.speaker_count > len(speakers):
            raise ValueError(
                f"speaker_count {manifest.speaker_count} exceeds the {len(speakers)} "
                "speakers in the dataset"
            )
        chosen_speakers = speakers[: manifest.speaker_count]
        by_speaker = {
            speaker: [u.id for u in dataset if u.speaker == speaker]
            for speaker in chosen_speakers
        }
        references = {speaker: rng.choice(ids) for speaker, ids in by_speaker.items()}
        candidates = [
            utterance_id
            for speaker in chosen_speakers
            for utterance_id in by_speaker[speaker]
            if utterance_id != references[speaker]
        ]
        if manifest.training_pool_size > len(candidates):
            raise ValueError(
                f"training_pool_size {manifest.training_pool_size} exceeds the "
                f"{len(candidates)} candidate utterances"
            )
        if manifest.subset_size > manifest.training_pool_size:
            raise ValueError(
                f"subset_size {manifest.subset_size} exceeds training_pool_size "
                f"{manifest.training_pool_size}"
            )
        training_pool = sorted(rng.sample(candidates, manifest.training_pool_size))
        subsets = [
            sorted(rng.sample(training_pool, manifest.subset_size))
            for _ in range(manifest.subset_count)
        ]
        return cls(references=references, training_pool=training_pool, subsets=subsets)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
