from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from torch.utils.data import Dataset


@dataclass(frozen=True)
class Utterance:
    id: str
    speaker: str
    dialogue: str
    text_ids: list[int]
    audio_codes: list[list[int]]


class UtteranceDataset(Dataset[Utterance]):
    def __init__(self, utterances: Iterable[Utterance]) -> None:
        self.utterances = tuple(utterances)
        self.utterances_by_id = {
            utterance.id: utterance for utterance in self.utterances
        }
        if len(self.utterances_by_id) != len(self.utterances):
            raise ValueError("utterance IDs must be unique")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Self:
        with Path(path).open(encoding="utf-8") as stream:
            return cls(Utterance(**json.loads(line)) for line in stream if line.strip())

    def to_jsonl(self, path: str | Path, append: bool = False) -> None:
        with Path(path).open(
            "a" if append else "w", encoding="utf-8", newline="\n"
        ) as stream:
            for utterance in self.utterances:
                json.dump(
                    asdict(utterance),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")

    def ids(self) -> set[str]:
        return set(self.utterances_by_id)

    def get_utterances_by_ids(self, identifiers: list[str]) -> list[Utterance]:
        return [self.utterances_by_id[identifier] for identifier in identifiers]

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> Utterance:
        return self.utterances[index]
