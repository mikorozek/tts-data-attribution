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
    text: str
    speaker: str
    dialogue: str
    audio_path: str
    audio_codes: list[list[int]] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class UtteranceDataset(Dataset[Utterance]):
    def __init__(self, utterances: Iterable[Utterance]) -> None:
        self.utterances = tuple(utterances)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Self:
        with Path(path).open(encoding="utf-8") as stream:
            return cls(Utterance(**json.loads(line)) for line in stream if line.strip())

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
            stream.writelines(utterance.to_json() + "\n" for utterance in self.utterances)

    def ids(self) -> set[str]:
        return {utterance.id for utterance in self.utterances}

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> Utterance:
        return self.utterances[index]
