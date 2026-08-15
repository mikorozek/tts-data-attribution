from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self, TypeAlias

from torch.utils.data import Dataset

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class DatasetExample:
    id: str
    payload: Mapping[str, JsonValue]
    groups: Mapping[str, str] | None = None
    metadata: Mapping[str, JsonValue] | None = None


class AttributionDataset(Dataset[DatasetExample]):
    def __init__(self, examples: Iterable[DatasetExample]) -> None:
        self.examples = tuple(examples)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Self:
        with Path(path).open(encoding="utf-8") as stream:
            return cls(DatasetExample(**json.loads(line)) for line in stream)

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
            for example in self.examples:
                json.dump(
                    asdict(example),
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DatasetExample:
        return self.examples[index]
