from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: Path
    model: str
    model_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            dataset=Path(values["dataset"]),
            model=values["model"],
            model_path=Path(values["model_path"]),
        )

    def to_yaml(self, path: str | Path) -> None:
        values = {key: str(value) for key, value in asdict(self).items()}
        Path(path).write_text(yaml.safe_dump(values, sort_keys=True), encoding="utf-8")
