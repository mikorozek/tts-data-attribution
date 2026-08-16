from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import yaml


@dataclass(frozen=True)
class ExperimentManifest:
    dataset: str
    data_root: Path
    training_pool_size: int
    subset_count: int
    subset_size: int
    speaker_count: int
    seed: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            dataset=values["dataset"],
            data_root=Path(values["data_root"]),
            training_pool_size=values["training_pool_size"],
            subset_count=values["subset_count"],
            subset_size=values["subset_size"],
            speaker_count=values["speaker_count"],
            seed=values["seed"],
        )

    def to_yaml(self, path: str | Path) -> None:
        values = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
        Path(path).write_text(yaml.safe_dump(values, sort_keys=True), encoding="utf-8")
