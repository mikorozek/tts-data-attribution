from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from ..dataset import UtteranceDataset


class UtteranceAudioEncoder(ABC):
    @classmethod
    @abstractmethod
    def from_pretrained(cls, tokenizer_path: str | Path, device: str) -> Self: ...

    @abstractmethod
    def encode(
        self, dataset: UtteranceDataset, audio_root: Path, output: Path, batch_size: int
    ) -> None: ...
