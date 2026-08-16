from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import torch


class SpeakerReferenceAudioEncoder(ABC):
    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_path: str | Path, device: str) -> Self: ...

    @abstractmethod
    def encode(self, reference_audio_path: Path) -> torch.Tensor: ...
