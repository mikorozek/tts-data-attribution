from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self

import torch


class AudioCodesTokenizer(Protocol):
    def encode(self, audios: list[str]) -> object: ...


class CodesEncoder:
    def __init__(self, tokenizer: AudioCodesTokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, tokenizer_path: str | Path, device: str) -> Self:
        from qwen_tts import Qwen3TTSTokenizer

        return cls(Qwen3TTSTokenizer.from_pretrained(str(tokenizer_path), device_map=device))

    def encode(self, audio_paths: list[Path]) -> list[torch.Tensor]:
        encoded = self.tokenizer.encode([str(path) for path in audio_paths])
        return [codes.cpu() for codes in encoded.audio_codes]
