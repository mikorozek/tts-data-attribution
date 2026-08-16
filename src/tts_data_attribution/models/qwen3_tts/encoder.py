from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, Self

import librosa
import torch
from qwen_tts import Qwen3TTSModel

from ...dataset import Utterance


class _SourceRecord(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def speaker(self) -> str: ...

    @property
    def dialogue(self) -> str: ...

    @property
    def audio_path(self) -> str: ...


class Qwen3TTSEncoder:
    def __init__(self, model: Qwen3TTSModel) -> None:
        self.model = model

    @classmethod
    def from_pretrained(cls, model_path: str | Path, device: str) -> Self:
        return cls(Qwen3TTSModel.from_pretrained(str(model_path), device_map=device))

    def encode_text(self, text: str) -> list[int]:
        prompt = f"<|im_start|>assistant\n{text}"
        return list(self.model.processor(text=prompt)["input_ids"][0])

    def encode_audio(self, audio_paths: list[Path]) -> list[list[list[int]]]:
        encoded = self.model.model.speech_tokenizer.encode(
            [str(path) for path in audio_paths]
        )
        return [codes.cpu().tolist() for codes in encoded.audio_codes]

    def encode_utterances(
        self, utterances: Sequence[_SourceRecord], data_root: Path
    ) -> list[Utterance]:
        audio_codes = self.encode_audio(
            [data_root / utterance.audio_path for utterance in utterances]
        )
        return [
            Utterance(
                id=utterance.id,
                speaker=utterance.speaker,
                dialogue=utterance.dialogue,
                text_ids=self.encode_text(utterance.text),
                audio_codes=codes,
            )
            for utterance, codes in zip(utterances, audio_codes, strict=True)
        ]

    def encode_speaker(self, audio_path: Path) -> torch.Tensor:
        sample_rate = self.model.model.speaker_encoder_sample_rate
        audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        return self.model.model.extract_speaker_embedding(audio, sample_rate).cpu()
