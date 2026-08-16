from __future__ import annotations

from pathlib import Path
from typing import Self

import librosa
import torch
from qwen_tts import Qwen3TTSModel

from ..speaker_reference_audio_encoder import SpeakerReferenceAudioEncoder


class Qwen3TTSSpeakerReferenceAudioEncoder(SpeakerReferenceAudioEncoder):
    def __init__(self, model: Qwen3TTSModel) -> None:
        self.model = model

    @classmethod
    def from_pretrained(cls, model_path: str | Path, device: str) -> Self:
        return cls(Qwen3TTSModel.from_pretrained(str(model_path), device_map=device))

    def encode(self, reference_audio_path: Path) -> torch.Tensor:
        sample_rate = self.model.model.speaker_encoder_sample_rate
        wav, _ = librosa.load(str(reference_audio_path), sr=sample_rate, mono=True)
        return self.model.model.extract_speaker_embedding(wav, sample_rate).cpu()
