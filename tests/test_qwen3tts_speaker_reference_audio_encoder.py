from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tts_data_attribution.models.qwen3_tts import Qwen3TTSSpeakerReferenceAudioEncoder


class FakeInnerModel:
    speaker_encoder_sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        self.calls.append((len(audio), sr))
        return torch.full((3,), float(len(audio)))


class FakeUpstreamModel:
    def __init__(self) -> None:
        self.model = FakeInnerModel()


def test_encode_loads_at_the_speaker_encoder_rate_and_returns_a_cpu_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[tuple[str, int, bool]] = []

    def fake_load(path: str, sr: int, mono: bool) -> tuple[np.ndarray, int]:
        loaded.append((path, sr, mono))
        return np.zeros(5, dtype=np.float32), sr

    monkeypatch.setattr(
        "tts_data_attribution.models.qwen3_tts.speaker_reference_audio_encoder.librosa.load",
        fake_load,
    )
    upstream = FakeUpstreamModel()

    embedding = Qwen3TTSSpeakerReferenceAudioEncoder(upstream).encode(Path("ref.wav"))

    assert loaded == [("ref.wav", 24000, True)]
    assert upstream.model.calls == [(5, 24000)]
    assert torch.equal(embedding, torch.full((3,), 5.0))
    assert embedding.device.type == "cpu"
