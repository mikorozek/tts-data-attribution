from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from tts_data_attribution.models.qwen3_tts import Qwen3TTSEncoder


@dataclass(frozen=True)
class SourceRecord:
    id: str
    text: str
    speaker: str
    dialogue: str
    audio_path: str


class FakeEncodedAudio:
    def __init__(self) -> None:
        self.audio_codes = [torch.full((2, 16), 7), torch.full((1, 16), 8)]


class FakeSpeechTokenizer:
    def __init__(self) -> None:
        self.received: list[list[str]] = []

    def encode(self, audio_paths: list[str]) -> FakeEncodedAudio:
        self.received.append(audio_paths)
        return FakeEncodedAudio()


class FakeProcessor:
    def __init__(self) -> None:
        self.received: list[str] = []

    def __call__(self, text: str) -> dict:
        self.received.append(text)
        return {"input_ids": [[151644, 77091, 198, 9707]]}


class FakeInnerModel:
    speaker_encoder_sample_rate = 24000

    def __init__(self) -> None:
        self.speech_tokenizer = FakeSpeechTokenizer()
        self.speaker_calls: list[tuple[int, int]] = []

    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        self.speaker_calls.append((len(audio), sr))
        return torch.full((3,), float(len(audio)))


class FakeModel:
    def __init__(self) -> None:
        self.processor = FakeProcessor()
        self.model = FakeInnerModel()


def test_encode_text_uses_the_assistant_prefix() -> None:
    model = FakeModel()

    text_ids = Qwen3TTSEncoder(model).encode_text("Hello")

    assert text_ids == [151644, 77091, 198, 9707]
    assert model.processor.received == ["<|im_start|>assistant\nHello"]


def test_encode_audio_uses_the_model_speech_tokenizer() -> None:
    model = FakeModel()

    codes = Qwen3TTSEncoder(model).encode_audio([Path("a.wav"), Path("b.wav")])

    assert codes == [[[7] * 16] * 2, [[8] * 16]]
    assert model.model.speech_tokenizer.received == [["a.wav", "b.wav"]]


def test_encode_speaker_uses_the_model_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[tuple[str, int, bool]] = []

    def fake_load(path: str, sr: int, mono: bool) -> tuple[np.ndarray, int]:
        loaded.append((path, sr, mono))
        return np.zeros(5, dtype=np.float32), sr

    monkeypatch.setattr(
        "tts_data_attribution.models.qwen3_tts.encoder.librosa.load", fake_load
    )
    model = FakeModel()

    embedding = Qwen3TTSEncoder(model).encode_speaker(Path("reference.wav"))

    assert loaded == [("reference.wav", 24000, True)]
    assert model.model.speaker_calls == [(5, 24000)]
    assert torch.equal(embedding, torch.full((3,), 5.0))
    assert embedding.device.type == "cpu"


def test_from_pretrained_loads_one_qwen_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[tuple[str, str]] = []
    model = FakeModel()

    def fake_from_pretrained(model_path: str, device_map: str) -> FakeModel:
        loaded.append((model_path, device_map))
        return model

    monkeypatch.setattr(
        "tts_data_attribution.models.qwen3_tts.encoder.Qwen3TTSModel.from_pretrained",
        fake_from_pretrained,
    )

    encoder = Qwen3TTSEncoder.from_pretrained(tmp_path / "model", "cpu")

    assert encoder.model is model
    assert loaded == [(str(tmp_path / "model"), "cpu")]


def test_encode_utterances_combines_text_and_audio_codes(tmp_path: Path) -> None:
    model = FakeModel()
    utterances = [
        SourceRecord(
            id="a", text="Hello", speaker="0", dialogue="1", audio_path="a.wav"
        ),
        SourceRecord(id="b", text="Bye", speaker="1", dialogue="1", audio_path="b.wav"),
    ]

    encoded = Qwen3TTSEncoder(model).encode_utterances(utterances, tmp_path)

    assert [item.id for item in encoded] == ["a", "b"]
    assert encoded[0].text_ids == [151644, 77091, 198, 9707]
    assert encoded[0].audio_codes == [[7] * 16] * 2
    assert encoded[1].audio_codes == [[8] * 16]
    assert model.model.speech_tokenizer.received == [
        [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]
    ]
