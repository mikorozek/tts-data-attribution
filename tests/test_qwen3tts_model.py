from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tts_data_attribution.models.qwen3_tts import model as qwen_model


def test_load_model_extracts_the_trainable_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core: Any = SimpleNamespace(
        speech_tokenizer=object(),
        speaker_encoder=object(),
    )
    loaded_arguments: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(path: str, **kwargs: Any) -> SimpleNamespace:
        loaded_arguments.append((path, kwargs))
        return SimpleNamespace(model=core)

    monkeypatch.setattr(
        qwen_model.Qwen3TTSModel,
        "from_pretrained",
        staticmethod(from_pretrained),
    )

    result = qwen_model.load_model(
        Path("model"),
        device="cpu",
        dtype=torch.float32,
    )

    assert result is core
    assert result.speech_tokenizer is None
    assert result.speaker_encoder is None
    assert loaded_arguments == [
        ("model", {"device_map": "cpu", "dtype": torch.float32})
    ]
