from __future__ import annotations

from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
)

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_model(
    model_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Qwen3TTSForConditionalGeneration:
    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map=device,
        dtype=dtype,
    )
    model = wrapper.model
    model.speech_tokenizer = None
    model.speaker_encoder = None
    return model
