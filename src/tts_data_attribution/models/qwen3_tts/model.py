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
    "codec_head",
    "code_predictor.lm_head.0",
    "code_predictor.lm_head.1",
    "code_predictor.lm_head.2",
    "code_predictor.lm_head.3",
    "code_predictor.lm_head.4",
    "code_predictor.lm_head.5",
    "code_predictor.lm_head.6",
    "code_predictor.lm_head.7",
    "code_predictor.lm_head.8",
    "code_predictor.lm_head.9",
    "code_predictor.lm_head.10",
    "code_predictor.lm_head.11",
    "code_predictor.lm_head.12",
    "code_predictor.lm_head.13",
    "code_predictor.lm_head.14",
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
