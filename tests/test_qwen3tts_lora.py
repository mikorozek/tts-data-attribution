from __future__ import annotations

from types import SimpleNamespace

import torch
from peft import LoraConfig
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerForConditionalGeneration,
)
from torch import nn

from tts_data_attribution.dataset import Utterance
from tts_data_attribution.models import apply_lora
from tts_data_attribution.models.qwen3_tts import collate, objective


class TinyQwen3TTS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        predictor_config = {
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "num_code_groups": 16,
            "layer_types": ["full_attention"],
            "use_cache": False,
        }
        talker_config = Qwen3TTSTalkerConfig(
            code_predictor_config=predictor_config,
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            num_code_groups=16,
            text_hidden_size=16,
            text_vocab_size=64,
            codec_eos_token_id=53,
            codec_nothink_id=54,
            codec_think_bos_id=55,
            codec_think_eos_id=56,
            codec_pad_id=57,
            codec_bos_id=58,
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "interleaved": True,
            },
            layer_types=["full_attention"],
            use_cache=False,
        )
        talker_config._attn_implementation = "eager"
        self.talker = Qwen3TTSTalkerForConditionalGeneration(talker_config)
        self.config = SimpleNamespace(
            tts_pad_token_id=59,
            tts_bos_token_id=60,
            tts_eos_token_id=61,
            talker_config=talker_config,
        )


def test_qwen_objective_backpropagates_only_through_talker_lora() -> None:
    torch.manual_seed(0)
    model = TinyQwen3TTS()
    model.requires_grad_(False)
    model.talker = apply_lora(
        model.talker,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    batch = collate(
        [
            Utterance(
                id="a",
                speaker="speaker-a",
                dialogue="dialogue-a",
                text_ids=[1, 2, 3, 4],
                audio_codes=[list(range(1, 17)), list(range(2, 18))],
            )
        ],
        {"speaker-a": torch.randn(16)},
    )

    losses = objective(model, batch)
    losses.mean().backward()

    trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert losses.shape == (1,)
    assert torch.isfinite(losses).all()
    assert len(trainable_parameters) == 28
    assert all("lora_" in name for name in trainable_parameters)
    assert any(
        ".model.layers.0." in name and ".code_predictor." not in name
        for name in trainable_parameters
    )
    assert any(
        ".code_predictor.model.layers.0." in name for name in trainable_parameters
    )
    assert all(
        parameter.grad is not None for parameter in trainable_parameters.values()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in trainable_parameters
    )
