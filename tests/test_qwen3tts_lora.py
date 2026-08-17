from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import torch
from peft import LoraConfig
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerForConditionalGeneration,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tts_data_attribution.dataset import Utterance
from tts_data_attribution.models import apply_lora
from tts_data_attribution.models.qwen3_tts import (
    collate,
    evaluate,
    objective,
    train,
)


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


def tiny_lora_model() -> TinyQwen3TTS:
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
    return model


def utterance(identifier: str, offset: int) -> Utterance:
    return Utterance(
        id=identifier,
        speaker="speaker-a",
        dialogue=f"dialogue-{identifier}",
        text_ids=[1 + offset, 2 + offset, 3 + offset, 4 + offset],
        audio_codes=[
            list(range(1 + offset, 17 + offset)),
            list(range(2 + offset, 18 + offset)),
        ],
    )


def test_qwen_objective_backpropagates_only_through_talker_lora() -> None:
    torch.manual_seed(0)
    model = tiny_lora_model()
    batch = collate(
        [utterance("a", 0)],
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


def test_train_processes_every_batch_and_evaluates_without_gradients() -> None:
    torch.manual_seed(0)
    model = tiny_lora_model()
    speaker_embeddings = {"speaker-a": torch.randn(16)}
    collate_batch = partial(collate, speaker_embeddings=speaker_embeddings)
    training_loader = DataLoader(
        [utterance("a", 0), utterance("b", 1), utterance("c", 2)],
        batch_size=2,
        collate_fn=collate_batch,
    )
    validation_loader = DataLoader(
        [utterance("d", 3)],
        batch_size=1,
        collate_fn=collate_batch,
    )
    trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    initial_trainable_parameters = {
        name: parameter.detach().clone()
        for name, parameter in trainable_parameters.items()
    }
    initial_frozen_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = AdamW(trainable_parameters.values(), lr=0.01)

    model.zero_grad(set_to_none=True)
    validation_loss = evaluate(model, validation_loader, "cpu")

    assert torch.isfinite(torch.tensor(validation_loss))
    assert all(parameter.grad is None for parameter in model.parameters())

    history = train(
        model,
        training_loader,
        validation_loader,
        optimizer,
        epochs=2,
        device="cpu",
    )

    assert len(history) == 2
    assert all(
        torch.isfinite(torch.tensor(value))
        for metrics in history
        for value in metrics.values()
    )
    first_parameter = next(iter(trainable_parameters.values()))
    assert optimizer.state[first_parameter]["step"].item() == 4
    assert any(
        not torch.equal(parameter, initial_trainable_parameters[name])
        for name, parameter in trainable_parameters.items()
    )
    assert all(
        torch.equal(parameter, initial_frozen_parameters[name])
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
