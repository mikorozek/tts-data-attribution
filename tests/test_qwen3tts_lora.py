from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, PeftModel
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerForConditionalGeneration,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tts_data_attribution.dataset import Utterance
from tts_data_attribution.models import (
    TrackStarTransform,
    apply_lora,
    attribution_scores,
    collect_per_example_gradients,
    correct_gradients_with_adamw,
    evaluate,
    save_lora_checkpoint,
    train,
)
from tts_data_attribution.models.qwen3_tts import (
    Qwen3TTSGradientProjector,
    collate,
    objective,
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


def test_qwen_objective_produces_all_talker_lora_gradients() -> None:
    torch.manual_seed(0)
    model = tiny_lora_model()
    batch = collate(
        [utterance("a", 0)],
        {"speaker-a": torch.randn(16)},
    )

    losses = objective(model, batch)
    [gradients] = list(collect_per_example_gradients(model.talker, losses))

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
    adapter_parameters = {
        name: parameter
        for name, parameter in model.talker.named_parameters()
        if parameter.requires_grad
    }
    assert list(gradients) == list(adapter_parameters)
    assert all(
        gradient.shape == adapter_parameters[name].shape
        and torch.isfinite(gradient).all()
        for name, gradient in gradients.items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())

    projector = Qwen3TTSGradientProjector(
        model.talker,
        talker_output_dimension=4,
        code_predictor_output_dimension=9,
        talker_layers_per_block=4,
        code_predictor_layers_per_block=5,
        seed=13,
    )
    repeated_projector = Qwen3TTSGradientProjector(
        model.talker,
        talker_output_dimension=4,
        code_predictor_output_dimension=9,
        talker_layers_per_block=4,
        code_predictor_layers_per_block=5,
        seed=13,
    )
    projected = projector(gradients)

    assert projector.block_names == (
        "talker.layers.0-0.attention",
        "talker.layers.0-0.mlp",
        "code_predictor.layers.0-0.attention",
        "code_predictor.layers.0-0.mlp",
    )
    assert [
        len(projector.block_parameter_names[name]) for name in projector.block_names
    ] == [8, 6, 8, 6]
    assert sum(parameter.numel() for parameter in adapter_parameters.values()) == sum(
        adapter_parameters[name].numel()
        for names in projector.block_parameter_names.values()
        for name in names
    )
    assert [value.shape for value in projected.values()] == [
        (4,),
        (4,),
        (9,),
        (9,),
    ]
    assert all(torch.isfinite(value).all() for value in projected.values())
    for name, value in projected.items():
        torch.testing.assert_close(value, repeated_projector(gradients)[name])


def test_train_processes_every_batch_and_evaluates_without_gradients(
    tmp_path: Path,
) -> None:
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
    validation_loss = evaluate(model, validation_loader, objective, "cpu")

    assert torch.isfinite(torch.tensor(validation_loss))
    assert all(parameter.grad is None for parameter in model.parameters())

    reported_metrics: list[dict[str, int | float]] = []
    history = train(
        model,
        training_loader,
        validation_loader,
        optimizer,
        objective,
        epochs=2,
        device="cpu",
        epoch_callback=reported_metrics.append,
    )

    assert reported_metrics == history
    assert len(history) == 2
    assert [metrics["epoch"] for metrics in history] == [1, 2]
    assert [metrics["step"] for metrics in history] == [2, 4]
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

    checkpoint = tmp_path / "checkpoint"
    save_lora_checkpoint(
        checkpoint,
        model.talker,
        optimizer,
        epoch=2,
        step=4,
    )

    adapter_parameters = {
        name: parameter
        for name, parameter in model.talker.named_parameters()
        if parameter.requires_grad
    }
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["epoch"] == 2
    assert metadata["step"] == 4
    assert metadata["format_version"] == 1
    assert metadata["parameter_groups"] == [list(adapter_parameters)]
    assert [item["name"] for item in metadata["parameters"]] == list(adapter_parameters)

    optimizer_state = torch.load(
        checkpoint / "optimizer.pt", map_location="cpu", weights_only=True
    )
    assert len(optimizer_state["state"]) == len(adapter_parameters)

    torch.manual_seed(0)
    reloaded_model = TinyQwen3TTS()
    reloaded_model.requires_grad_(False)
    reloaded_model.talker = PeftModel.from_pretrained(
        reloaded_model.talker,
        checkpoint / "adapter",
        is_trainable=True,
    )
    reloaded_parameters = {
        name: parameter
        for name, parameter in reloaded_model.talker.named_parameters()
        if parameter.requires_grad
    }
    reloaded_optimizer = AdamW(reloaded_parameters.values(), lr=0.01)
    reloaded_optimizer.load_state_dict(optimizer_state)

    assert list(reloaded_parameters) == list(adapter_parameters)
    for name, parameter in adapter_parameters.items():
        torch.testing.assert_close(
            reloaded_optimizer.state[reloaded_parameters[name]]["exp_avg_sq"],
            optimizer.state[parameter]["exp_avg_sq"],
        )

    validation_batch = next(iter(validation_loader))
    model.eval()
    reloaded_model.eval()
    with torch.no_grad():
        original_losses = objective(model, validation_batch)
        reloaded_losses = objective(reloaded_model, validation_batch)
    torch.testing.assert_close(reloaded_losses, original_losses)

    losses = objective(reloaded_model, validation_batch)
    [gradients] = list(collect_per_example_gradients(reloaded_model.talker, losses))
    corrected_gradients = correct_gradients_with_adamw(
        reloaded_model.talker,
        reloaded_optimizer,
        gradients,
    )
    assert list(corrected_gradients) == list(reloaded_parameters)
    assert all(
        torch.isfinite(gradient).all() for gradient in corrected_gradients.values()
    )

    projector = Qwen3TTSGradientProjector(
        reloaded_model.talker,
        talker_output_dimension=4,
        code_predictor_output_dimension=4,
        talker_layers_per_block=4,
        code_predictor_layers_per_block=5,
        seed=13,
    )
    projected = {
        name: value.unsqueeze(0)
        for name, value in projector(corrected_gradients).items()
    }
    transform = TrackStarTransform(projected, projected, task_weight=0.0)
    encoding = transform(projected)

    assert encoding.shape == (1, 16)
    torch.testing.assert_close(torch.linalg.vector_norm(encoding, dim=1), torch.ones(1))
    torch.testing.assert_close(attribution_scores(encoding, encoding), torch.ones(1, 1))
