from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tts_data_attribution.models import apply_lora


class TinyDecoderConfig(PretrainedConfig):
    model_type = "tiny-decoder"


class TinyDecoder(PreTrainedModel):
    config_class = TinyDecoderConfig

    def __init__(self, config: TinyDecoderConfig) -> None:
        super().__init__(config)
        self.q_proj = nn.Linear(4, 4)
        self.down_proj = nn.Linear(4, 4)
        self.post_init()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.q_proj(hidden_states)))


def test_apply_lora_trains_and_serializes_only_adapter_parameters(
    tmp_path,
) -> None:
    torch.manual_seed(0)
    base_path = tmp_path / "base"
    adapter_path = tmp_path / "adapter"
    base_config = TinyDecoderConfig(name_or_path=str(base_path))
    base_config.save_pretrained(base_path)
    base_model = TinyDecoder(base_config)
    base_state = {
        name: value.detach().clone() for name, value in base_model.state_dict().items()
    }
    model = apply_lora(
        base_model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "down_proj"],
        ),
    )

    trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_parameters
    assert all("lora_" in name for name in trainable_parameters)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in trainable_parameters
    )

    inputs = torch.randn(3, 4)
    loss = model(inputs).square().mean()
    loss.backward()

    assert all(
        parameter.grad is not None for parameter in trainable_parameters.values()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in trainable_parameters
    )

    optimizer = torch.optim.AdamW(trainable_parameters.values(), lr=0.1)
    optimizer.step()
    model.save_pretrained(adapter_path)

    reloaded_base_model = TinyDecoder(base_config)
    reloaded_base_model.load_state_dict(base_state)
    reloaded_model = PeftModel.from_pretrained(
        reloaded_base_model, adapter_path, is_trainable=True
    )
    model.eval()
    reloaded_model.eval()

    assert torch.allclose(model(inputs), reloaded_model(inputs))
