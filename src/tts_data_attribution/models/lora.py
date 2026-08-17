from __future__ import annotations

from typing import cast

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel


def apply_lora(model: PreTrainedModel, config: LoraConfig) -> PeftModel:
    model.requires_grad_(False)
    return cast(PeftModel, get_peft_model(model, config))
