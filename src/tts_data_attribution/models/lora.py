from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel


def apply_lora(
    model: PreTrainedModel,
    config: LoraConfig,
    *,
    seed: int | None = None,
) -> PeftModel:
    model.requires_grad_(False)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return cast(PeftModel, get_peft_model(model, config))


def is_lora_checkpoint_complete(directory: str | Path) -> bool:
    directory = Path(directory)
    return (
        (directory / "adapter" / "adapter_config.json").is_file()
        and (directory / "adapter" / "adapter_model.safetensors").is_file()
        and (directory / "optimizer.pt").is_file()
        and (directory / "metadata.json").is_file()
    )


def save_lora_checkpoint(
    directory: str | Path,
    adapter: PeftModel,
    optimizer: torch.optim.AdamW,
    *,
    epoch: int,
    step: int,
) -> None:
    directory = Path(directory)

    named_parameters = {
        id(parameter): (name, parameter)
        for name, parameter in adapter.named_parameters()
    }
    trainable_parameter_ids = {
        id(parameter) for parameter in adapter.parameters() if parameter.requires_grad
    }
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if set(optimizer_parameter_ids) != trainable_parameter_ids or len(
        optimizer_parameter_ids
    ) != len(trainable_parameter_ids):
        raise ValueError(
            "optimizer parameters must exactly match the trainable adapter parameters"
        )

    parameter_groups = []
    parameters = []
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            name, adapter_parameter = named_parameters[id(parameter)]
            state = optimizer.state[adapter_parameter]
            if "exp_avg_sq" not in state or "step" not in state:
                raise ValueError(f"AdamW state is incomplete for {name}")
            state_step = state["step"]
            if isinstance(state_step, torch.Tensor):
                state_step = state_step.item()
            if int(state_step) != step:
                raise ValueError(
                    f"AdamW state step for {name} is {int(state_step)}, expected {step}"
                )
            group_names.append(name)
            parameters.append(
                {
                    "name": name,
                    "shape": list(adapter_parameter.shape),
                    "dtype": str(adapter_parameter.dtype).removeprefix("torch."),
                }
            )
        parameter_groups.append(group_names)

    metadata = {
        "epoch": epoch,
        "format_version": 1,
        "parameter_groups": parameter_groups,
        "parameters": parameters,
        "step": step,
    }
    if directory.exists():
        raise FileExistsError(f"checkpoint already exists at {directory}")
    temporary_directory = directory.with_name(f".{directory.name}.tmp")
    if temporary_directory.exists():
        shutil.rmtree(temporary_directory)
    temporary_directory.mkdir(parents=True)
    adapter.save_pretrained(
        str(temporary_directory / "adapter"), safe_serialization=True
    )
    torch.save(optimizer.state_dict(), temporary_directory / "optimizer.pt")
    (temporary_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_directory.rename(directory)
