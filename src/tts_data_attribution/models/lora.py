from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, cast

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


def lora_parameter_layout(
    adapter: PeftModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[dict[str, object]], list[list[str]]]:
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

    parameters = []
    parameter_groups = []
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            name, adapter_parameter = named_parameters[id(parameter)]
            group_names.append(name)
            parameters.append(
                {
                    "name": name,
                    "shape": list(adapter_parameter.shape),
                    "dtype": str(adapter_parameter.dtype).removeprefix("torch."),
                }
            )
        parameter_groups.append(group_names)
    return parameters, parameter_groups


def validate_checkpoint_metrics(
    metrics: Any,
    *,
    expected_epoch: int | None = None,
    maximum_epoch: int | None = None,
) -> dict[str, int | float]:
    expected_keys = {"epoch", "step", "training_loss", "validation_loss"}
    if not isinstance(metrics, dict) or set(metrics) != expected_keys:
        raise ValueError(
            "checkpoint metrics must contain exactly epoch, step, training_loss, "
            "and validation_loss"
        )
    epoch = metrics["epoch"]
    step = metrics["step"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("checkpoint epoch must be a positive integer")
    if type(step) is not int or step < 1:
        raise ValueError("checkpoint step must be a positive integer")
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError("checkpoint metrics epoch differs from its directory")
    if maximum_epoch is not None and epoch > maximum_epoch:
        raise ValueError("checkpoint epoch exceeds the training run")
    losses = []
    for name in ["training_loss", "validation_loss"]:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"checkpoint {name.replace('_', ' ')} must be a number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"checkpoint {name.replace('_', ' ')} must be finite")
        losses.append(value)
    return {
        "epoch": epoch,
        "step": step,
        "training_loss": losses[0],
        "validation_loss": losses[1],
    }


def load_checkpoint_metrics(
    path: str | Path,
    *,
    expected_epoch: int | None = None,
    maximum_epoch: int | None = None,
) -> dict[str, int | float]:
    metrics = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_checkpoint_metrics(
        metrics,
        expected_epoch=expected_epoch,
        maximum_epoch=maximum_epoch,
    )


def is_lora_checkpoint_complete(directory: str | Path) -> bool:
    directory = Path(directory)
    return (
        (directory / "adapter" / "adapter_config.json").is_file()
        and (directory / "adapter" / "adapter_model.safetensors").is_file()
        and (directory / "optimizer.pt").is_file()
        and (directory / "metrics.json").is_file()
    )


def save_lora_checkpoint(
    directory: str | Path,
    adapter: PeftModel,
    optimizer: torch.optim.AdamW,
    metrics: dict[str, int | float],
) -> None:
    metrics = validate_checkpoint_metrics(metrics)
    step = metrics["step"]

    parameters, _ = lora_parameter_layout(adapter, optimizer)
    named_parameters = dict(adapter.named_parameters())
    for parameter in parameters:
        name = cast(str, parameter["name"])
        state = optimizer.state[named_parameters[name]]
        if not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise ValueError(f"AdamW state is incomplete for {name}")
        for moment_name in ["exp_avg", "exp_avg_sq"]:
            moment = state[moment_name]
            if (
                not isinstance(moment, torch.Tensor)
                or not moment.is_floating_point()
                or moment.shape != named_parameters[name].shape
                or not torch.isfinite(moment).all()
            ):
                raise ValueError(f"AdamW {moment_name} is invalid for {name}")
        state_step = state["step"]
        if isinstance(state_step, torch.Tensor):
            if state_step.numel() != 1:
                raise ValueError(f"AdamW state step is invalid for {name}")
            state_step = state_step.item()
        if (
            isinstance(state_step, bool)
            or not isinstance(state_step, int | float)
            or not math.isfinite(float(state_step))
            or not float(state_step).is_integer()
            or int(state_step) != step
        ):
            raise ValueError(
                f"AdamW state step for {name} differs from checkpoint metrics"
            )

    directory = Path(directory)
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
    (temporary_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_directory.rename(directory)
