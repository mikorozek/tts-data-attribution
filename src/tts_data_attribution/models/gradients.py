from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn


def collect_per_example_gradients(
    module: nn.Module,
    losses: torch.Tensor,
) -> Iterator[dict[str, torch.Tensor]]:
    if losses.ndim != 1:
        raise ValueError("per-example losses must be a one-dimensional tensor")
    named_parameters = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("module has no trainable parameters")
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]

    for index in range(losses.shape[0]):
        gradients = torch.autograd.grad(
            losses[index],
            parameters,
            retain_graph=index + 1 < losses.shape[0],
        )
        yield {
            name: gradient.detach()
            for name, gradient in zip(names, gradients, strict=True)
        }


def correct_gradients_with_adamw(
    module: nn.Module,
    optimizer: torch.optim.AdamW,
    gradients: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("gradient correction dtype must be floating point")
    parameters = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if gradients.keys() != parameters.keys():
        raise ValueError("gradient names must match the trainable module parameters")
    groups_by_parameter = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    corrected = {}
    for name, gradient in gradients.items():
        parameter = parameters[name]
        if id(parameter) not in groups_by_parameter:
            raise ValueError(f"optimizer does not contain parameter {name}")
        group = groups_by_parameter[id(parameter)]
        if group["amsgrad"]:
            raise ValueError("AMSGrad optimizer state is not supported")
        state = optimizer.state[parameter]
        if "step" not in state or "exp_avg_sq" not in state:
            raise ValueError(f"AdamW state is incomplete for {name}")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            step = step.item()
        step = int(step)
        if step < 1:
            raise ValueError(f"AdamW state step must be positive for {name}")

        beta2 = group["betas"][1]
        bias_correction = 1.0 if beta2 == 0.0 else -math.expm1(step * math.log(beta2))
        second_moment = state["exp_avg_sq"].to(
            device=gradient.device,
            dtype=dtype,
        )
        if second_moment.shape != gradient.shape:
            raise ValueError(f"AdamW second moment shape differs for {name}")
        if not torch.isfinite(second_moment).all():
            raise ValueError(f"AdamW second moment is not finite for {name}")
        corrected_gradient = gradient.to(dtype=dtype)
        denominator = (second_moment / bias_correction).sqrt() + group["eps"]
        corrected[name] = corrected_gradient / denominator
    return corrected
