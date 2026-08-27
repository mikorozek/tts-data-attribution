from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn


def collect_per_example_gradients(
    module: nn.Module,
    losses: torch.Tensor,
) -> Iterator[tuple[torch.Tensor, ...]]:
    if losses.ndim != 1:
        raise ValueError("per-example losses must be a one-dimensional tensor")
    parameters = tuple(
        parameter for parameter in module.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("module has no trainable parameters")

    for index in range(losses.shape[0]):
        yield torch.autograd.grad(
            losses[index],
            parameters,
            retain_graph=index + 1 < losses.shape[0],
        )


def correct_gradients_with_adamw(
    module: nn.Module,
    optimizer: torch.optim.AdamW,
    gradients: tuple[torch.Tensor, ...],
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("gradient correction dtype must be floating point")
    parameters = tuple(
        parameter for parameter in module.parameters() if parameter.requires_grad
    )
    if len(gradients) != len(parameters):
        raise ValueError("gradient count must match the trainable module parameters")

    groups_by_parameter = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if groups_by_parameter.keys() != {id(parameter) for parameter in parameters}:
        raise ValueError(
            "optimizer parameters must exactly match the trainable module parameters"
        )

    corrected = []
    for index, (parameter, gradient) in enumerate(
        zip(parameters, gradients, strict=True)
    ):
        if gradient.shape != parameter.shape:
            raise ValueError(f"gradient shape differs for parameter {index}")
        group = groups_by_parameter[id(parameter)]
        if group["amsgrad"]:
            raise ValueError("AMSGrad optimizer state is not supported")
        state = optimizer.state[parameter]
        if "step" not in state or "exp_avg_sq" not in state:
            raise ValueError(f"AdamW state is incomplete for parameter {index}")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            step = step.item()
        step = int(step)
        if step < 1:
            raise ValueError(f"AdamW state step must be positive for parameter {index}")

        beta2 = group["betas"][1]
        bias_correction = 1.0 if beta2 == 0.0 else -math.expm1(step * math.log(beta2))
        second_moment = state["exp_avg_sq"].to(
            device=gradient.device,
            dtype=dtype,
        )
        if second_moment.shape != gradient.shape:
            raise ValueError(f"AdamW second moment shape differs for parameter {index}")
        if not torch.isfinite(second_moment).all():
            raise ValueError("AdamW second moment is not finite")
        corrected_gradient = gradient.to(dtype=dtype)
        denominator = (second_moment / bias_correction).sqrt() + group["eps"]
        corrected.append(corrected_gradient / denominator)
    return tuple(corrected)
