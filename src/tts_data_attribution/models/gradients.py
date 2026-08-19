from __future__ import annotations

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
) -> dict[str, torch.Tensor]:
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
        group = groups_by_parameter[id(parameter)]
        state = optimizer.state[parameter]
        step = state["step"]
        if isinstance(step, torch.Tensor):
            step = step.item()
        beta2 = group["betas"][1]
        bias_correction = 1 - beta2 ** int(step)
        second_moment = state["exp_avg_sq"].to(
            device=gradient.device,
            dtype=gradient.dtype,
        )
        denominator = (second_moment / bias_correction).sqrt() + group["eps"]
        corrected[name] = gradient / denominator
    return corrected
