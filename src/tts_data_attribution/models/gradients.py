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
