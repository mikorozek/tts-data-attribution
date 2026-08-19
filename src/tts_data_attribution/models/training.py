from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

import torch
from torch import nn

ModelType = TypeVar("ModelType", bound=nn.Module)
Batch = dict[str, torch.Tensor]
Objective = Callable[[ModelType, Batch], torch.Tensor]


def _losses(
    model: ModelType,
    batch: Batch,
    objective: Objective[ModelType],
) -> torch.Tensor:
    losses = objective(model, batch)
    if losses.ndim != 1:
        raise ValueError("objective must return one loss per example")
    return losses


def evaluate(
    model: ModelType,
    data_loader: Iterable[Batch],
    objective: Objective[ModelType],
    device: torch.device | str,
) -> float:
    model.eval()
    total_loss = 0.0
    example_count = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            losses = _losses(model, batch, objective)
            total_loss += losses.sum().item()
            example_count += losses.numel()
    if example_count == 0:
        raise ValueError("cannot evaluate an empty data loader")
    return total_loss / example_count


def train(
    model: ModelType,
    training_loader: Iterable[Batch],
    validation_loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    objective: Objective[ModelType],
    epochs: int,
    device: torch.device | str,
    epoch_callback: Callable[[dict[str, int | float]], None] | None = None,
) -> list[dict[str, int | float]]:
    history = []
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        example_count = 0
        for batch in training_loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            losses = _losses(model, batch, objective)
            losses.mean().backward()
            optimizer.step()
            step += 1
            total_loss += losses.detach().sum().item()
            example_count += losses.numel()
        if example_count == 0:
            raise ValueError("cannot train with an empty data loader")
        metrics: dict[str, int | float] = {
            "epoch": epoch,
            "step": step,
            "training_loss": total_loss / example_count,
            "validation_loss": evaluate(model, validation_loader, objective, device),
        }
        history.append(metrics)
        if epoch_callback is not None:
            epoch_callback(metrics)
    return history
