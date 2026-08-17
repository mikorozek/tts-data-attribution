from __future__ import annotations

from collections.abc import Iterable

import torch
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
)

from .objective import objective


def evaluate(
    model: Qwen3TTSForConditionalGeneration,
    data_loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device | str,
) -> float:
    model.eval()
    total_loss = 0.0
    example_count = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            losses = objective(model, batch)
            total_loss += losses.sum().item()
            example_count += losses.numel()
    if example_count == 0:
        raise ValueError("cannot evaluate an empty data loader")
    return total_loss / example_count


def train(
    model: Qwen3TTSForConditionalGeneration,
    training_loader: Iterable[dict[str, torch.Tensor]],
    validation_loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str,
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
            losses = objective(model, batch)
            losses.mean().backward()
            optimizer.step()
            step += 1
            total_loss += losses.detach().sum().item()
            example_count += losses.numel()
        if example_count == 0:
            raise ValueError("cannot train with an empty data loader")
        history.append(
            {
                "epoch": epoch,
                "step": step,
                "training_loss": total_loss / example_count,
                "validation_loss": evaluate(model, validation_loader, device),
            }
        )
    return history
