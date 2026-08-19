from __future__ import annotations

import torch
from torch import nn

from tts_data_attribution.models import evaluate, train


class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def squared_error(
    model: LinearModel,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    return (model(batch["inputs"]) - batch["targets"]).square().flatten()


def test_train_uses_the_supplied_per_example_objective() -> None:
    torch.manual_seed(0)
    model = LinearModel()
    batches = [
        {
            "inputs": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "targets": torch.tensor([[1.0], [2.0]]),
        },
        {
            "inputs": torch.tensor([[1.0, 1.0]]),
            "targets": torch.tensor([[3.0]]),
        },
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    reported: list[dict[str, int | float]] = []

    history = train(
        model,
        batches,
        batches[:1],
        optimizer,
        squared_error,
        epochs=2,
        device="cpu",
        epoch_callback=reported.append,
    )

    assert reported == history
    assert [metrics["epoch"] for metrics in history] == [1, 2]
    assert [metrics["step"] for metrics in history] == [2, 4]
    assert all(
        torch.isfinite(torch.tensor(value))
        for metrics in history
        for value in metrics.values()
    )
    assert any(
        not torch.equal(parameter, initial_parameters[name])
        for name, parameter in model.named_parameters()
    )
    assert torch.isfinite(torch.tensor(evaluate(model, batches, squared_error, "cpu")))
