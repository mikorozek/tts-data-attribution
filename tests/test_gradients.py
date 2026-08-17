from __future__ import annotations

import pytest
import torch
from torch import nn

from tts_data_attribution.models import collect_per_example_gradients


def test_collect_per_example_gradients_matches_backward() -> None:
    model = nn.Linear(2, 1)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([0.5, -0.5])
    losses = (model(inputs).squeeze(1) - targets).square()

    collected = list(collect_per_example_gradients(model, losses))

    assert len(collected) == 2
    assert all(parameter.grad is None for parameter in model.parameters())
    for index, gradients in enumerate(collected):
        model.zero_grad(set_to_none=True)
        loss = (model(inputs[index]).squeeze(0) - targets[index]).square()
        loss.backward()
        expected = {
            name: parameter.grad for name, parameter in model.named_parameters()
        }
        assert list(gradients) == list(expected)
        for name, gradient in gradients.items():
            torch.testing.assert_close(gradient, expected[name])


def test_collect_per_example_gradients_requires_vector_losses() -> None:
    model = nn.Linear(2, 1)

    with pytest.raises(ValueError, match="one-dimensional"):
        list(
            collect_per_example_gradients(
                model,
                torch.ones((2, 1), requires_grad=True),
            )
        )
