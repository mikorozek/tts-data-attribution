from __future__ import annotations

import pytest
import torch

from tts_data_attribution.trackstar import GaussNewtonHessianApproximation


def test_gauss_newton_hessian_matches_weighted_gradient_autocorrelation() -> None:
    training = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float64,
    )
    queries = torch.tensor(
        [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [3.0, 2.0, 1.0]],
        dtype=torch.float64,
    )
    approximation = GaussNewtonHessianApproximation(task_weight=0.75)

    hessian = approximation.compute(training, queries)

    expected = 0.25 * (training.T @ training) + 0.75 * (queries.T @ queries)
    torch.testing.assert_close(hessian, expected)
    torch.testing.assert_close(hessian, hessian.T)


def test_gauss_newton_hessian_weight_boundaries_select_one_collection() -> None:
    training = torch.tensor([[1.0, 2.0]])
    queries = torch.tensor([[3.0, 4.0]])

    training_hessian = GaussNewtonHessianApproximation(
        task_weight=0.0
    ).compute(training, queries)
    query_hessian = GaussNewtonHessianApproximation(task_weight=1.0).compute(
        training,
        queries,
    )

    torch.testing.assert_close(training_hessian, training.T @ training)
    torch.testing.assert_close(query_hessian, queries.T @ queries)


@pytest.mark.parametrize("task_weight", [-0.1, 1.1, float("nan")])
def test_gauss_newton_hessian_rejects_invalid_task_weight(
    task_weight: float,
) -> None:
    with pytest.raises(ValueError, match="task weight"):
        GaussNewtonHessianApproximation(task_weight=task_weight)


def test_gauss_newton_hessian_validates_projected_gradients() -> None:
    approximation = GaussNewtonHessianApproximation(task_weight=0.5)

    with pytest.raises(ValueError, match="matrices"):
        approximation.compute(torch.ones(2), torch.ones(1, 2))
    with pytest.raises(ValueError, match="dimensions"):
        approximation.compute(torch.ones(1, 2), torch.ones(1, 3))
    with pytest.raises(ValueError, match="finite"):
        approximation.compute(
            torch.tensor([[1.0, float("nan")]]),
            torch.ones(1, 2),
        )
