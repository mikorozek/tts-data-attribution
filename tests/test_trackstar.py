from __future__ import annotations

import torch

from tts_data_attribution.models import (
    TrackStar,
    TrackStarTransform,
    attribution_scores,
    stack_projected_gradients,
)


def inverse_square_root(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * values.rsqrt()) @ vectors.T


def test_trackstar_transform_uses_the_mixed_unnormalized_autocorrelation() -> None:
    training = {
        "attention": torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        "mlp": torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64),
    }
    task = {
        "attention": torch.tensor([[1.0, 0.0], [0.0, 3.0]], dtype=torch.float64),
        "mlp": torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    }
    projected = {
        "attention": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "mlp": torch.tensor([3.0, 4.0], dtype=torch.float64),
    }
    task_weight = 0.25
    transform = TrackStarTransform(
        training,
        task,
        task_weight=task_weight,
        singular_value_rcond=0.0,
    )

    corrected_blocks = []
    for name in training:
        autocorrelation = (1.0 - task_weight) * training[name].T @ training[
            name
        ] + task_weight * task[name].T @ task[name]
        corrected_blocks.append(projected[name] @ inverse_square_root(autocorrelation))
    expected = torch.cat(corrected_blocks)
    expected = expected / torch.linalg.vector_norm(expected)

    torch.testing.assert_close(transform(projected), expected)


def test_trackstar_transform_uses_a_pseudoinverse_for_missing_directions() -> None:
    training = {"block": torch.tensor([[1.0, 0.0]])}
    task = {"block": torch.tensor([[2.0, 0.0]])}
    transform = TrackStarTransform(
        training,
        task,
        task_weight=0.5,
    )

    transformed = transform({"block": torch.tensor([[3.0, 4.0], [-2.0, 7.0]])})

    torch.testing.assert_close(
        transformed,
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
    )
    assert torch.isfinite(transformed).all()


def test_trackstar_normalizes_once_after_concatenating_blocks() -> None:
    training = {
        "first": torch.eye(2),
        "second": torch.eye(2),
    }
    transform = TrackStarTransform(
        training,
        training,
        task_weight=0.0,
        singular_value_rcond=0.0,
    )

    transformed = transform(
        {
            "first": torch.tensor([3.0, 4.0]),
            "second": torch.tensor([0.0, 12.0]),
        }
    )

    torch.testing.assert_close(
        transformed,
        torch.tensor([3.0, 4.0, 0.0, 12.0]) / 13.0,
    )


def test_projected_gradients_stack_and_score_in_query_by_training_order() -> None:
    stacked = stack_projected_gradients(
        [
            {"block": torch.tensor([1.0, 0.0])},
            {"block": torch.tensor([0.0, 1.0])},
        ]
    )
    queries = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    scores = attribution_scores(stacked["block"], queries)

    torch.testing.assert_close(stacked["block"], torch.eye(2))
    torch.testing.assert_close(
        scores,
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
    )


def test_trackstar_reuses_training_gradients_with_different_query_pools() -> None:
    training = {
        "block": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
    }
    first_queries = {"block": torch.tensor([[1.0, 0.0]])}
    second_queries = {
        "block": torch.tensor(
            [
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        )
    }
    trackstar = TrackStar(training)

    first_scores = trackstar.score(first_queries, task_weight=0.5)
    second_scores = trackstar.score(second_queries, task_weight=0.5)

    assert first_scores.shape == (1, 3)
    assert second_scores.shape == (2, 3)
    assert torch.isfinite(first_scores).all()
    assert torch.isfinite(second_scores).all()
    torch.testing.assert_close(trackstar.training["block"], training["block"])
