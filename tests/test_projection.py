from __future__ import annotations

import pytest
import torch

from tts_data_attribution.models import TwoSidedProjector


def test_two_sided_projection_has_requested_dimension_and_is_reproducible() -> None:
    matrix = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    first = TwoSidedProjector(
        matrix.shape,
        output_dimension=16,
        seed=7,
        device="cpu",
    )
    second = TwoSidedProjector(
        matrix.shape,
        output_dimension=16,
        seed=7,
        device="cpu",
    )

    projected = first(matrix)

    assert first.left.shape == (4, 3)
    assert first.right.shape == (4, 5)
    assert projected.shape == (16,)
    torch.testing.assert_close(first.left, second.left)
    torch.testing.assert_close(first.right, second.right)
    torch.testing.assert_close(projected, second(matrix))


def test_two_sided_projection_is_linear() -> None:
    first_matrix = torch.randn(3, 5)
    second_matrix = torch.randn(3, 5)
    projector = TwoSidedProjector(
        first_matrix.shape,
        output_dimension=4,
        seed=11,
        device="cpu",
    )

    projected_sum = projector(2 * first_matrix - 3 * second_matrix)
    sum_of_projections = 2 * projector(first_matrix) - 3 * projector(second_matrix)

    torch.testing.assert_close(projected_sum, sum_of_projections)


@pytest.mark.parametrize("output_dimension", [0, 8])
def test_two_sided_projection_requires_a_positive_square(
    output_dimension: int,
) -> None:
    with pytest.raises(ValueError, match="positive square"):
        TwoSidedProjector(
            (3, 5),
            output_dimension,
            seed=0,
            device="cpu",
        )
