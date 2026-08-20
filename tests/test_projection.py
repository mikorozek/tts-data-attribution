from __future__ import annotations

import pytest
import torch

from tts_data_attribution.trackstar import (
    RandomProjection,
    TwoSidedRandomProjection,
)


def test_two_sided_projection_matches_explicit_block_diagonal_matrix() -> None:
    gradients = (
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(2, dtype=torch.float32).reshape(1, 2),
    )
    projector = TwoSidedRandomProjection(
        tuple(gradient.shape for gradient in gradients),
        output_dimension=4,
        seed=17,
        device="cpu",
    )

    explicit_gradient = torch.block_diag(*gradients)
    explicit_left = torch.cat(projector.left_matrices, dim=1)
    explicit_right = torch.cat(projector.right_matrices, dim=1)
    expected = (explicit_left @ explicit_gradient @ explicit_right.T).flatten()

    assert isinstance(projector, RandomProjection)
    assert projector.side_size == 2
    assert projector(gradients).shape == (4,)
    torch.testing.assert_close(projector(gradients), expected)


def test_two_sided_projection_is_reproducible_and_uses_distinct_maps() -> None:
    gradients = (torch.randn(2, 3), torch.randn(2, 3))
    shapes = tuple(gradient.shape for gradient in gradients)
    first = TwoSidedRandomProjection(
        shapes,
        output_dimension=4,
        seed=23,
        device="cpu",
    )
    second = TwoSidedRandomProjection(
        shapes,
        output_dimension=4,
        seed=23,
        device="cpu",
    )

    torch.testing.assert_close(first(gradients), second(gradients))
    assert not torch.equal(first.left_matrices[0], first.left_matrices[1])
    assert not torch.equal(first.right_matrices[0], first.right_matrices[1])


def test_two_sided_projection_is_linear() -> None:
    first_gradients = (torch.randn(2, 3), torch.randn(1, 2))
    second_gradients = (torch.randn(2, 3), torch.randn(1, 2))
    projector = TwoSidedRandomProjection(
        tuple(gradient.shape for gradient in first_gradients),
        output_dimension=4,
        seed=11,
        device="cpu",
    )
    combined = tuple(
        2 * first - 3 * second
        for first, second in zip(first_gradients, second_gradients, strict=True)
    )

    torch.testing.assert_close(
        projector(combined),
        2 * projector(first_gradients) - 3 * projector(second_gradients),
    )


@pytest.mark.parametrize("output_dimension", [0, 8])
def test_two_sided_projection_requires_a_positive_square(
    output_dimension: int,
) -> None:
    with pytest.raises(ValueError, match="positive|square"):
        TwoSidedRandomProjection(
            ((3, 5),),
            output_dimension,
            seed=0,
            device="cpu",
        )


def test_two_sided_projection_rejects_changed_input_shape() -> None:
    projector = TwoSidedRandomProjection(
        ((2, 3),),
        output_dimension=4,
        seed=0,
        device="cpu",
    )

    with pytest.raises(ValueError, match="shape"):
        projector((torch.zeros(3, 2),))
