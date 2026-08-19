from __future__ import annotations

import math

import torch


class TwoSidedProjector:
    def __init__(
        self,
        input_shape: tuple[int, int] | torch.Size,
        output_dimension: int,
        *,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        side_size = math.isqrt(output_dimension)
        if output_dimension < 1 or side_size * side_size != output_dimension:
            raise ValueError("two-sided projection dimension must be a positive square")
        if len(input_shape) != 2:
            raise ValueError("two-sided projection input must be a matrix")
        rows, columns = input_shape
        if rows < 1 or columns < 1:
            raise ValueError("two-sided projection input dimensions must be positive")

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        standard_deviation = side_size**-0.5
        self.input_shape = (rows, columns)
        self.output_dimension = output_dimension
        self.left = (
            torch.randn(
                (side_size, rows),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            * standard_deviation
        )
        self.right = (
            torch.randn(
                (side_size, columns),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            * standard_deviation
        )

    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        if tuple(matrix.shape) != self.input_shape:
            raise ValueError(
                f"projection expects matrix shape {self.input_shape}, "
                f"received {tuple(matrix.shape)}"
            )
        return (self.left @ matrix @ self.right.T).flatten()


class BlockDiagonalProjector:
    def __init__(
        self,
        input_shapes: dict[str, tuple[int, int] | torch.Size],
        output_dimension: int,
        *,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not input_shapes:
            raise ValueError("block-diagonal projection requires matrices")
        self.input_shapes = {name: tuple(shape) for name, shape in input_shapes.items()}
        self.matrix_names = tuple(input_shapes)
        self.output_dimension = output_dimension
        self.matrix_projectors = {
            name: TwoSidedProjector(
                shape,
                output_dimension,
                seed=seed + index,
                device=device,
                dtype=dtype,
            )
            for index, (name, shape) in enumerate(input_shapes.items())
        }

    def __call__(self, matrices: dict[str, torch.Tensor]) -> torch.Tensor:
        if matrices.keys() != self.input_shapes.keys():
            raise ValueError("projection matrices do not match the block layout")
        projected = [
            self.matrix_projectors[name](matrices[name]) for name in self.matrix_names
        ]
        return torch.stack(projected).sum(dim=0)
