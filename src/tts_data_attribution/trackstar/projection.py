from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class RandomProjection(ABC):
    def __init__(
        self,
        input_shapes: Sequence[Sequence[int]],
        output_dimension: int,
        *,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shapes = tuple(tuple(shape) for shape in input_shapes)
        if not shapes:
            raise ValueError("random projection requires input tensors")
        if any(
            not shape or any(dimension < 1 for dimension in shape) for shape in shapes
        ):
            raise ValueError("random projection input dimensions must be positive")
        if output_dimension < 1:
            raise ValueError("random projection output dimension must be positive")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("random projection dtype must be floating point")

        self.input_shapes = shapes
        self.output_dimension = output_dimension
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype

    def _prepare(
        self,
        gradients: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(gradients) != len(self.input_shapes):
            raise ValueError("gradient count does not match the projection input")
        prepared = []
        for index, (gradient, shape) in enumerate(
            zip(gradients, self.input_shapes, strict=True)
        ):
            if tuple(gradient.shape) != shape:
                raise ValueError(f"gradient shape differs for projection input {index}")
            if gradient.device != self.device:
                raise ValueError(
                    f"gradient device differs for projection input {index}"
                )
            if not gradient.is_floating_point():
                raise ValueError("random projection inputs must be floating point")
            if not torch.isfinite(gradient).all():
                raise ValueError("random projection inputs must be finite")
            prepared.append(gradient.to(dtype=self.dtype))
        return tuple(prepared)

    @abstractmethod
    def __call__(self, gradients: tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError


class TwoSidedRandomProjection(RandomProjection):
    def __init__(
        self,
        input_shapes: Sequence[Sequence[int]],
        output_dimension: int,
        *,
        seed: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            input_shapes,
            output_dimension,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        side_size = math.isqrt(output_dimension)
        if side_size * side_size != output_dimension:
            raise ValueError("two-sided projection output dimension must be a square")
        if any(len(shape) != 2 for shape in self.input_shapes):
            raise ValueError("two-sided projection inputs must be matrices")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        standard_deviation = side_size**-0.5
        left_matrices = []
        right_matrices = []
        for rows, columns in self.input_shapes:
            left = torch.empty(
                (side_size, rows),
                device=self.device,
                dtype=self.dtype,
            )
            right = torch.empty(
                (side_size, columns),
                device=self.device,
                dtype=self.dtype,
            )
            left.normal_(
                mean=0.0,
                std=standard_deviation,
                generator=generator,
            )
            right.normal_(
                mean=0.0,
                std=standard_deviation,
                generator=generator,
            )
            left_matrices.append(left)
            right_matrices.append(right)
        self.left_matrices = tuple(left_matrices)
        self.right_matrices = tuple(right_matrices)
        self.side_size = side_size

    def __call__(self, gradients: tuple[torch.Tensor, ...]) -> torch.Tensor:
        gradients = self._prepare(gradients)
        projected = torch.zeros(
            (self.side_size, self.side_size),
            device=self.device,
            dtype=self.dtype,
        )
        for gradient, left, right in zip(
            gradients,
            self.left_matrices,
            self.right_matrices,
            strict=True,
        ):
            projected.add_(left @ gradient @ right.T)
        return projected.flatten()
