from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def stack_projected_gradients(
    examples: Iterable[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    iterator = iter(examples)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError(
            "cannot stack an empty projected gradient collection"
        ) from error
    if not first:
        raise ValueError("projected gradients must contain at least one block")

    block_names = tuple(first)
    for value in first.values():
        if value.ndim != 1 or not value.is_floating_point():
            raise ValueError("projected gradient blocks must be floating-point vectors")
        if not torch.isfinite(value).all():
            raise ValueError("projected gradient blocks must be finite")
    blocks = {name: [first[name]] for name in block_names}
    shapes = {name: first[name].shape for name in block_names}
    for example in iterator:
        if example.keys() != blocks.keys():
            raise ValueError("projected gradient blocks must match")
        for name in block_names:
            if example[name].shape != shapes[name]:
                raise ValueError(f"projected gradient shape changed for {name}")
            if not torch.isfinite(example[name]).all():
                raise ValueError("projected gradient blocks must be finite")
            blocks[name].append(example[name])
    return {name: torch.stack(values) for name, values in blocks.items()}


class TrackStarTransform:
    def __init__(
        self,
        training: dict[str, torch.Tensor],
        task: dict[str, torch.Tensor],
        *,
        task_weight: float,
        singular_value_rcond: float | None = None,
    ) -> None:
        if not 0.0 <= task_weight <= 1.0:
            raise ValueError("task weight must be between zero and one")
        if singular_value_rcond is not None and singular_value_rcond < 0.0:
            raise ValueError("singular value rcond must not be negative")
        if not training:
            raise ValueError("training representations must contain blocks")
        if training.keys() != task.keys():
            raise ValueError("training and task representation blocks must match")

        block_names = tuple(training)
        feature_dimensions = {}
        bases = {}
        inverse_singular_values = {}
        hessian_eigenvalues = {}
        for name in block_names:
            training_block = training[name]
            task_block = task[name]
            if training_block.ndim != 2 or task_block.ndim != 2:
                raise ValueError("Hessian representations must be matrices")
            if training_block.shape[0] < 1 or task_block.shape[0] < 1:
                raise ValueError("Hessian representation pools must not be empty")
            if training_block.shape[1] != task_block.shape[1]:
                raise ValueError(f"Hessian feature dimension differs for {name}")
            if (
                training_block.device != task_block.device
                or training_block.dtype != task_block.dtype
            ):
                raise ValueError(f"Hessian representation type differs for {name}")
            if (
                not training_block.is_floating_point()
                or not task_block.is_floating_point()
            ):
                raise ValueError("Hessian representations must be floating point")

            mixed = torch.cat(
                [
                    math.sqrt(1.0 - task_weight) * training_block,
                    math.sqrt(task_weight) * task_block,
                ]
            )
            _, values, vectors = torch.linalg.svd(mixed, full_matrices=False)
            default_rcond = (
                torch.finfo(values.dtype).eps * max(mixed.shape)
                if singular_value_rcond is None
                else singular_value_rcond
            )
            threshold = default_rcond * values[0]
            retained = values > threshold
            if not torch.any(retained):
                raise ValueError(
                    f"Hessian approximation has no nonzero directions for {name}"
                )

            feature_dimensions[name] = training_block.shape[1]
            bases[name] = vectors[retained]
            inverse_singular_values[name] = values[retained].reciprocal()
            hessian_eigenvalues[name] = values.square()

        self.block_names = block_names
        self.feature_dimensions = feature_dimensions
        self.hessian_eigenvalues = hessian_eigenvalues
        self._bases = bases
        self._inverse_singular_values = inverse_singular_values

    def __call__(
        self,
        projected: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if projected.keys() != self.feature_dimensions.keys():
            raise ValueError(
                "projected gradient blocks do not match the Hessian transform"
            )

        transformed = []
        leading_shape = None
        for name in self.block_names:
            block = projected[name]
            if block.ndim not in {1, 2}:
                raise ValueError("projected gradients must be vectors or matrices")
            if block.shape[-1] != self.feature_dimensions[name]:
                raise ValueError(f"projected gradient dimension differs for {name}")
            if not torch.isfinite(block).all():
                raise ValueError("projected gradients must be finite")
            if leading_shape is None:
                leading_shape = block.shape[:-1]
            elif block.shape[:-1] != leading_shape:
                raise ValueError("projected gradient batch dimensions must match")

            basis = self._bases[name]
            if block.device != basis.device or block.dtype != basis.dtype:
                raise ValueError(f"projected gradient type differs for {name}")
            inverse_values = self._inverse_singular_values[name]
            coordinates = block @ basis.T
            transformed.append((coordinates * inverse_values) @ basis)

        combined = torch.cat(transformed, dim=-1)
        norms = torch.linalg.vector_norm(combined, dim=-1, keepdim=True)
        if torch.any(norms <= torch.finfo(combined.dtype).eps):
            raise ValueError("cannot normalize a zero TrackStar representation")
        return combined / norms


class TrackStar:
    def __init__(self, training: dict[str, torch.Tensor]) -> None:
        self.training = training

    def fit(
        self,
        queries: dict[str, torch.Tensor],
        *,
        task_weight: float,
        singular_value_rcond: float | None = None,
    ) -> TrackStarTransform:
        return TrackStarTransform(
            self.training,
            queries,
            task_weight=task_weight,
            singular_value_rcond=singular_value_rcond,
        )

    def score(
        self,
        queries: dict[str, torch.Tensor],
        *,
        task_weight: float,
        singular_value_rcond: float | None = None,
    ) -> torch.Tensor:
        transform = self.fit(
            queries,
            task_weight=task_weight,
            singular_value_rcond=singular_value_rcond,
        )
        training_encodings = transform(self.training)
        query_encodings = transform(queries)
        return attribution_scores(training_encodings, query_encodings)


def attribution_scores(
    training: torch.Tensor,
    queries: torch.Tensor,
) -> torch.Tensor:
    if training.ndim != 2 or queries.ndim != 2:
        raise ValueError("TrackStar representations must be matrices")
    if training.shape[1] != queries.shape[1]:
        raise ValueError("TrackStar representation dimensions must match")
    if not torch.isfinite(training).all() or not torch.isfinite(queries).all():
        raise ValueError("TrackStar representations must be finite")
    return queries @ training.T
