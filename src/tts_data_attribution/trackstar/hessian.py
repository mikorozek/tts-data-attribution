from __future__ import annotations

import torch


class GaussNewtonHessianApproximation:
    def __init__(self, *, task_weight: float) -> None:
        if not 0.0 <= task_weight <= 1.0:
            raise ValueError("task weight must be between zero and one")
        self.task_weight = task_weight

    def compute(
        self,
        training: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        if training.ndim != 2 or queries.ndim != 2:
            raise ValueError("projected gradients must be matrices")
        if training.shape[0] < 1 or queries.shape[0] < 1:
            raise ValueError("projected gradient collections must not be empty")
        if training.shape[1] != queries.shape[1]:
            raise ValueError("projected gradient dimensions must match")
        if training.device != queries.device or training.dtype != queries.dtype:
            raise ValueError("projected gradient types must match")
        if not training.is_floating_point():
            raise ValueError("projected gradients must be floating point")
        if not torch.isfinite(training).all() or not torch.isfinite(queries).all():
            raise ValueError("projected gradients must be finite")

        training_hessian = training.T @ training
        query_hessian = queries.T @ queries
        return (1.0 - self.task_weight) * training_hessian + (
            self.task_weight * query_hessian
        )
