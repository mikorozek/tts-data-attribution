from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata


def create_membership_matrix(
    training_ids: list[str],
    subset_training_ids: list[list[str]],
) -> torch.Tensor:
    if not training_ids:
        raise ValueError("LDS training IDs must not be empty")
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("LDS training IDs must be unique")
    positions = {identifier: index for index, identifier in enumerate(training_ids)}
    membership = torch.zeros(
        (len(subset_training_ids), len(training_ids)),
        dtype=torch.bool,
    )
    for row, identifiers in enumerate(subset_training_ids):
        if not identifiers:
            raise ValueError(f"LDS subset at row {row} must not be empty")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"LDS subset at row {row} contains duplicate IDs")
        for identifier in identifiers:
            try:
                membership[row, positions[identifier]] = True
            except KeyError as error:
                raise ValueError(
                    f"LDS subset at row {row} contains an unknown ID: {identifier}"
                ) from error
    return membership


def compute_lds(
    attributions: torch.Tensor,
    membership: torch.Tensor,
    observed_responses: torch.Tensor,
) -> float:
    if not isinstance(attributions, torch.Tensor) or attributions.ndim != 2:
        raise ValueError("LDS attributions must be a two-dimensional tensor")
    if not attributions.is_floating_point():
        raise ValueError("LDS attributions must be floating point")
    if not isinstance(membership, torch.Tensor) or membership.ndim != 2:
        raise ValueError("LDS membership must be a two-dimensional tensor")
    if membership.dtype != torch.bool:
        raise ValueError("LDS membership must be boolean")
    if not isinstance(observed_responses, torch.Tensor) or observed_responses.ndim != 2:
        raise ValueError("LDS observed responses must be a two-dimensional tensor")
    if not observed_responses.is_floating_point():
        raise ValueError("LDS observed responses must be floating point")
    subset_count, training_count = membership.shape
    if subset_count < 2:
        raise ValueError("LDS requires at least two subsets")
    if training_count < 1 or attributions.shape[1] < 1:
        raise ValueError("LDS requires training examples and queries")
    if attributions.shape[0] != training_count:
        raise ValueError("LDS attribution and membership dimensions differ")
    if observed_responses.shape != (subset_count, attributions.shape[1]):
        raise ValueError("LDS observed response dimensions are invalid")
    if not membership.any(dim=1).all():
        raise ValueError("LDS subset membership rows must not be empty")

    attributions = attributions.detach().to(device="cpu", dtype=torch.float64)
    membership = membership.detach().to(device="cpu")
    observed_responses = observed_responses.detach().to(
        device="cpu", dtype=torch.float64
    )
    if not torch.isfinite(attributions).all():
        raise ValueError("LDS attributions must be finite")
    if not torch.isfinite(observed_responses).all():
        raise ValueError("LDS observed responses must be finite")
    predicted_responses = membership.to(torch.float64) @ attributions
    if not torch.isfinite(predicted_responses).all():
        raise ValueError("LDS predicted responses must be finite")

    predicted_ranks = torch.from_numpy(
        np.asarray(rankdata(predicted_responses.numpy(), axis=0), dtype=np.float64)
    )
    observed_ranks = torch.from_numpy(
        np.asarray(rankdata(observed_responses.numpy(), axis=0), dtype=np.float64)
    )
    predicted_ranks -= predicted_ranks.mean(dim=0)
    observed_ranks -= observed_ranks.mean(dim=0)
    denominators = torch.linalg.vector_norm(
        predicted_ranks, dim=0
    ) * torch.linalg.vector_norm(observed_ranks, dim=0)
    undefined = torch.nonzero(denominators == 0).flatten()
    if len(undefined) > 0:
        raise ValueError(
            f"LDS correlation is undefined for query at column {int(undefined[0])}"
        )
    correlations = (predicted_ranks * observed_ranks).sum(dim=0) / denominators
    non_finite = torch.nonzero(~torch.isfinite(correlations)).flatten()
    if len(non_finite) > 0:
        raise ValueError(
            f"LDS correlation is non-finite for query at column {int(non_finite[0])}"
        )
    return float(correlations.clamp(min=-1.0, max=1.0).mean())
