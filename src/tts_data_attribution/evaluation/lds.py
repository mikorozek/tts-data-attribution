from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.stats import rankdata

Correlation = Literal["spearman", "pearson"]
Aggregation = Literal["mean", "median"]
Uncertainty = Literal["none", "subset-bootstrap"]


@dataclass(frozen=True)
class LDSConfiguration:
    correlation: Correlation = "spearman"
    aggregation: Aggregation = "mean"
    uncertainty: Uncertainty = "none"
    bootstrap_samples: int = 0
    confidence_level: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if self.correlation not in {"spearman", "pearson"}:
            raise ValueError(f"unsupported LDS correlation: {self.correlation}")
        if self.aggregation not in {"mean", "median"}:
            raise ValueError(f"unsupported LDS aggregation: {self.aggregation}")
        if self.uncertainty not in {"none", "subset-bootstrap"}:
            raise ValueError(f"unsupported LDS uncertainty: {self.uncertainty}")
        if isinstance(self.bootstrap_samples, bool) or self.bootstrap_samples < 0:
            raise ValueError("bootstrap samples must not be negative")
        if self.uncertainty == "none" and self.bootstrap_samples != 0:
            raise ValueError("bootstrap samples require subset-bootstrap uncertainty")
        if self.uncertainty == "subset-bootstrap" and self.bootstrap_samples < 1:
            raise ValueError("subset-bootstrap uncertainty requires bootstrap samples")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must be between zero and one")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("LDS seed must not be negative")

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "correlation": self.correlation,
            "aggregation": self.aggregation,
            "uncertainty": self.uncertainty,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class LDSResult:
    predicted_responses: torch.Tensor
    per_query_correlations: torch.Tensor
    aggregate_correlation: float
    bootstrap_aggregate_correlations: torch.Tensor | None
    confidence_interval: tuple[float, float] | None


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
    configuration: LDSConfiguration,
) -> LDSResult:
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
    expected_response_shape = (subset_count, attributions.shape[1])
    if observed_responses.shape != expected_response_shape:
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

    correlations = _column_correlations(
        predicted_responses,
        observed_responses,
        configuration.correlation,
    )
    aggregate = _aggregate(correlations, configuration.aggregation)
    bootstrap_aggregates = _bootstrap_aggregates(
        predicted_responses,
        observed_responses,
        configuration,
    )
    confidence_interval = _confidence_interval(
        bootstrap_aggregates,
        configuration.confidence_level,
    )
    return LDSResult(
        predicted_responses=predicted_responses,
        per_query_correlations=correlations,
        aggregate_correlation=float(aggregate),
        bootstrap_aggregate_correlations=bootstrap_aggregates,
        confidence_interval=confidence_interval,
    )


def _column_correlations(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    correlation: Correlation,
) -> torch.Tensor:
    if correlation == "spearman":
        predicted = torch.from_numpy(
            np.asarray(rankdata(predicted.numpy(), axis=0), dtype=np.float64)
        )
        observed = torch.from_numpy(
            np.asarray(rankdata(observed.numpy(), axis=0), dtype=np.float64)
        )
    predicted = predicted - predicted.mean(dim=0)
    observed = observed - observed.mean(dim=0)
    denominators = torch.linalg.vector_norm(
        predicted, dim=0
    ) * torch.linalg.vector_norm(observed, dim=0)
    undefined = torch.nonzero(denominators == 0).flatten()
    if len(undefined) > 0:
        raise ValueError(
            f"LDS correlation is undefined for query at column {int(undefined[0])}"
        )
    correlations = (predicted * observed).sum(dim=0) / denominators
    non_finite = torch.nonzero(~torch.isfinite(correlations)).flatten()
    if len(non_finite) > 0:
        raise ValueError(
            f"LDS correlation is non-finite for query at column {int(non_finite[0])}"
        )
    return correlations.clamp(min=-1.0, max=1.0)


def _aggregate(values: torch.Tensor, aggregation: Aggregation) -> torch.Tensor:
    if aggregation == "mean":
        return values.mean()
    return torch.quantile(values, 0.5)


def _bootstrap_aggregates(
    predicted_responses: torch.Tensor,
    observed_responses: torch.Tensor,
    configuration: LDSConfiguration,
) -> torch.Tensor | None:
    if configuration.uncertainty == "none":
        return None
    generator = torch.Generator(device="cpu").manual_seed(configuration.seed)
    aggregates = []
    attempts = 0
    maximum_attempts = configuration.bootstrap_samples * 10
    while len(aggregates) < configuration.bootstrap_samples:
        attempts += 1
        if attempts > maximum_attempts:
            raise ValueError("could not obtain finite subset-bootstrap samples")
        indices = torch.randint(
            len(predicted_responses),
            (len(predicted_responses),),
            generator=generator,
        )
        try:
            correlations = _column_correlations(
                predicted_responses[indices],
                observed_responses[indices],
                configuration.correlation,
            )
        except ValueError:
            continue
        aggregates.append(_aggregate(correlations, configuration.aggregation))
    return torch.stack(aggregates)


def _confidence_interval(
    bootstrap_aggregates: torch.Tensor | None,
    confidence_level: float,
) -> tuple[float, float] | None:
    if bootstrap_aggregates is None:
        return None
    tail = (1.0 - confidence_level) / 2.0
    bounds = torch.quantile(
        bootstrap_aggregates,
        torch.tensor([tail, 1.0 - tail], dtype=bootstrap_aggregates.dtype),
    )
    return float(bounds[0]), float(bounds[1])
