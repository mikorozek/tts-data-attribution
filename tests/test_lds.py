from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pytest
import torch

from tts_data_attribution.evaluation import (
    LDSConfiguration,
    compute_lds,
    create_membership_matrix,
    save_immutable_torch_artifact,
)


def test_lds_uses_membership_masks_and_aggregates_per_query_correlations() -> None:
    training_ids = ["train-0", "train-1", "train-2", "train-3"]
    membership = create_membership_matrix(
        training_ids,
        [
            ["train-0", "train-1"],
            ["train-1", "train-2"],
            ["train-2", "train-3"],
            ["train-0", "train-3"],
        ],
    )
    attributions = torch.tensor(
        [
            [1.0, 8.0, 1.0],
            [2.0, 4.0, 4.0],
            [4.0, 2.0, 2.0],
            [8.0, 1.0, 8.0],
        ],
        dtype=torch.bfloat16,
    )
    observed_responses = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
            [4.0, 4.0, 2.0],
            [3.0, 2.0, 3.0],
        ]
    )

    result = compute_lds(
        attributions,
        membership,
        observed_responses,
        LDSConfiguration(),
    )

    torch.testing.assert_close(
        result.predicted_responses,
        torch.tensor(
            [
                [3.0, 12.0, 5.0],
                [6.0, 6.0, 6.0],
                [12.0, 3.0, 10.0],
                [9.0, 9.0, 9.0],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        result.per_query_correlations,
        torch.tensor([1.0, -1.0, 0.2], dtype=torch.float64),
    )
    assert result.aggregate_correlation == pytest.approx(1.0 / 15.0)
    assert result.bootstrap_aggregate_correlations is None
    assert result.confidence_interval is None


def test_lds_supports_configured_subset_bootstrap_uncertainty() -> None:
    membership = torch.eye(4, dtype=torch.bool)
    attributions = torch.tensor([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    observed_responses = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    configuration = LDSConfiguration(
        correlation="pearson",
        aggregation="median",
        uncertainty="subset-bootstrap",
        bootstrap_samples=100,
        confidence_level=0.9,
        seed=17,
    )

    first = compute_lds(
        attributions,
        membership,
        observed_responses,
        configuration,
    )
    second = compute_lds(
        attributions,
        observed_responses=torch.clone(observed_responses),
        membership=membership,
        configuration=configuration,
    )

    assert first.aggregate_correlation == pytest.approx(0.0)
    assert first.bootstrap_aggregate_correlations is not None
    assert second.bootstrap_aggregate_correlations is not None
    assert first.bootstrap_aggregate_correlations.shape == (100,)
    torch.testing.assert_close(
        first.bootstrap_aggregate_correlations,
        second.bootstrap_aggregate_correlations,
    )
    assert first.confidence_interval == second.confidence_interval
    assert first.confidence_interval is not None
    assert first.confidence_interval[0] >= -1.0
    assert first.confidence_interval[1] <= 1.0


def test_lds_rejects_an_undefined_query_correlation() -> None:
    with pytest.raises(ValueError, match="undefined for query at column 0"):
        compute_lds(
            torch.ones(2, 1),
            torch.eye(2, dtype=torch.bool),
            torch.tensor([[1.0], [2.0]]),
            LDSConfiguration(),
        )


def test_create_membership_matrix_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="unknown ID: absent"):
        create_membership_matrix(["train-0"], [["absent"]])


def test_lds_rejects_non_finite_membership_sums() -> None:
    maximum = torch.finfo(torch.float64).max
    with pytest.raises(ValueError, match="predicted responses must be finite"):
        compute_lds(
            torch.full((2, 1), maximum, dtype=torch.float64),
            torch.ones((2, 2), dtype=torch.bool),
            torch.tensor([[1.0], [2.0]]),
            LDSConfiguration(),
        )


def test_immutable_artifact_refuses_replacement_and_cleans_temporary_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation.pt"
    save_immutable_torch_artifact(path, {"value": torch.tensor([1.0])})

    with pytest.raises(FileExistsError):
        save_immutable_torch_artifact(path, {"value": torch.tensor([2.0])})

    result = torch.load(path, weights_only=True)
    torch.testing.assert_close(result["value"], torch.tensor([1.0]))
    assert sorted(tmp_path.iterdir()) == [path]


def test_immutable_artifact_removes_a_partial_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_save(artifact: object, file: BinaryIO) -> None:
        file.write(b"partial")
        raise RuntimeError("disk failure")

    monkeypatch.setattr(torch, "save", fail_save)
    path = tmp_path / "evaluation.pt"

    with pytest.raises(RuntimeError, match="disk failure"):
        save_immutable_torch_artifact(path, {"value": torch.tensor([1.0])})

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
