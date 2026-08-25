from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pytest
import torch

from tts_data_attribution.evaluation import (
    compute_lds,
    create_membership_matrix,
    save_immutable_torch_artifact,
)


def test_lds_uses_subset_membership_spearman_correlation_and_mean() -> None:
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

    result = compute_lds(attributions, membership, observed_responses)

    assert result == pytest.approx(1.0 / 15.0)


def test_create_membership_matrix_preserves_subset_and_training_order() -> None:
    result = create_membership_matrix(
        ["train-0", "train-1", "train-2"],
        [["train-2", "train-0"], ["train-1"]],
    )

    torch.testing.assert_close(
        result,
        torch.tensor([[True, False, True], [False, True, False]]),
    )


def test_create_membership_matrix_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="unknown ID: absent"):
        create_membership_matrix(["train-0"], [["absent"]])


def test_lds_rejects_an_undefined_query_correlation() -> None:
    with pytest.raises(ValueError, match="undefined for query at column 0"):
        compute_lds(
            torch.ones(2, 1),
            torch.eye(2, dtype=torch.bool),
            torch.tensor([[1.0], [2.0]]),
        )


def test_lds_rejects_non_finite_membership_sums() -> None:
    maximum = torch.finfo(torch.float64).max
    with pytest.raises(ValueError, match="predicted responses must be finite"):
        compute_lds(
            torch.full((2, 1), maximum, dtype=torch.float64),
            torch.ones((2, 2), dtype=torch.bool),
            torch.tensor([[1.0], [2.0]]),
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
