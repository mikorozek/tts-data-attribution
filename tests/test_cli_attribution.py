from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from tts_data_attribution.cli.main import main
from tts_data_attribution.experiment import Plan


def create_projected_pools(root: Path) -> Path:
    experiment = root / "experiments" / "study"
    experiment.mkdir(parents=True)
    plan = Plan(
        references={"speaker": "reference"},
        training_pool=["train-0", "train-1"],
        validation_pool=[],
        query_pool=["query-0"],
        subsets={},
    )
    plan.to_json(experiment / "plan.json")
    projection = experiment / "training-runs" / "run-1" / "projections" / "two-sided-4"
    projected = projection / "projected"
    projected.mkdir(parents=True)
    (projection / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "two-sided",
                "training_checkpoint": "training-pool/checkpoints/epoch-0002",
                "output_dimension": 4,
                "seed": 7,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "ids": plan.training_pool,
            "projected_gradients": torch.tensor(
                [[2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
        },
        projected / "training-pool.pt",
    )
    torch.save(
        {
            "ids": plan.query_pool,
            "projected_gradients": torch.tensor([[3.0, 0.0, 0.0, 0.0]]),
        },
        projected / "query-pool.pt",
    )
    return projection


def compute_arguments(task_weight: str = "0.5") -> list[str]:
    return [
        "attribution",
        "compute",
        "study",
        "run-1",
        "two-sided-4",
        "--task-weight",
        task_weight,
        "--device",
        "cpu",
    ]


def test_attribution_compute_uses_a_pseudoinverse_for_a_singular_hessian(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)

    assert main(compute_arguments()) == 0

    result = torch.load(
        projection / "attribution" / "attributions.pt", weights_only=True
    )
    torch.testing.assert_close(result, torch.ones(2, 1))
    assert yaml.safe_load((projection / "attribution" / "config.yaml").read_text()) == {
        "task_weight": 0.5
    }


def test_attribution_compute_rejects_projected_ids_outside_the_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    query_path = projection / "projected" / "query-pool.pt"
    query = torch.load(query_path, weights_only=True)
    query["ids"] = ["different"]
    torch.save(query, query_path)

    assert main(compute_arguments()) == 1

    assert "query IDs differ" in capsys.readouterr().err
    assert not (projection / "attribution").exists()


def test_attribution_compute_refuses_to_replace_its_method_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    output = projection / "attribution"
    output.mkdir()

    assert main(compute_arguments()) == 1

    assert "already exists" in capsys.readouterr().err


def test_attribution_compute_rejects_tensor_rows_that_do_not_match_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    training_path = projection / "projected" / "training-pool.pt"
    training = torch.load(training_path, weights_only=True)
    training["projected_gradients"] = training["projected_gradients"][:1]
    torch.save(training, training_path)

    assert main(compute_arguments()) == 1

    assert "dimensions or dtype are invalid" in capsys.readouterr().err
    assert not (projection / "attribution").exists()


def test_attribution_normalization_accepts_nonzero_norm_below_machine_epsilon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    training_path = projection / "projected" / "training-pool.pt"
    training = torch.load(training_path, weights_only=True)
    training["projected_gradients"] = torch.tensor(
        [[2e-20, 0.0, 0.0, 0.0], [1e-20, 0.0, 0.0, 0.0]]
    )
    torch.save(training, training_path)

    assert main(compute_arguments("1.0")) == 0

    result = torch.load(
        projection / "attribution" / "attributions.pt", weights_only=True
    )
    torch.testing.assert_close(result, torch.ones(2, 1))
