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
        training_pool=["train-0", "train-1", "train-2"],
        validation_pool=[],
        query_pool=["query-0", "query-1"],
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
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [-2.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0],
                ]
            ),
        },
        projected / "training-pool.pt",
    )
    torch.save(
        {
            "ids": plan.query_pool,
            "projected_gradients": torch.tensor(
                [[4.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]]
            ),
        },
        projected / "query-pool.pt",
    )
    return projection


def arguments() -> list[str]:
    return [
        "projected-cosine",
        "compute",
        "study",
        "run-1",
        "two-sided-4",
        "--device",
        "cpu",
    ]


def test_projected_cosine_compute_saves_known_direction_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)

    assert main(arguments()) == 0

    result = torch.load(
        projection / "projected-cosine" / "attributions.pt",
        weights_only=True,
    )
    torch.testing.assert_close(
        result,
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
    )


def test_projected_cosine_accepts_a_tiny_nonzero_norm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    training_path = projection / "projected" / "training-pool.pt"
    training = torch.load(training_path, weights_only=True)
    training["projected_gradients"][0] = torch.tensor([1e-20, 0.0, 0.0, 0.0])
    torch.save(training, training_path)

    assert main(arguments()) == 0

    result = torch.load(
        projection / "projected-cosine" / "attributions.pt",
        weights_only=True,
    )
    assert result[0, 0] == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (torch.tensor([0.0, 0.0, 0.0, 0.0]), "zero projected gradient"),
        (torch.tensor([float("nan"), 0.0, 0.0, 0.0]), "must be finite"),
        (torch.tensor([float("inf"), 0.0, 0.0, 0.0]), "must be finite"),
    ],
)
def test_projected_cosine_rejects_invalid_projected_gradients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    replacement: torch.Tensor,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    training_path = projection / "projected" / "training-pool.pt"
    training = torch.load(training_path, weights_only=True)
    training["projected_gradients"][0] = replacement
    torch.save(training, training_path)

    assert main(arguments()) == 1

    assert message in capsys.readouterr().err
    assert not (projection / "projected-cosine" / "attributions.pt").exists()


def test_projected_cosine_rejects_integer_projected_gradients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    training_path = projection / "projected" / "training-pool.pt"
    training = torch.load(training_path, weights_only=True)
    training["projected_gradients"] = training["projected_gradients"].to(torch.int64)
    torch.save(training, training_path)

    assert main(arguments()) == 1

    assert "dimensions or dtype are invalid" in capsys.readouterr().err


def test_projected_cosine_rejects_query_ids_outside_the_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    query_path = projection / "projected" / "query-pool.pt"
    query = torch.load(query_path, weights_only=True)
    query["ids"] = ["different", "query-1"]
    torch.save(query, query_path)

    assert main(arguments()) == 1

    assert "query IDs differ" in capsys.readouterr().err


def test_projected_cosine_refuses_to_replace_attributions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    output = projection / "projected-cosine" / "attributions.pt"
    output.parent.mkdir()
    output.write_bytes(b"existing")

    assert main(arguments()) == 1

    assert "already exists" in capsys.readouterr().err
    assert output.read_bytes() == b"existing"


def test_projected_cosine_rejects_tensor_rows_that_do_not_match_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    projection = create_projected_pools(tmp_path)
    training_path = projection / "projected" / "training-pool.pt"
    training = torch.load(training_path, weights_only=True)
    training["projected_gradients"] = training["projected_gradients"][:2]
    torch.save(training, training_path)

    assert main(arguments()) == 1

    assert "dimensions or dtype are invalid" in capsys.readouterr().err
