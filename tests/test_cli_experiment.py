from __future__ import annotations

from pathlib import Path

import pytest

from tts_data_attribution.cli.main import main
from tts_data_attribution.dataset import Utterance, UtteranceDataset
from tts_data_attribution.experiment import ExperimentManifest, Plan


class FakeUpstreamModel:
    loaded: list[tuple[str, str]] = []

    @classmethod
    def from_pretrained(cls, path: str, device_map: str) -> FakeUpstreamModel:
        cls.loaded.append((path, device_map))
        return cls()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dataset = tmp_path / "encoded.jsonl"
    UtteranceDataset(
        Utterance(
            id=f"{speaker}-{index}",
            text="hi",
            speaker=speaker,
            dialogue=str(index),
            audio_path=f"data/{speaker}-{index}.wav",
            audio_codes=[[7] * 16],
        )
        for speaker in ("0", "1")
        for index in range(6)
    ).to_jsonl(dataset)
    (tmp_path / "model").mkdir()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setitem(__import__("sys").modules, "qwen_tts", FakeQwenTts())
    FakeUpstreamModel.loaded.clear()
    return tmp_path


class FakeQwenTts:
    Qwen3TTSModel = FakeUpstreamModel


def init_arguments(root: Path, name: str = "study") -> list[str]:
    return [
        "experiment",
        "init",
        name,
        "--dataset",
        str(root / "encoded.jsonl"),
        "--model",
        "qwen3-tts",
        "--model-path",
        str(root / "model"),
        "--training-pool-size",
        "8",
        "--subset-count",
        "3",
        "--subset-size",
        "4",
        "--speaker-count",
        "2",
        "--seed",
        "7",
        "--device",
        "cpu",
        "--root",
        str(root / "experiments"),
    ]


def test_init_writes_the_manifest_and_the_plan(workspace: Path) -> None:
    assert main(init_arguments(workspace)) == 0

    directory = workspace / "experiments/study"
    manifest = ExperimentManifest.from_yaml(directory / "manifest.yaml")
    assert manifest == ExperimentManifest(
        dataset=workspace / "encoded.jsonl",
        model="qwen3-tts",
        model_path=workspace / "model",
        training_pool_size=8,
        subset_count=3,
        subset_size=4,
        speaker_count=2,
        seed=7,
    )
    plan = Plan.from_json(directory / "plan.json")
    assert plan == Plan.sample(manifest, UtteranceDataset.from_jsonl(manifest.dataset))
    assert FakeUpstreamModel.loaded == [(str(workspace / "model"), "cpu")]


def test_init_fails_cleanly_when_the_sampling_does_not_fit(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = init_arguments(workspace)
    arguments[arguments.index("--training-pool-size") + 1] = "11"

    assert main(arguments) == 1
    assert "training_pool_size 11 exceeds" in capsys.readouterr().err
    assert FakeUpstreamModel.loaded == []
    assert not (workspace / "experiments/study").exists()


def test_init_refuses_an_existing_experiment(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / "experiments/study").mkdir(parents=True)

    assert main(init_arguments(workspace)) == 1
    assert "already exists" in capsys.readouterr().err
    assert FakeUpstreamModel.loaded == []


def test_init_fails_cleanly_when_the_dataset_is_missing(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / "encoded.jsonl").unlink()

    assert main(init_arguments(workspace)) == 1
    assert "encoded dataset not found" in capsys.readouterr().err


def test_init_fails_cleanly_when_the_dataset_is_not_ours(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / "encoded.jsonl").write_text('{"id": "2-0"}\n', encoding="utf-8")

    assert main(init_arguments(workspace)) == 1
    assert "not an encoded utterance file" in capsys.readouterr().err
    assert not (workspace / "experiments/study").exists()


def test_init_fails_cleanly_when_the_model_does_not_load(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(cls, path: str, device_map: str) -> None:
        raise OSError("no config.json")

    monkeypatch.setattr(FakeUpstreamModel, "from_pretrained", classmethod(explode))

    assert main(init_arguments(workspace)) == 1
    assert "cannot load qwen3-tts" in capsys.readouterr().err
    assert not (workspace / "experiments/study").exists()
