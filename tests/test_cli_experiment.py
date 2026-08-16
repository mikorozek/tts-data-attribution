from __future__ import annotations

from pathlib import Path

import pytest

from tts_data_attribution.cli.main import main
from tts_data_attribution.dataset import Utterance, UtteranceDataset
from tts_data_attribution.experiment import ExperimentConfig


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
        [
            Utterance(
                id="2-0",
                text="First",
                speaker="0",
                dialogue="2",
                audio_path="data/2/0_0_d2.wav",
                audio_codes=[[7] * 16],
            )
        ]
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
        "--device",
        "cpu",
        "--root",
        str(root / "experiments"),
    ]


def test_init_creates_the_workspace_with_its_config(workspace: Path) -> None:
    assert main(init_arguments(workspace)) == 0

    config = ExperimentConfig.from_yaml(workspace / "experiments/study/config.yaml")
    assert config == ExperimentConfig(
        dataset=workspace / "encoded.jsonl",
        model="qwen3-tts",
        model_path=workspace / "model",
    )
    assert FakeUpstreamModel.loaded == [(str(workspace / "model"), "cpu")]


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
