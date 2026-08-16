from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tts_data_attribution.cli.main import main
from tts_data_attribution.dataset import Utterance, UtteranceDataset
from tts_data_attribution.experiment import ExperimentManifest, Plan
from tts_data_attribution.models import SPEAKER_REFERENCE_AUDIO_ENCODERS


class FakeSpeakerReferenceAudioEncoder:
    loaded: list[tuple[str, str]] = []
    encoded: list[str] = []

    @classmethod
    def from_pretrained(
        cls, model_path: Path, device: str
    ) -> FakeSpeakerReferenceAudioEncoder:
        cls.loaded.append((str(model_path), device))
        return cls()

    def encode(self, reference_audio_path: Path) -> torch.Tensor:
        FakeSpeakerReferenceAudioEncoder.encoded.append(reference_audio_path.name)
        return torch.full((4,), float(int(reference_audio_path.stem.split("-")[1])))


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    ).to_jsonl(tmp_path / "encoded.jsonl")
    (tmp_path / "model").mkdir()
    monkeypatch.setitem(
        SPEAKER_REFERENCE_AUDIO_ENCODERS, "qwen3-tts", FakeSpeakerReferenceAudioEncoder
    )
    FakeSpeakerReferenceAudioEncoder.loaded.clear()
    FakeSpeakerReferenceAudioEncoder.encoded.clear()
    return tmp_path


def init_arguments(root: Path, name: str = "study") -> list[str]:
    return [
        "experiment",
        "init",
        name,
        "--dataset",
        str(root / "encoded.jsonl"),
        "--audio-root",
        str(root / "raw"),
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


def test_init_writes_manifest_plan_and_speaker_embeddings(workspace: Path) -> None:
    assert main(init_arguments(workspace)) == 0

    directory = workspace / "experiments/study"
    manifest = ExperimentManifest.from_yaml(directory / "manifest.yaml")
    assert manifest == ExperimentManifest(
        dataset=workspace / "encoded.jsonl",
        audio_root=workspace / "raw",
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
    assert FakeSpeakerReferenceAudioEncoder.loaded == [
        (str(workspace / "model"), "cpu")
    ]

    embeddings = torch.load(directory / "speaker_embeddings.pt")
    assert sorted(embeddings) == ["0", "1"]
    for speaker, reference_id in plan.references.items():
        expected = torch.full((4,), float(int(reference_id.split("-")[1])))
        assert torch.equal(embeddings[speaker], expected)
    assert sorted(FakeSpeakerReferenceAudioEncoder.encoded) == sorted(
        f"{reference_id}.wav" for reference_id in plan.references.values()
    )


def test_init_refuses_an_existing_experiment(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / "experiments/study").mkdir(parents=True)

    assert main(init_arguments(workspace)) == 1
    assert "already exists" in capsys.readouterr().err
    assert FakeSpeakerReferenceAudioEncoder.loaded == []


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


def test_init_fails_cleanly_when_the_sampling_does_not_fit(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = init_arguments(workspace)
    arguments[arguments.index("--training-pool-size") + 1] = "11"

    assert main(arguments) == 1
    assert "training_pool_size 11 exceeds" in capsys.readouterr().err
    assert FakeSpeakerReferenceAudioEncoder.loaded == []
    assert not (workspace / "experiments/study").exists()


def test_init_fails_cleanly_when_the_model_does_not_load(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(cls, model_path: Path, device: str) -> None:
        raise OSError("no config.json")

    monkeypatch.setattr(
        FakeSpeakerReferenceAudioEncoder, "from_pretrained", classmethod(explode)
    )

    assert main(init_arguments(workspace)) == 1
    assert "cannot load qwen3-tts" in capsys.readouterr().err
    assert not (workspace / "experiments/study").exists()
