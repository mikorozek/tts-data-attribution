from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml

from tts_data_attribution.cli.experiment import load_dataset
from tts_data_attribution.cli.main import main
from tts_data_attribution.dataset import (
    DATASETS,
    DailyTalkDataset,
    Utterance,
    UtteranceDataset,
)
from tts_data_attribution.experiment import ExperimentManifest, Plan
from tts_data_attribution.models import EXPERIMENT_ENCODERS


@dataclass(frozen=True)
class SourceRecord:
    id: str
    text: str
    speaker: str
    dialogue: str
    audio_path: str


class SourceDataset:
    def __init__(self, records: list[SourceRecord]) -> None:
        self.records_by_id = {record.id: record for record in records}

    def get_records(self) -> list[SourceRecord]:
        return list(self.records_by_id.values())

    def get_records_by_ids(self, identifiers: list[str]) -> list[SourceRecord]:
        return [self.records_by_id[identifier] for identifier in identifiers]

    def __len__(self) -> int:
        return len(self.records_by_id)

    def __getitem__(self, identifier: str) -> SourceRecord:
        return self.records_by_id[identifier]


class FakeExperimentEncoder:
    loaded: list[tuple[str, str]] = []
    load_error: OSError | None = None
    texts: list[str] = []
    audio_batches: list[list[str]] = []
    speakers: list[str] = []

    @classmethod
    def from_pretrained(cls, model_path: Path, device: str) -> FakeExperimentEncoder:
        cls.loaded.append((str(model_path), device))
        if not model_path.is_dir():
            raise OSError(f"model directory not found at {model_path}")
        if cls.load_error is not None:
            raise cls.load_error
        return cls()

    def encode_text(self, text: str) -> list[int]:
        self.texts.append(text)
        return [len(text)]

    def encode_audio(self, audio_paths: list[Path]) -> list[list[list[int]]]:
        self.audio_batches.append([path.name for path in audio_paths])
        return [[[int(path.stem.split("-")[1])] * 16] for path in audio_paths]

    def encode_utterances(
        self, utterances: list[SourceRecord], data_root: Path
    ) -> list[Utterance]:
        audio_codes = self.encode_audio(
            [data_root / utterance.audio_path for utterance in utterances]
        )
        return [
            Utterance(
                id=utterance.id,
                speaker=utterance.speaker,
                dialogue=utterance.dialogue,
                text_ids=self.encode_text(utterance.text),
                audio_codes=codes,
            )
            for utterance, codes in zip(utterances, audio_codes, strict=True)
        ]

    def encode_speaker(self, audio_path: Path) -> torch.Tensor:
        self.speakers.append(audio_path.name)
        return torch.full((4,), float(int(audio_path.stem.split("-")[1])))


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model").mkdir()
    dataset = SourceDataset(
        [
            SourceRecord(
                id=f"{speaker}-{index}",
                text=f"text-{speaker}-{index}",
                speaker=speaker,
                dialogue=str(index),
                audio_path=f"data/{speaker}-{index}.wav",
            )
            for speaker in ("0", "1")
            for index in range(6)
        ]
    )
    monkeypatch.setitem(DATASETS, "dailytalk", lambda root: dataset)
    monkeypatch.setitem(EXPERIMENT_ENCODERS, "qwen3-tts", lambda: FakeExperimentEncoder)
    FakeExperimentEncoder.loaded.clear()
    FakeExperimentEncoder.load_error = None
    FakeExperimentEncoder.texts.clear()
    FakeExperimentEncoder.audio_batches.clear()
    FakeExperimentEncoder.speakers.clear()
    return tmp_path


def init_arguments(root: Path, name: str = "study") -> list[str]:
    return [
        "experiment",
        "init",
        name,
        "--dataset",
        "dailytalk",
        "--data-root",
        str(root / "raw"),
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
        "--root",
        str(root / "experiments"),
    ]


def encode_arguments(root: Path, name: str = "study") -> list[str]:
    return [
        "experiment",
        "encode",
        name,
        "--model",
        "qwen3-tts",
        "--model-path",
        str(root / "model"),
        "--device",
        "cpu",
        "--batch-size",
        "3",
        "--root",
        str(root / "experiments"),
    ]


def test_init_only_writes_the_manifest_and_plan(workspace: Path) -> None:
    assert main(init_arguments(workspace)) == 0

    directory = workspace / "experiments/study"
    manifest = ExperimentManifest.from_yaml(directory / "manifest.yaml")
    assert manifest == ExperimentManifest(
        dataset="dailytalk",
        data_root=workspace / "raw",
        training_pool_size=8,
        subset_count=3,
        subset_size=4,
        speaker_count=2,
        seed=7,
    )
    plan = Plan.from_json(directory / "plan.json")
    assert plan == Plan.sample(manifest, load_dataset(manifest).get_records())
    assert FakeExperimentEncoder.loaded == []
    assert sorted(path.name for path in directory.iterdir()) == [
        "manifest.yaml",
        "plan.json",
    ]


def test_encode_writes_only_the_sampled_utterances_and_speakers(
    workspace: Path,
) -> None:
    assert main(init_arguments(workspace)) == 0
    assert main(encode_arguments(workspace)) == 0

    directory = workspace / "experiments/study"
    plan = Plan.from_json(directory / "plan.json")
    selected_ids = set(plan.training_pool) | set(plan.references.values())
    encoded = UtteranceDataset.from_jsonl(
        directory / "sampled_utterances_encoded.jsonl"
    )

    assert encoded.ids() == selected_ids
    assert all(item.text_ids == [len(f"text-{item.id}")] for item in encoded)
    assert all(len(item.audio_codes[0]) == 16 for item in encoded)
    assert all(not hasattr(item, "text") for item in encoded)
    assert FakeExperimentEncoder.loaded == [(str(workspace / "model"), "cpu")]
    assert set(FakeExperimentEncoder.texts) == {
        f"text-{identifier}" for identifier in selected_ids
    }
    assert {
        name.removesuffix(".wav")
        for batch in FakeExperimentEncoder.audio_batches
        for name in batch
    } == selected_ids
    assert sorted(FakeExperimentEncoder.speakers) == sorted(
        f"{identifier}.wav" for identifier in plan.references.values()
    )
    assert sorted(
        torch.load(directory / "speaker_embeddings.pt", weights_only=True)
    ) == [
        "0",
        "1",
    ]
    assert yaml.safe_load((directory / "encoding.yaml").read_text()) == {
        "model": "qwen3-tts",
        "model_path": (workspace / "model").as_posix(),
    }


def test_encode_resumes_without_loading_the_model_again(workspace: Path) -> None:
    assert main(init_arguments(workspace)) == 0
    assert main(encode_arguments(workspace)) == 0
    directory = workspace / "experiments/study"
    output = directory / "sampled_utterances_encoded.jsonl"
    before = output.read_bytes()
    FakeExperimentEncoder.loaded.clear()

    assert main(encode_arguments(workspace)) == 0

    assert FakeExperimentEncoder.loaded == []
    assert output.read_bytes() == before


def test_encode_resumes_only_the_missing_utterance(workspace: Path) -> None:
    assert main(init_arguments(workspace)) == 0
    assert main(encode_arguments(workspace)) == 0
    directory = workspace / "experiments/study"
    output = directory / "sampled_utterances_encoded.jsonl"
    records = output.read_text(encoding="utf-8").splitlines()
    missing_id = UtteranceDataset.from_jsonl(output)[-1].id
    output.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")
    FakeExperimentEncoder.texts.clear()
    FakeExperimentEncoder.audio_batches.clear()

    assert main(encode_arguments(workspace)) == 0

    assert FakeExperimentEncoder.texts == [f"text-{missing_id}"]
    assert FakeExperimentEncoder.audio_batches == [[f"{missing_id}.wav"]]
    encoded = UtteranceDataset.from_jsonl(output)
    assert len(encoded.ids()) == len(encoded)


def test_model_load_failure_leaves_no_encoding_artifacts(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(init_arguments(workspace)) == 0
    FakeExperimentEncoder.load_error = OSError("broken model")
    directory = workspace / "experiments/study"

    assert main(encode_arguments(workspace)) == 1

    assert "cannot load qwen3-tts" in capsys.readouterr().err
    assert not (directory / "encoding.yaml").exists()
    assert not (directory / "sampled_utterances_encoded.jsonl").exists()
    assert not (directory / "speaker_embeddings.pt").exists()


def test_init_refuses_an_existing_experiment(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / "experiments/study").mkdir(parents=True)

    assert main(init_arguments(workspace)) == 1
    assert "already exists" in capsys.readouterr().err
    assert FakeExperimentEncoder.loaded == []


def test_init_fails_cleanly_when_the_dataset_is_missing(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(DATASETS, "dailytalk", DailyTalkDataset)
    (workspace / "raw/metadata.json").unlink()

    assert main(init_arguments(workspace)) == 1
    assert "metadata.json is missing" in capsys.readouterr().err


def test_init_fails_cleanly_when_the_sampling_does_not_fit(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = init_arguments(workspace)
    arguments[arguments.index("--training-pool-size") + 1] = "11"

    assert main(arguments) == 1
    assert "training_pool_size 11 exceeds" in capsys.readouterr().err
    assert FakeExperimentEncoder.loaded == []


def test_encode_fails_cleanly_when_the_model_is_missing(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(init_arguments(workspace)) == 0
    (workspace / "model").rmdir()

    assert main(encode_arguments(workspace)) == 1
    assert "model directory not found" in capsys.readouterr().err


def test_init_uses_dataset_specific_layout_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "custom"
    data_root.mkdir()
    dataset = SourceDataset(
        [
            SourceRecord(
                id=f"0-{index}",
                text="text",
                speaker="0",
                dialogue=str(index),
                audio_path=f"{index}.wav",
            )
            for index in range(3)
        ]
    )
    monkeypatch.setitem(DATASETS, "custom", lambda root: dataset)

    assert (
        main(
            [
                "experiment",
                "init",
                "custom-study",
                "--dataset",
                "custom",
                "--data-root",
                str(data_root),
                "--training-pool-size",
                "2",
                "--subset-count",
                "1",
                "--subset-size",
                "1",
                "--speaker-count",
                "1",
                "--seed",
                "1",
                "--root",
                str(tmp_path / "experiments"),
            ]
        )
        == 0
    )


def test_encode_rejects_a_different_recorded_model(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(init_arguments(workspace)) == 0
    directory = workspace / "experiments/study"
    (directory / "encoding.yaml").write_text(
        "model: qwen3-tts\nmodel_path: another-model\n", encoding="utf-8"
    )

    assert main(encode_arguments(workspace)) == 1

    assert "already encoded with" in capsys.readouterr().err
    assert FakeExperimentEncoder.loaded == []
