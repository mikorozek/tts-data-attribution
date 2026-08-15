from __future__ import annotations

import json
from pathlib import Path

from tts_data_attribution.dataset import AttributionDataset, DailyTalkDataset


def write_utterance(
    root: Path,
    dialogue_id: int,
    utterance_id: int,
    speaker: int,
    text: str,
) -> None:
    directory = root / "data" / str(dialogue_id)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{utterance_id}_{speaker}_d{dialogue_id}"
    (directory / f"{stem}.txt").write_text(text, encoding="utf-8")
    (directory / f"{stem}.wav").write_bytes(b"")


def test_dailytalk_dataset_builds_examples_from_the_official_layout(
    tmp_path: Path,
) -> None:
    metadata = {
        "10": {
            "0": {
                "index": "10-0",
                "turn": 1,
                "topic": 2,
                "emotion": "happiness",
                "act": "inform",
                "speaker": 1,
                "text": "Metadata text",
                "dialog_idx": 10,
                "utterance_idx": 0,
            }
        },
        "2": {
            "1": {
                "index": "2-1",
                "turn": 2,
                "topic": 1,
                "emotion": "no emotion",
                "act": "question",
                "speaker": 1,
                "text": "Second",
                "dialog_idx": 2,
                "utterance_idx": 1,
            },
            "0": {
                "index": "2-0",
                "turn": 2,
                "topic": 1,
                "emotion": "no emotion",
                "act": "inform",
                "speaker": 0,
                "text": "First",
                "dialog_idx": 2,
                "utterance_idx": 0,
            },
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_utterance(tmp_path, 10, 0, 1, "Transcript from the utterance file")
    write_utterance(tmp_path, 2, 1, 1, "Second")
    write_utterance(tmp_path, 2, 0, 0, "First")

    dataset = DailyTalkDataset.from_directory(tmp_path)

    assert isinstance(dataset, AttributionDataset)
    assert [example.id for example in dataset.examples] == ["2-0", "2-1", "10-0"]
    assert dataset[0].payload == {
        "audio_path": "data/2/0_0_d2.wav",
        "text": "First",
    }
    assert dataset[0].groups == {"dialogue": "2", "speaker": "0"}
    assert dataset[0].metadata == {
        "utterance_index": 0,
        "topic": 1,
        "emotion": "no emotion",
        "dialogue_act": "inform",
    }
    assert dataset[2].payload["text"] == "Transcript from the utterance file"
