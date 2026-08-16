from __future__ import annotations

import json
from pathlib import Path


def write_dailytalk_fixture(root: Path) -> None:
    metadata = {
        "10": {"0": dailytalk_record("10-0", dialogue=10, utterance=0, speaker=1)},
        "2": {
            "1": dailytalk_record("2-1", dialogue=2, utterance=1, speaker=1),
            "0": dailytalk_record("2-0", dialogue=2, utterance=0, speaker=0),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_dailytalk_utterance(root, 10, 0, 1, "Transcript from the utterance file")
    write_dailytalk_utterance(root, 2, 1, 1, "Second")
    write_dailytalk_utterance(root, 2, 0, 0, "First")


def dailytalk_record(index: str, dialogue: int, utterance: int, speaker: int) -> dict:
    return {
        "index": index,
        "turn": 1,
        "topic": 1,
        "emotion": "no emotion",
        "act": "inform",
        "speaker": speaker,
        "text": "Metadata text",
        "dialog_idx": dialogue,
        "utterance_idx": utterance,
    }


def write_dailytalk_utterance(
    root: Path, dialogue_id: int, utterance_id: int, speaker: int, text: str
) -> None:
    directory = root / "data" / str(dialogue_id)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{utterance_id}_{speaker}_d{dialogue_id}"
    (directory / f"{stem}.txt").write_text(text, encoding="utf-8")
    (directory / f"{stem}.wav").write_bytes(b"")
