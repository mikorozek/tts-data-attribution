from __future__ import annotations

import json
from pathlib import Path

from .utterance import Utterance, UtteranceDataset


def load_dailytalk(root: str | Path) -> UtteranceDataset:
    root = Path(root)
    source = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    utterances: list[Utterance] = []
    for dialogue_id in sorted(source, key=int):
        for utterance_id in sorted(source[dialogue_id], key=int):
            record = source[dialogue_id][utterance_id]
            speaker = str(record["speaker"])
            stem = f"{utterance_id}_{speaker}_d{dialogue_id}"
            audio_path = Path("data") / dialogue_id / f"{stem}.wav"
            utterances.append(
                Utterance(
                    id=record["index"],
                    text=(root / audio_path.with_suffix(".txt")).read_text(encoding="utf-8"),
                    speaker=speaker,
                    dialogue=str(record["dialog_idx"]),
                    audio_path=audio_path.as_posix(),
                )
            )
    return UtteranceDataset(utterances)
