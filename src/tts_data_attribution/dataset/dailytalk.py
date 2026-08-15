from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from .example import AttributionDataset, DatasetExample


class DailyTalkDataset(AttributionDataset):
    @classmethod
    def from_directory(cls, root: str | Path) -> Self:
        root = Path(root)
        source = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        examples: list[DatasetExample] = []
        for dialogue_id in sorted(source, key=int):
            dialogue = source[dialogue_id]
            for utterance_id in sorted(dialogue, key=int):
                record = dialogue[utterance_id]
                speaker = str(record["speaker"])
                stem = f"{utterance_id}_{speaker}_d{dialogue_id}"
                audio_path = Path("data") / dialogue_id / f"{stem}.wav"
                text_path = root / "data" / dialogue_id / f"{stem}.txt"
                examples.append(
                    DatasetExample(
                        id=record["index"],
                        payload={
                            "audio_path": audio_path.as_posix(),
                            "text": text_path.read_text(encoding="utf-8"),
                        },
                        groups={
                            "dialogue": str(record["dialog_idx"]),
                            "speaker": speaker,
                        },
                        metadata={
                            "utterance_index": record["utterance_idx"],
                            "topic": record["topic"],
                            "emotion": record["emotion"],
                            "dialogue_act": record["act"],
                        },
                    )
                )
        return cls(examples)
