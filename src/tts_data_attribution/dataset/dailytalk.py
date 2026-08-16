from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset


@dataclass(frozen=True)
class DailyTalkRecord:
    id: str
    text: str
    speaker: str
    dialogue: str
    audio_path: str


class DailyTalkDataset(Dataset[DailyTalkRecord]):
    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        metadata_path = self.data_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"{metadata_path} is missing")
        source = json.loads(metadata_path.read_text(encoding="utf-8"))
        records_by_id: dict[str, DailyTalkRecord] = {}
        for dialogue_id in sorted(source, key=int):
            for utterance_id in sorted(source[dialogue_id], key=int):
                metadata = source[dialogue_id][utterance_id]
                speaker = str(metadata["speaker"])
                stem = f"{utterance_id}_{speaker}_d{dialogue_id}"
                audio_path = Path("data") / dialogue_id / f"{stem}.wav"
                text = (self.data_root / audio_path.with_suffix(".txt")).read_text(
                    encoding="utf-8"
                )
                record = DailyTalkRecord(
                    id=metadata["index"],
                    text=text,
                    speaker=speaker,
                    dialogue=str(metadata["dialog_idx"]),
                    audio_path=audio_path.as_posix(),
                )
                records_by_id[record.id] = record
        self.records_by_id = records_by_id

    def get_records(self) -> list[DailyTalkRecord]:
        return list(self.records_by_id.values())

    def get_records_by_ids(self, identifiers: list[str]) -> list[DailyTalkRecord]:
        return [self.records_by_id[identifier] for identifier in identifiers]

    def __len__(self) -> int:
        return len(self.records_by_id)

    def __getitem__(self, identifier: str) -> DailyTalkRecord:
        return self.records_by_id[identifier]
