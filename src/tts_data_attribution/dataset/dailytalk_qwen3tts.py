from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from torch.utils.data import DataLoader

from ..models.qwen3_tts import CodesEncoder
from .utterance import Utterance, UtteranceDataset


class DailyTalkQwen3TTSDataset(UtteranceDataset):
    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        super().__init__(self.load())

    def load(self) -> list[Utterance]:
        source = json.loads((self.data_root / "metadata.json").read_text(encoding="utf-8"))
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
                        text=(self.data_root / audio_path.with_suffix(".txt")).read_text(
                            encoding="utf-8"
                        ),
                        speaker=speaker,
                        dialogue=str(record["dialog_idx"]),
                        audio_path=audio_path.as_posix(),
                    )
                )
        return utterances

    def encode(self, tokenizer_path: Path, output: Path, device: str, batch_size: int) -> None:
        encoded_ids = UtteranceDataset.from_jsonl(output).ids() if output.is_file() else set()
        pending = UtteranceDataset(u for u in self.utterances if u.id not in encoded_ids)
        if not len(pending):
            print(f"all {len(self)} utterances are already encoded")
            return
        encoder = CodesEncoder.from_pretrained(tokenizer_path, device)
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = 0
        with output.open("a", encoding="utf-8", newline="\n") as stream:
            for batch in DataLoader(pending, batch_size=batch_size, collate_fn=list):
                codes = encoder.encode([self.data_root / u.audio_path for u in batch])
                for utterance, utterance_codes in zip(batch, codes, strict=True):
                    encoded = replace(utterance, audio_codes=utterance_codes.tolist())
                    stream.write(encoded.to_json() + "\n")
                stream.flush()
                completed += len(batch)
                print(f"encoded {completed}/{len(pending)}", flush=True)
