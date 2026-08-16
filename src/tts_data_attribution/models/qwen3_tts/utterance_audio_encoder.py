from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Self

from qwen_tts import Qwen3TTSTokenizer
from torch.utils.data import DataLoader

from ...dataset import Utterance, UtteranceDataset
from ..utterance_audio_encoder import UtteranceAudioEncoder


class Qwen3TTSUtteranceAudioEncoder(UtteranceAudioEncoder):
    def __init__(self, tokenizer: Qwen3TTSTokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, tokenizer_path: str | Path, device: str) -> Self:
        return cls(
            Qwen3TTSTokenizer.from_pretrained(str(tokenizer_path), device_map=device)
        )

    def encode(
        self, dataset: UtteranceDataset, audio_root: Path, output: Path, batch_size: int
    ) -> None:
        encoded_ids = (
            UtteranceDataset.from_jsonl(output).ids() if output.is_file() else set()
        )
        pending = UtteranceDataset(u for u in dataset if u.id not in encoded_ids)
        if not len(pending):
            print(f"all {len(dataset)} utterances are already encoded")
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = 0
        for batch in DataLoader(pending, batch_size=batch_size, collate_fn=list):
            UtteranceDataset(self.encode_batch(batch, audio_root)).to_jsonl(
                output, append=True
            )
            completed += len(batch)
            print(f"encoded {completed}/{len(pending)}", flush=True)

    def encode_batch(self, batch: list[Utterance], audio_root: Path) -> list[Utterance]:
        encoded = self.tokenizer.encode([str(audio_root / u.audio_path) for u in batch])
        return [
            replace(utterance, audio_codes=codes.cpu().tolist())
            for utterance, codes in zip(batch, encoded.audio_codes, strict=True)
        ]
