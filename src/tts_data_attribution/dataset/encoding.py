from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import torch

from .utterance import Utterance, UtteranceDataset

Encoder = Callable[[list[Path]], list[torch.Tensor]]


def encode_utterances(
    dataset: UtteranceDataset,
    audio_root: Path,
    encoder: Encoder,
    output: Path,
    batch_size: int,
    report: Callable[[str], None] = print,
) -> None:
    pending = missing_utterances(dataset, output)
    if not pending:
        report(f"all {len(dataset)} utterances are already encoded")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for batch in batches(pending, batch_size):
            codes = encoder([audio_root / utterance.audio_path for utterance in batch])
            for utterance, utterance_codes in zip(batch, codes, strict=True):
                stream.write(
                    replace(utterance, audio_codes=utterance_codes.tolist()).to_json() + "\n"
                )
            stream.flush()
            completed += len(batch)
            report(f"encoded {completed}/{len(pending)}")


def missing_utterances(dataset: UtteranceDataset, output: Path) -> list[Utterance]:
    encoded_ids = UtteranceDataset.from_jsonl(output).ids() if output.is_file() else set()
    return [utterance for utterance in dataset if utterance.id not in encoded_ids]


def batches(utterances: list[Utterance], batch_size: int) -> Iterator[list[Utterance]]:
    for start in range(0, len(utterances), batch_size):
        yield utterances[start : start + batch_size]
