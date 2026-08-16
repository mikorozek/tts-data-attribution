from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from tts_data_attribution.dataset import Utterance, UtteranceDataset, encode_utterances


def fake_encoder(audio_paths: list[Path]) -> list[torch.Tensor]:
    return [torch.full((2, 16), 7, dtype=torch.long) for _ in audio_paths]


def source(identifier: str) -> Utterance:
    return Utterance(
        id=identifier, text="hi", speaker="0", dialogue="2", audio_path=f"data/2/{identifier}.wav"
    )


def test_encode_appends_only_missing_utterances_and_reports_progress(tmp_path: Path) -> None:
    output = tmp_path / "encoded.jsonl"
    already = source("2-0")
    UtteranceDataset([replace(already, audio_codes=[[1] * 16])]).to_jsonl(output)
    dataset = UtteranceDataset([already, source("2-1"), source("2-2")])
    messages: list[str] = []

    encode_utterances(dataset, tmp_path, fake_encoder, output, batch_size=1, report=messages.append)

    encoded = UtteranceDataset.from_jsonl(output)
    assert [utterance.id for utterance in encoded] == ["2-0", "2-1", "2-2"]
    assert encoded[0].audio_codes == [[1] * 16]
    assert encoded[1].audio_codes == [[7] * 16] * 2
    assert messages == ["encoded 1/2", "encoded 2/2"]


def test_encode_does_nothing_when_everything_is_encoded(tmp_path: Path) -> None:
    output = tmp_path / "encoded.jsonl"
    dataset = UtteranceDataset([replace(source("2-0"), audio_codes=[[1] * 16])])
    dataset.to_jsonl(output)
    before = output.read_bytes()
    messages: list[str] = []

    encode_utterances(dataset, tmp_path, fake_encoder, output, batch_size=4, report=messages.append)

    assert output.read_bytes() == before
    assert messages == ["all 1 utterances are already encoded"]
