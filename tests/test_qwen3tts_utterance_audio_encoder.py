from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeTokenizer

from tts_data_attribution.dataset import Utterance, UtteranceDataset
from tts_data_attribution.models.qwen3_tts import Qwen3TTSUtteranceAudioEncoder


def utterance(identifier: str) -> Utterance:
    return Utterance(
        id=identifier,
        text="hi",
        speaker="0",
        dialogue="2",
        audio_path=f"data/{identifier}.wav",
    )


def test_encode_writes_every_utterance_with_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer = FakeTokenizer()
    dataset = UtteranceDataset([utterance("2-0"), utterance("2-1"), utterance("2-2")])
    output = tmp_path / "encoded.jsonl"

    Qwen3TTSUtteranceAudioEncoder(tokenizer).encode(
        dataset, tmp_path / "raw", output, batch_size=2
    )

    encoded = UtteranceDataset.from_jsonl(output)
    assert [item.id for item in encoded] == ["2-0", "2-1", "2-2"]
    assert encoded[0].audio_codes == [[7] * 16] * 2
    assert encoded[0].audio_path == "data/2-0.wav"
    assert tokenizer.received == [
        [str(tmp_path / "raw/data/2-0.wav"), str(tmp_path / "raw/data/2-1.wav")],
        [str(tmp_path / "raw/data/2-2.wav")],
    ]
    assert capsys.readouterr().out == "encoded 2/3\nencoded 3/3\n"


def test_encode_resumes_by_skipping_encoded_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer = FakeTokenizer()
    dataset = UtteranceDataset([utterance("2-0"), utterance("2-1")])
    output = tmp_path / "encoded.jsonl"
    UtteranceDataset([dataset[0]]).to_jsonl(output)

    Qwen3TTSUtteranceAudioEncoder(tokenizer).encode(
        dataset, tmp_path, output, batch_size=4
    )

    assert [item.id for item in UtteranceDataset.from_jsonl(output)] == ["2-0", "2-1"]
    assert tokenizer.received == [[str(tmp_path / "data/2-1.wav")]]
    assert capsys.readouterr().out == "encoded 1/1\n"


def test_encode_does_nothing_when_everything_is_encoded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer = FakeTokenizer()
    dataset = UtteranceDataset([utterance("2-0")])
    output = tmp_path / "encoded.jsonl"
    dataset.to_jsonl(output)
    before = output.read_bytes()

    Qwen3TTSUtteranceAudioEncoder(tokenizer).encode(
        dataset, tmp_path, output, batch_size=4
    )

    assert output.read_bytes() == before
    assert tokenizer.received == []
    assert capsys.readouterr().out == "all 1 utterances are already encoded\n"
