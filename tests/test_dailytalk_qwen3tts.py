from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeTokenizer, write_dailytalk_fixture

from tts_data_attribution.dataset import DailyTalkQwen3TTSDataset, Utterance, UtteranceDataset


def test_loads_utterances_from_the_official_layout(tmp_path: Path) -> None:
    write_dailytalk_fixture(tmp_path)

    dataset = DailyTalkQwen3TTSDataset(tmp_path)

    assert isinstance(dataset, UtteranceDataset)
    assert [utterance.id for utterance in dataset] == ["2-0", "2-1", "10-0"]
    assert dataset[0] == Utterance(
        id="2-0", text="First", speaker="0", dialogue="2", audio_path="data/2/0_0_d2.wav"
    )
    assert dataset[2].text == "Transcript from the utterance file"
    assert dataset[2].audio_codes is None


def test_encode_writes_every_utterance_with_codes(
    tmp_path: Path, fake_tokenizer: FakeTokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    write_dailytalk_fixture(tmp_path / "raw")
    output = tmp_path / "encoded.jsonl"

    DailyTalkQwen3TTSDataset(tmp_path / "raw").encode(
        tokenizer_path=tmp_path / "tokenizer", output=output, device="cpu", batch_size=2
    )

    encoded = UtteranceDataset.from_jsonl(output)
    assert [utterance.id for utterance in encoded] == ["2-0", "2-1", "10-0"]
    assert encoded[0].audio_codes == [[7] * 16] * 2
    assert encoded[0].audio_path == "data/2/0_0_d2.wav"
    assert fake_tokenizer.received == [
        [str(tmp_path / "raw/data/2/0_0_d2.wav"), str(tmp_path / "raw/data/2/1_1_d2.wav")],
        [str(tmp_path / "raw/data/10/0_1_d10.wav")],
    ]
    assert capsys.readouterr().out == "encoded 2/3\nencoded 3/3\n"


def test_encode_resumes_by_skipping_encoded_ids(
    tmp_path: Path, fake_tokenizer: FakeTokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    write_dailytalk_fixture(tmp_path / "raw")
    output = tmp_path / "encoded.jsonl"
    dataset = DailyTalkQwen3TTSDataset(tmp_path / "raw")
    UtteranceDataset([dataset[0], dataset[1]]).to_jsonl(output)

    dataset.encode(tokenizer_path=tmp_path / "tokenizer", output=output, device="cpu", batch_size=4)

    assert [utterance.id for utterance in UtteranceDataset.from_jsonl(output)] == [
        "2-0",
        "2-1",
        "10-0",
    ]
    assert fake_tokenizer.received == [[str(tmp_path / "raw/data/10/0_1_d10.wav")]]
    assert capsys.readouterr().out == "encoded 1/1\n"


def test_encode_does_nothing_when_everything_is_encoded(
    tmp_path: Path, fake_tokenizer: FakeTokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    write_dailytalk_fixture(tmp_path / "raw")
    output = tmp_path / "encoded.jsonl"
    dataset = DailyTalkQwen3TTSDataset(tmp_path / "raw")
    dataset.to_jsonl(output)
    before = output.read_bytes()

    dataset.encode(tokenizer_path=tmp_path / "tokenizer", output=output, device="cpu", batch_size=4)

    assert output.read_bytes() == before
    assert fake_tokenizer.received == []
    assert capsys.readouterr().out == "all 3 utterances are already encoded\n"
