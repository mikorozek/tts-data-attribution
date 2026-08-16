from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from torch.utils.data import Dataset

from tts_data_attribution.dataset import Utterance, UtteranceDataset


def utterance(identifier: str, codes: list[list[int]] | None = None) -> Utterance:
    return Utterance(
        id=identifier,
        text="héllo",
        speaker="0",
        dialogue="2",
        audio_path=f"data/2/{identifier}.wav",
        audio_codes=codes,
    )


def test_utterance_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        utterance("2-0").id = "changed"


def test_dataset_writes_one_sorted_json_line_per_utterance(tmp_path: Path) -> None:
    path = tmp_path / "utterances.jsonl"

    UtteranceDataset([utterance("2-0", [[1, 2]])]).to_jsonl(path)

    assert path.read_text(encoding="utf-8") == (
        '{"audio_codes":[[1,2]],"audio_path":"data/2/2-0.wav",'
        '"dialogue":"2","id":"2-0","speaker":"0","text":"héllo"}\n'
    )


def test_dataset_appends_to_an_existing_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "utterances.jsonl"
    UtteranceDataset([utterance("2-0")]).to_jsonl(path)

    UtteranceDataset([utterance("2-1")]).to_jsonl(path, append=True)

    assert [item.id for item in UtteranceDataset.from_jsonl(path)] == ["2-0", "2-1"]


def test_dataset_is_a_pytorch_dataset() -> None:
    utterances = [utterance("2-0"), utterance("2-1")]

    dataset = UtteranceDataset(utterances)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[1] == utterances[1]
    assert dataset.ids() == {"2-0", "2-1"}


def test_dataset_has_a_stable_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "utterances.jsonl"
    utterances = [utterance("2-0", [[1] * 16, [2] * 16]), utterance("2-1")]

    UtteranceDataset(utterances).to_jsonl(path)

    assert UtteranceDataset.from_jsonl(path).utterances == tuple(utterances)
    assert (
        json.loads(path.read_text(encoding="utf-8").splitlines()[1])["audio_codes"]
        is None
    )


def test_missing_required_field_raises_type_error(tmp_path: Path) -> None:
    path = tmp_path / "utterances.jsonl"
    path.write_text('{"id": "2-0"}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        UtteranceDataset.from_jsonl(path)


def test_malformed_json_raises_json_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "utterances.jsonl"
    path.write_text('{"id":\n', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        UtteranceDataset.from_jsonl(path)
