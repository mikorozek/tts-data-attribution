from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from torch.utils.data import Dataset

from tts_data_attribution.dataset import Utterance, UtteranceDataset


def utterance(identifier: str) -> Utterance:
    return Utterance(
        id=identifier,
        speaker="0",
        dialogue="2",
        text_ids=[1, 2, 3],
        audio_codes=[[1] * 16, [2] * 16],
    )


def test_utterance_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(utterance("2-0"), "id", "changed")


def test_dataset_has_a_stable_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "utterances.jsonl"
    utterances = [utterance("2-0"), utterance("2-1")]

    UtteranceDataset(utterances).to_jsonl(path)

    assert UtteranceDataset.from_jsonl(path).utterances == tuple(utterances)
    assert set(json.loads(path.read_text().splitlines()[0])) == {
        "audio_codes",
        "dialogue",
        "id",
        "speaker",
        "text_ids",
    }


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


def test_utterance_dataset_gets_utterances_in_requested_id_order() -> None:
    first = Utterance(
        id="first",
        speaker="speaker",
        dialogue="dialogue-1",
        text_ids=[1],
        audio_codes=[[2] * 16],
    )
    second = Utterance(
        id="second",
        speaker="speaker",
        dialogue="dialogue-2",
        text_ids=[3],
        audio_codes=[[4] * 16],
    )
    dataset = UtteranceDataset([first, second])

    assert dataset.get_utterances_by_ids(["second", "first"]) == [second, first]


def test_utterance_dataset_rejects_duplicate_ids() -> None:
    utterance = Utterance(
        id="duplicate",
        speaker="speaker",
        dialogue="dialogue",
        text_ids=[1],
        audio_codes=[[2] * 16],
    )

    with pytest.raises(ValueError, match="utterance IDs must be unique"):
        UtteranceDataset([utterance, utterance])
