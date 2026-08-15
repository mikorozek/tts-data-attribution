from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from torch.utils.data import Dataset

from tts_data_attribution.dataset import AttributionDataset, DatasetExample


def test_dataset_example_is_created_from_a_mapping() -> None:
    value = {
        "id": "example-1",
        "payload": {"text": "hello"},
        "groups": {"speaker": "speaker-a"},
        "metadata": {"duration_seconds": 1.25},
    }

    example = DatasetExample(**value)

    assert example.id == "example-1"
    assert example.payload == {"text": "hello"}
    assert example.groups == {"speaker": "speaker-a"}
    assert example.metadata == {"duration_seconds": 1.25}


def test_dataset_example_is_frozen() -> None:
    example = DatasetExample(id="example-1", payload={"text": "hello"})

    with pytest.raises(FrozenInstanceError):
        example.id = "changed"


def test_attribution_dataset_is_a_pytorch_dataset() -> None:
    examples = [
        DatasetExample(id="example-1", payload={"text": "hello"}),
        DatasetExample(id="example-2", payload={"text": "world"}),
    ]

    dataset = AttributionDataset(examples)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0] == examples[0]
    assert dataset[1] == examples[1]


def test_attribution_dataset_has_a_stable_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    examples = [
        DatasetExample(
            id="dialogue-1/turn-2",
            payload={
                "audio_path": "audio/dialogue-1/turn-2.wav",
                "text": "héllo",
                "tokens": [1, 2, 3],
            },
            groups={"conversation": "dialogue-1", "speaker": "speaker-a"},
            metadata={"duration_seconds": 1.25, "reviewed": True},
        ),
        DatasetExample(
            id="dialogue-2/turn-1",
            payload={"items": ["generic", None]},
        ),
    ]

    AttributionDataset(examples).to_jsonl(path)
    restored = AttributionDataset.from_jsonl(path)

    assert restored.examples == tuple(examples)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1] == {
        "groups": None,
        "id": "dialogue-2/turn-1",
        "metadata": None,
        "payload": {"items": ["generic", None]},
    }


def test_jsonl_output_is_stable_for_mapping_order(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"

    AttributionDataset(
        [DatasetExample("same", {"z": 1, "a": {"y": 2, "b": 3}})]
    ).to_jsonl(left)
    AttributionDataset(
        [DatasetExample("same", {"a": {"b": 3, "y": 2}, "z": 1})]
    ).to_jsonl(right)

    assert left.read_bytes() == right.read_bytes()


def test_missing_required_field_raises_type_error(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text('{"payload": {}}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        AttributionDataset.from_jsonl(path)


def test_malformed_json_raises_json_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text('{"id":\n', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        AttributionDataset.from_jsonl(path)
