from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tts_data_attribution.core import (
    DatasetExample,
    DatasetFormatError,
    DatasetIntegration,
    prepare_dataset,
    read_examples_jsonl,
    write_examples_jsonl,
)


class FakeDatasetIntegration:
    """Tiny test double demonstrating dependency injection through the protocol."""

    def __init__(self, examples: Iterable[DatasetExample]) -> None:
        self.examples = list(examples)

    def prepare(self) -> Iterable[DatasetExample]:
        return iter(self.examples)


def test_dataset_example_is_frozen_and_integration_is_structural() -> None:
    example = DatasetExample(id="example-1", payload={"text": "hello"})
    integration = FakeDatasetIntegration([example])

    assert isinstance(integration, DatasetIntegration)
    with pytest.raises(FrozenInstanceError):
        example.id = "changed"  # type: ignore[misc]


def test_examples_have_a_stable_json_round_trip(tmp_path: Path) -> None:
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
        DatasetExample(id="dialogue-2/turn-1", payload={"items": ["generic", None]}),
    ]

    write_examples_jsonl(examples, path)

    assert list(read_examples_jsonl(path)) == examples
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["payload"]["audio_path"] == "audio/dialogue-1/turn-2.wav"
    assert isinstance(rows[0]["payload"]["audio_path"], str)
    assert rows[1] == {
        "id": "dialogue-2/turn-1",
        "payload": {"items": ["generic", None]},
    }


def test_jsonl_output_is_stable_for_mapping_order(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"

    write_examples_jsonl(
        [DatasetExample("same", {"z": 1, "a": {"y": 2, "b": 3}})], left
    )
    write_examples_jsonl(
        [DatasetExample("same", {"a": {"b": 3, "y": 2}, "z": 1})], right
    )

    assert left.read_bytes() == right.read_bytes()


def test_prepare_dataset_is_protocol_composition(tmp_path: Path) -> None:
    path = tmp_path / "prepared.jsonl"
    expected = [DatasetExample("one", {"target": "hello"})]

    prepare_dataset(FakeDatasetIntegration(expected), path)

    assert list(read_examples_jsonl(path)) == expected


def test_writer_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"

    with pytest.raises(DatasetFormatError, match="duplicate dataset example ID"):
        write_examples_jsonl(
            [
                DatasetExample("same", {"value": 1}),
                DatasetExample("same", {"value": 2}),
            ],
            path,
        )


def test_reader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text(
        '{"id":"same","payload":{"value":1}}\n{"id":"same","payload":{"value":2}}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetFormatError, match="duplicate dataset example ID"):
        list(read_examples_jsonl(path))


@pytest.mark.parametrize(
    ("example", "message"),
    [
        (DatasetExample("", {}), "id must be a non-empty string"),
        (DatasetExample("bad-path", {"path": Path("audio.wav")}), "non-JSON value"),
        (DatasetExample("bad-number", {"value": float("nan")}), "non-finite"),
        (
            DatasetExample("bad-payload", []),  # type: ignore[arg-type]
            "payload must be a JSON object",
        ),
        (DatasetExample("bad-groups", {}, groups={"speaker": 1}), "groups"),
    ],
)
def test_writer_rejects_invalid_example_shapes(
    tmp_path: Path, example: DatasetExample, message: str
) -> None:
    with pytest.raises(DatasetFormatError, match=message):
        write_examples_jsonl([example], tmp_path / "examples.jsonl")


def test_reader_reports_line_for_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text('{"id":"one","payload":{}}\n{"id":"two"}\n', encoding="utf-8")

    with pytest.raises(DatasetFormatError, match=r"examples.jsonl:2: missing fields"):
        list(read_examples_jsonl(path))
