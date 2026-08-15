"""Small dataset-integration boundary and JSONL persistence helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class DatasetFormatError(ValueError):
    """Raised when an example or examples JSONL file has an invalid shape."""


@dataclass(frozen=True)
class DatasetExample:
    """Dataset-agnostic example prepared for an experiment.

    Dataset integrations define the contents of ``payload``. Paths within it
    must be portable strings (normally relative POSIX paths), not ``Path``
    objects. ``groups`` carries leakage-control identities such as speaker or
    document IDs; ``metadata`` is optional descriptive JSON data.
    """

    id: str
    payload: Mapping[str, JsonValue]
    groups: Mapping[str, str] | None = None
    metadata: Mapping[str, JsonValue] | None = None


@runtime_checkable
class DatasetIntegration(Protocol):
    """The only interface a dataset-specific integration must implement."""

    def prepare(self) -> Iterable[DatasetExample]:
        """Prepare examples from the integration's configured source."""


def write_examples_jsonl(examples: Iterable[DatasetExample], path: str | Path) -> None:
    """Write examples as stable JSON Lines, rejecting duplicate IDs."""

    path = Path(path)
    seen_ids: set[str] = set()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, example in enumerate(examples, start=1):
            row = _example_to_json(example, f"example {index}")
            if example.id in seen_ids:
                raise DatasetFormatError(
                    f"duplicate dataset example ID: {example.id!r}"
                )
            seen_ids.add(example.id)
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def read_examples_jsonl(path: str | Path) -> Iterator[DatasetExample]:
    """Yield examples from a JSON Lines file, rejecting malformed or duplicate rows."""

    path = Path(path)
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            location = f"{path}:{line_number}"
            if not line.strip():
                raise DatasetFormatError(f"{location}: blank JSONL line")
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, DatasetFormatError) as exc:
                raise DatasetFormatError(f"{location}: invalid JSON: {exc}") from exc
            example = _example_from_json(row, location)
            if example.id in seen_ids:
                raise DatasetFormatError(
                    f"{location}: duplicate dataset example ID: {example.id!r}"
                )
            seen_ids.add(example.id)
            yield example


def prepare_dataset(integration: DatasetIntegration, output_path: str | Path) -> None:
    """Prepare one integration and write its examples."""

    write_examples_jsonl(integration.prepare(), output_path)


def _example_to_json(example: object, location: str) -> dict[str, JsonValue]:
    if not isinstance(example, DatasetExample):
        raise TypeError(f"{location} is not a DatasetExample")
    if not isinstance(example.id, str) or not example.id:
        raise DatasetFormatError(f"{location}.id must be a non-empty string")

    row: dict[str, JsonValue] = {
        "id": example.id,
        "payload": _normalize_json_mapping(example.payload, f"{location}.payload"),
    }
    if example.groups is not None:
        if not isinstance(example.groups, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in example.groups.items()
        ):
            raise DatasetFormatError(f"{location}.groups must map strings to strings")
        row["groups"] = dict(example.groups)
    if example.metadata is not None:
        metadata = _normalize_json(example.metadata, f"{location}.metadata", set())
        if not isinstance(metadata, dict):  # Mapping input normalizes to a dict.
            raise DatasetFormatError(f"{location}.metadata must be a JSON object")
        row["metadata"] = metadata
    return row


def _example_from_json(value: object, location: str) -> DatasetExample:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise DatasetFormatError(f"{location}: example must be a JSON object")

    allowed = {"id", "payload", "groups", "metadata"}
    unknown = set(value).difference(allowed)
    if unknown:
        raise DatasetFormatError(f"{location}: unknown fields: {sorted(unknown)!r}")
    missing = {"id", "payload"}.difference(value)
    if missing:
        raise DatasetFormatError(f"{location}: missing fields: {sorted(missing)!r}")

    example_id = value["id"]
    if not isinstance(example_id, str) or not example_id:
        raise DatasetFormatError(f"{location}: id must be a non-empty string")

    groups_value = value.get("groups")
    groups: dict[str, str] | None = None
    if groups_value is not None:
        if not isinstance(groups_value, dict) or any(
            not isinstance(key, str) or not isinstance(group, str)
            for key, group in groups_value.items()
        ):
            raise DatasetFormatError(f"{location}: groups must map strings to strings")
        groups = dict(groups_value)

    metadata_value = value.get("metadata")
    metadata: dict[str, JsonValue] | None = None
    if metadata_value is not None:
        normalized_metadata = _normalize_json(
            metadata_value, f"{location}.metadata", set()
        )
        if not isinstance(normalized_metadata, dict):
            raise DatasetFormatError(f"{location}: metadata must be a JSON object")
        metadata = normalized_metadata

    return DatasetExample(
        id=example_id,
        payload=_normalize_json_mapping(value["payload"], f"{location}.payload"),
        groups=groups,
        metadata=metadata,
    )


def _normalize_json_mapping(value: object, location: str) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise DatasetFormatError(f"{location} must be a JSON object")
    normalized = _normalize_json(value, location, set())
    if not isinstance(normalized, dict):  # Mapping always normalizes to dict.
        raise DatasetFormatError(f"{location} must be a JSON object")
    return normalized


def _normalize_json(value: object, location: str, active: set[int]) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetFormatError(f"{location} contains a non-finite float")
        return value
    if isinstance(value, (list, Mapping)):
        identity = id(value)
        if identity in active:
            raise DatasetFormatError(f"{location} contains a cycle")
        active.add(identity)
        try:
            if isinstance(value, list):
                return [
                    _normalize_json(item, f"{location}[{index}]", active)
                    for index, item in enumerate(value)
                ]
            normalized: dict[str, JsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise DatasetFormatError(
                        f"{location} contains a non-string object key"
                    )
                normalized[key] = _normalize_json(item, f"{location}.{key}", active)
            return normalized
        finally:
            active.remove(identity)
    raise DatasetFormatError(
        f"{location} contains non-JSON value of type {type(value).__name__}"
    )


def _reject_json_constant(token: str) -> None:
    raise DatasetFormatError(f"non-standard JSON constant {token!r}")
