# Dataset Integration MVP

Status: **implemented; intentionally small**

## Scope

A dataset integration converts its configured source into framework-shaped
examples. The framework persists those examples in one JSONL file:

```text
raw dataset → integration.prepare() → Iterable[DatasetExample] → examples.jsonl
```

This boundary exists to keep dataset-specific parsing out of framework core and
to let a new integration be supplied without modifying the writer or later
experiment components (the Open/Closed Principle). It is not a plugin system or
a dataset orchestration layer.

The implementation is `tts_data_attribution.core.dataset` and is re-exported
from `tts_data_attribution.core`.

## API

```python
JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@frozen
class DatasetExample:
    id: str
    payload: Mapping[str, JsonValue]
    groups: Mapping[str, str] | None = None
    metadata: Mapping[str, JsonValue] | None = None


class DatasetIntegration(Protocol):
    def prepare(self) -> Iterable[DatasetExample]: ...


def write_examples_jsonl(
    examples: Iterable[DatasetExample], path: str | Path
) -> None: ...


def read_examples_jsonl(path: str | Path) -> Iterator[DatasetExample]: ...


def prepare_dataset(
    integration: DatasetIntegration, output_path: str | Path
) -> None: ...
```

`prepare_dataset` is only the composition
`write_examples_jsonl(integration.prepare(), output_path)`.

`DatasetFormatError` is the single format exception raised for invalid example
shapes, malformed JSONL, non-finite numbers, and duplicate IDs.

## Example contract

- `id` is a non-empty string that the integration keeps stable across repeated
  preparation of the same logical example. Row numbers and absolute local paths
  are not stable IDs.
- `payload` is a JSON object whose dataset-specific keys and semantics belong
  to the integration. Model inputs, targets, transcripts, and asset references
  can be represented here without adding types to framework core.
- `groups`, when present, maps grouping-key names to string identities used by
  later splitting code to prevent leakage (for example, conversation or
  speaker IDs).
- `metadata`, when present, is a JSON object for descriptive values that are not
  the primary model input or target.
- Paths in `payload` are portable strings, normally paths relative to a separately
  configured dataset root. `Path` objects and absolute machine-local roots are
  not JSON values and must not be persisted.

For example:

```python
DatasetExample(
    id="dialogue-001/turn-003",
    payload={
        "audio_path": "audio/dialogue-001/turn-003.wav",
        "text": "Hello.",
    },
    groups={"conversation": "dialogue-001", "speaker": "speaker-a"},
    metadata={"duration_seconds": 0.8},
)
```

The integration owns source access, dataset-specific validation, filtering, and
preparation. It may be configured through its constructor or another
experiment-owned mechanism; the core API does not discover or register it.

## JSONL behavior

Each line is one JSON object with required `id` and `payload` fields. `groups` and
`metadata` are omitted when they are `None`. Object keys are sorted and output
uses UTF-8, so the same ordered examples produce stable bytes. Example order is
the integration's responsibility and is not changed by the writer.

Both the reader and writer reject duplicate IDs. The reader also reports the
file and line for malformed rows. The writer writes directly to the requested
file; there is no multi-file transaction or manifest commit protocol.

## Deliberately outside this MVP

- integration registries, entry points, import-path loading, and discovery;
- a production synthetic integration;
- DailyTalk or any other concrete dataset integration;
- source fingerprints, schema versions, hashes, and provenance records;
- issue/rejection/result/manifest-metadata object hierarchies;
- split orchestration, collation, tensor loading, and multi-dataset test runners;
- multi-file manifests and transactional lifecycle management.

Experiment runners may later persist the resolved IDs, groups, partitions,
hashes, configuration, and provenance required by `AGENTS.md`. Those concerns
do not need to expand this dataset-specific dependency-injection boundary.

## Tests

`tests/test_dataset.py` uses a test-local fake integration. The focused tests
cover protocol-based injection, frozen examples, stable JSON round-trips,
relative path strings, optional fields, malformed JSON, invalid JSON values,
and duplicate-ID rejection.
