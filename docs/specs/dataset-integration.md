# Dataset Integration

Status: **implemented; intentionally small**

## Scope

A concrete dataset integration converts its source into model-independent
`DatasetExample` values. `AttributionDataset` stores those values, implements the
PyTorch dataset interface, and optionally persists them as JSONL.

```text
raw dataset → concrete integration → AttributionDataset → PyTorch DataLoader
                                      ↕
                                examples.jsonl
```

The generic dataset types live in `tts_data_attribution.dataset`. Concrete
source parsing will be added under `tts_data_attribution.integrations` with the
first implemented integration.

## API

```python
@dataclass(frozen=True)
class DatasetExample:
    id: str
    payload: Mapping[str, JsonValue]
    groups: Mapping[str, str] | None = None
    metadata: Mapping[str, JsonValue] | None = None


class AttributionDataset(torch.utils.data.Dataset[DatasetExample]):
    def __init__(self, examples: Iterable[DatasetExample]) -> None: ...

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Self: ...

    def to_jsonl(self, path: str | Path) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> DatasetExample: ...
```

`AttributionDataset` is already a PyTorch dataset. It can be passed directly to a
`DataLoader` without an adapter. Model-specific loading, transforms, collation,
and tensor construction remain separate concerns.

## Example contract

- `id` identifies the same logical example across repeated indexing runs.
- `payload` contains model-independent inputs, targets, and asset references.
- `groups` contains identities used later to prevent split leakage.
- `metadata` contains optional descriptive values.
- Paths are portable strings, normally relative to a separately configured
  dataset root.

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

## JSONL behavior

`AttributionDataset.from_jsonl()` decodes every line with `json.loads()` and passes
the resulting mapping directly to the generated dataclass constructor:

```python
DatasetExample(**json.loads(line))
```

`AttributionDataset.to_jsonl()` serializes each example with `dataclasses.asdict()`
and `json.dump()`. Object keys are sorted and UTF-8 is preserved, so the same
ordered examples produce stable bytes.

There is no custom schema validator or format exception. Standard exceptions
such as `JSONDecodeError`, `TypeError`, and `OSError` propagate unchanged. Python
type annotations are not runtime validators.

## Integration boundary

The framework does not define a registry or a custom integration protocol. A
concrete integration can subclass `AttributionDataset` or construct one from its
source. Adding an integration does not require changes to the generic dataset
module.

The first concrete implementation will create
`tts_data_attribution.integrations.dailytalk.dataset`.

## Deliberately outside this interface

- integration registries and plugin discovery;
- dataset downloading;
- model-specific waveform loading, resampling, tokenization, and collation;
- split orchestration;
- source fingerprints and schema migrations;
- multi-dataset orchestration.
