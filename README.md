# TTS Data Attribution

Reusable PyTorch framework for data-attribution experiments validated through
counterfactual retraining. TrackStar and Linear Datamodeling Score are the first
planned attribution and evaluation methods.

The framework accepts user-provided datasets, `torch.nn.Module` models,
per-example objectives, parameter selectors, training procedures, subset masks,
and scalar evaluation responses. Reusable modules are added only with their
first implemented and tested behavior.

Read [`AGENTS.md`](AGENTS.md) for project-wide design rules.

## Current target

The first end-to-end target combines Qwen3-TTS-12Hz-1.7B-Base with DailyTalk.
Attribution will apply only to the controlled fine-tuning data, never to unknown
Qwen pretraining data. Exact source revisions, asset sizes, checksums, licenses,
and paper locations are recorded in [`references/sources.yaml`](references/sources.yaml).

The upstream Qwen fine-tuning loss is not treated as a trusted per-example
objective. Label alignment and within-frame target access must be tested before
any scaled run.

## Repository roles

- `src/tts_data_attribution/` contains importable implementation.
- `scripts/` contains commands intended for local and remote execution.
- `references/` contains immutable provenance, papers, checksums, and licenses.
- `third_party/` contains pinned vendored upstream source.
- `tests/` contains executable behavior checks.
- `data/` and `artifacts/` contain ignored runtime files.
- `configs/` will be created when the first implemented command consumes a configuration.

## DailyTalk download

The official DailyTalk distribution is a 5,341,371,062-byte Google Drive archive
that expands to 6,908,762,531 bytes. Allow at least 15 GB of free disk space
during extraction.

```bash
uv run --group data python scripts/download_dailytalk.py
```

Use `--data-root` and `--archive` to place the extracted dataset and temporary
archive on suitable remote storage. The archive is removed after successful
extraction unless `--keep-archive` is provided.

Create the framework JSONL index with:

```bash
uv run python scripts/index_dailytalk.py
```

`DailyTalkDataset` reads the per-utterance transcript files, preserves dialogue
and speaker groups, and carries the source topic, emotion, and dialogue act as
metadata. Both commands accept explicit paths for remote storage.

## Dataset API

`DatasetExample` represents one model-independent unit of attribution.
`AttributionDataset` stores examples, implements `torch.utils.data.Dataset`, and
reads or writes JSONL without a custom validation layer.

```python
from tts_data_attribution.dataset import AttributionDataset, DatasetExample

examples = AttributionDataset(
    [
        DatasetExample(
            id="conversation-1/turn-1",
            payload={"audio_path": "audio/turn-1.wav", "text": "Hello"},
            groups={"conversation": "conversation-1"},
        )
    ]
)
examples.to_jsonl("data/processed/examples.jsonl")
```

The interface is specified in
[`docs/specs/dataset-integration.md`](docs/specs/dataset-integration.md). Model
loading, gradients, attribution, and evaluation will be added only when their
behavior is implemented.
