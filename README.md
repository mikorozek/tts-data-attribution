# TTS Data Attribution

Reusable PyTorch framework for running data-attribution experiments and validating them with counterfactual retraining. TrackStar and LDS are the first attribution and evaluation backends.

The framework core is designed to accept user-provided datasets, `torch.nn.Module` models, per-example objectives, parameter selectors, training recipes, subset masks, and scalar evaluation responses. Concrete model/dataset studies are isolated under `experiments/` and `src/tts_data_attribution/integrations/`.

Read [`AGENTS.md`](AGENTS.md) for framework-wide design rules.

## Current reference experiment

The initial research study is isolated under [`experiments/qwen3_tts_dailytalk/`](experiments/qwen3_tts_dailytalk/). It must not introduce Qwen- or DailyTalk-specific assumptions into framework-core modules.

## Local assets

Large assets are excluded from Git:

- model snapshots: `artifacts/models/`;
- raw/processed data and caches: `data/raw/`, `data/processed/`, `data/cache/`;
- checkpoints, gradients, and run outputs: `artifacts/runs/`.

Core papers are stored under `references/papers/`. Experiment-specific asset download and source information live inside each experiment directory.

## Dataset integration

`DatasetExample` represents one model-independent example. `AttributionDataset`
stores examples, implements `torch.utils.data.Dataset`, and reads or writes
JSONL without a custom validation layer or integration protocol.

```python
from tts_data_attribution.dataset import DatasetExample, AttributionDataset

examples = AttributionDataset(
    [
        DatasetExample(
            id="conversation-1/turn-1",
            payload={"audio_path": "audio/turn-1.wav", "text": "Hello"},
            groups={"conversation": "conversation-1"},
        )
    ]
)
examples.to_jsonl("data/manifests/examples.jsonl")
```

Concrete integrations live under `tts_data_attribution.integrations`. The
interface is specified in
[`docs/specs/dataset-integration.md`](docs/specs/dataset-integration.md).
DailyTalk integration, model loading, splits, gradients, and attribution remain
to be implemented.
