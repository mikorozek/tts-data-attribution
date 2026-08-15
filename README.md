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

## Dataset integration MVP

Framework core currently provides one small dependency-injection boundary:
`DatasetExample`, a `DatasetIntegration.prepare()` protocol, JSONL read/write
helpers, and the one-line `prepare_dataset` composition. There is no integration
registry, multi-file manifest lifecycle, or production synthetic dataset.
Concrete integrations, including DailyTalk, remain deferred until an experiment
needs them.

```python
from tts_data_attribution.core import DatasetExample, prepare_dataset


class MyDataset:
    def prepare(self):
        yield DatasetExample(
            id="conversation-1/turn-1",
            payload={"audio_path": "audio/turn-1.wav", "text": "Hello"},
            groups={"conversation": "conversation-1"},
        )


prepare_dataset(MyDataset(), "data/manifests/examples.jsonl")
```

Interface and format: [`docs/specs/dataset-integration.md`](docs/specs/dataset-integration.md).
Model loading, dataset-specific integrations, splits, gradients, and attribution
remain deferred.
