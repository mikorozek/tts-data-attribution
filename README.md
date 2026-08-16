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

- `src/tts_data_attribution/` contains importable implementation, including the `tda` CLI.
- `references/` contains immutable provenance, papers, checksums, and licenses.
- `third_party/` contains pinned vendored upstream source.
- `tests/` contains executable behavior checks.
- `data/` contains ignored source dataset assets and derived dataset products.
- `artifacts/` contains ignored downloaded upstream model assets.
- `experiments/` contains ignored local experiment workspaces: configuration, plan, materialized training data, and run outputs.

## Providing the dataset

Obtaining the dataset is the user's responsibility. Place the extracted
DailyTalk distribution at `data/raw/dailytalk` and verify it against the
sizes and checksums recorded in [`references/sources.yaml`](references/sources.yaml).

## CLI

The package installs the `tda` command, the single user surface of the
framework. Prepare a provided dataset with:

```bash
uv run --group qwen tda data encode dailytalk qwen3-tts \
  --data-root data/raw/dailytalk \
  --tokenizer-path artifacts/models/Qwen3-TTS-Tokenizer-12Hz-7dd38ad \
  --output data/processed/dailytalk_qwen3tts.jsonl
```

One invocation loads the named dataset and encodes every utterance with the
named model's tokenizer. It writes complete utterance records to `--output`
and a manifest with its content hash.
The command validates the dataset layout and resumes after interruption by
skipping already-encoded IDs.

`DailyTalkDataset` reads the per-utterance transcript files and keeps the
speaker and dialogue of every utterance; `Qwen3TTSEncoder` encodes any
`UtteranceDataset` with the pinned 12Hz tokenizer.

An experiment is one untracked directory under `experiments/`, defined by one
command: an encoded dataset, a model, and the sampling. `init` validates all
of it, then writes `manifest.yaml` (what was asked) and `plan.json` (the
sampled reference utterances, training pool, and subsets):

```bash
uv run --group qwen tda experiment init voice-study-1 \
  --dataset data/processed/dailytalk_qwen3tts.jsonl \
  --model qwen3-tts \
  --model-path artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254 \
  --training-pool-size 2000 --subset-count 50 --subset-size 1000 \
  --speaker-count 2 --seed 1234
``` The full command surface, including
the planned experiment commands, is specified in
[`docs/specs/cli.md`](docs/specs/cli.md).

## Dataset API

`Utterance` is one spoken sentence: `id`, `text`, `speaker`, `dialogue`,
`audio_path`, and, after encoding, `audio_codes`. `UtteranceDataset` stores
utterances, implements `torch.utils.data.Dataset`, and reads or writes JSONL.

```python
from tts_data_attribution.dataset import UtteranceDataset

encoded = UtteranceDataset.from_jsonl("data/processed/dailytalk_qwen3tts.jsonl")
encoded[0].audio_codes
```

The interface is specified in
[`docs/specs/dataset-integration.md`](docs/specs/dataset-integration.md). Model
loading, gradients, attribution, and evaluation will be added only when their
behavior is implemented.
