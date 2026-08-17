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
- `experiments/` contains ignored local experiment workspaces: manifest, plan, encoded examples, speaker embeddings, and run outputs.

## Providing the dataset

Obtaining the dataset is the user's responsibility. Place the extracted
DailyTalk distribution at `data/raw/dailytalk` and verify it against the
sizes and checksums recorded in [`references/sources.yaml`](references/sources.yaml).

## CLI

The package installs the `tda` command. First sample an experiment from a raw
dataset without loading a model:

```bash
uv run tda experiment init voice-study-1 \
  --dataset dailytalk \
  --data-root data/raw/dailytalk \
  --training-pool-size 2000 \
  --validation-pool-size 200 --query-pool-size 100 \
  --subset-count 50 --subset-size 1000 \
  --speaker-count 2 --seed 1234
```

This writes `manifest.yaml` and the deterministic `plan.json`. Encode exactly
the sampled utterances in a separate, explicit step:

```bash
uv run tda experiment encode voice-study-1 \
  --model qwen3-tts \
  --model-path artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254 \
  --device cuda:0 --batch-size 16
```

One Qwen model supplies the text processor, bundled 12 Hz speech tokenizer,
and speaker encoder. The command writes
`sampled_utterances_encoded.jsonl`, `speaker_embeddings.pt`, and
`encoding.yaml` inside the experiment directory. It appends complete batches
and skips already encoded IDs when resumed.

## Dataset API

`DailyTalkDataset` transiently exposes raw source records for sampling and
encoding; it is never serialized. `Utterance` is the persisted, model-ready
record containing `id`, grouping metadata, `text_ids`, and 16-codebook
`audio_codes`. `UtteranceDataset` implements `torch.utils.data.Dataset` and
stable JSONL serialization.

```python
from tts_data_attribution.dataset import UtteranceDataset

encoded = UtteranceDataset.from_jsonl(
    "experiments/voice-study-1/sampled_utterances_encoded.jsonl"
)
encoded[0].text_ids
encoded[0].audio_codes
```

The interface is specified in
[`docs/specs/dataset-integration.md`](docs/specs/dataset-integration.md). Qwen3-TTS
batching, differentiable talker inputs, and the per-example codebook objective
are implemented. Generic LoRA injection, adapter serialization, and the core
training and validation loop are also implemented. Checkpoint orchestration,
attribution, and evaluation commands will be added only when their behavior is
implemented.
