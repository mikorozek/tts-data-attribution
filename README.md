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

The package installs the `tda` command. First define the dataset and base model
without loading the model:

```bash
uv run tda experiment init voice-study-1 \
  --dataset dailytalk \
  --data-root data/raw/dailytalk \
  --model qwen3-tts \
  --model-path artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254 \
  --training-pool-size 2000 --validation-pool-size 200 \
  --query-pool-size 200 --subset-count 50 --subset-size 1000 \
  --speaker-count 2 --seed 1234
```

This writes `manifest.yaml` and the deterministic `plan.json`. Encode the
training, validation, and query utterances in a separate, explicit step:

```bash
uv run tda experiment encode voice-study-1 \
  --device cuda:0 --batch-size 16
```

The Qwen model recorded in the manifest supplies the text processor, bundled
12 Hz speech tokenizer, and speaker encoder. The command writes
`sampled_utterances_encoded.jsonl` for the training, validation, and query pools and
`speaker_embeddings.pt` for the reference utterances. It appends complete
batches and skips already encoded IDs when resumed.

Configure one immutable, named training run for a selected training set:

```bash
uv run tda training configure voice-study-1 --training-pool \
  --lora-rank 8 --lora-alpha 16 \
  --learning-rate 2e-5 --epochs 3 --batch-size 1 --seed 5678
```

The command creates `training-runs/<set>-<UTC timestamp>/manifest.yaml` and
prints the generated run name. The strict schema is also shown in
[`training-run.example.yaml`](training-run.example.yaml). Configure a separate
run with `--subset subset-0007` to train on a named LDS subset.

Start the configured run by its generated name:

```bash
uv run tda training start voice-study-1 \
  training-pool-20260820T153012123456Z --device cuda:0
```

Epoch metrics are appended to `metrics.jsonl`. A successful run writes its one
final adapter and optimizer state atomically under `target/`.

Initialize a reusable two-sided projection from the ordered LoRA parameter
layout recorded by a completed training-pool target:

```bash
uv run tda projection init voice-study-1 two-sided-4096 \
  --training-run training-pool-20260820T153012123456Z \
  --output-dimension 4096 --seed 1234
```

The command stores an immutable manifest and the random left and right matrices
under `trackstar/projections/<projection-name>/`. It does not load the model or
compute gradients.

Apply the saved projection to every example in the training pool associated
with its training run:

```bash
uv run tda projection apply voice-study-1 two-sided-4096 \
  --training-pool --device cuda:0
```

The command reloads the final adapter and matching AdamW state, computes and
corrects each per-example gradient, and writes `projected/training-pool.pt`.
Apply the same model, optimizer state, and projection to the experiment query
pool with `--query-pool`; subset training targets are never projected.

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
are implemented. Generic LoRA injection, the core training and validation loop,
final adapter and AdamW checkpoint serialization, per-example gradient
collection, AdamW second-moment correction, Qwen LoRA block projection,
TrackStar Hessian correction, unit normalization, and attribution scoring are
also implemented. Projected-gradient artifacts, run orchestration, and
evaluation commands will be added only when their behavior is implemented.
