# TTS Data Attribution

Reusable PyTorch framework for data-attribution experiments validated through
counterfactual retraining. The implemented attribution methods are TrackStar and
projected cosine, evaluated with Linear Datamodeling Score.

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

Configure one immutable training run. The configuration is shared by the model
trained on the complete pool and every model trained on a named subset:

```bash
uv run tda training configure voice-study-1 \
  --lora-rank 8 --lora-alpha 16 \
  --learning-rate 2e-5 --epochs 3 --batch-size 1 --seed 5678
```

The command creates `training-runs/run-<UTC timestamp>/manifest.yaml` and prints
the generated run name. The strict schema is also shown in
[`training-run.example.yaml`](training-run.example.yaml).

Choose the data selection only when starting one model inside the run:

```bash
uv run tda training start voice-study-1 run-20260820T153012123456Z \
  --training-pool --device cuda:0

uv run tda training start voice-study-1 run-20260820T153012123456Z \
  --subset subset-0007 --device cuda:0
```

The training-pool model must be started before subset models because it resolves
the common ordered LoRA parameter layout in the run manifest. Every model writes
its own `metrics.jsonl` and one immutable checkpoint per trained epoch:

```text
training-runs/<run>/
├── manifest.yaml
├── training-pool/
│   ├── metrics.jsonl
│   └── checkpoints/epoch-NNNN/
└── subsets/<subset-id>/
    ├── metrics.jsonl
    └── checkpoints/epoch-NNNN/
```

Each checkpoint contains `adapter/`, `optimizer.pt`, and `metrics.json`. There
is no automatic checkpoint selection or duplicated final target.

Initialize a reusable two-sided projection inside the same training run:

```bash
uv run tda projection init voice-study-1 \
  run-20260820T153012123456Z two-sided-4096 \
  --training-checkpoint training-pool/checkpoints/epoch-0002 \
  --output-dimension 4096 --seed 1234
```

The explicit checkpoint path is relative to the training run. The command reads
the common parameter layout and stores the projection under
`training-runs/<run>/projections/<projection>/` without loading the model.

Apply the projection to the complete training pool and the query pool:

```bash
uv run tda projection apply voice-study-1 \
  run-20260820T153012123456Z two-sided-4096 \
  --training-pool --device cuda:0

uv run tda projection apply voice-study-1 \
  run-20260820T153012123456Z two-sided-4096 \
  --query-pool --device cuda:0
```

Both commands reload the explicitly selected adapter and matching AdamW state.
They validate the parameter order, AdamW recipe, optimizer step, and moments.
They write `projected/training-pool.pt` and `projected/query-pool.pt`.

Compute TrackStar and projected cosine from the same projected gradients:

```bash
uv run tda trackstar compute voice-study-1 \
  run-20260820T153012123456Z two-sided-4096 \
  --task-weight 0.9 --device cuda:0

uv run tda projected-cosine compute voice-study-1 \
  run-20260820T153012123456Z two-sided-4096 --device cuda:0
```

The immutable attribution tensors are stored separately:

```text
projections/<projection>/
├── trackstar/
│   ├── config.yaml
│   └── attributions.pt
└── projected-cosine/
    └── attributions.pt
```

Evaluate one attribution tensor by passing its path directly:

```bash
uv run tda lds compute \
  experiments/voice-study-1/training-runs/run-20260820T153012123456Z/projections/two-sided-4096/trackstar/attributions.pt --device cuda:0
```

LDS uses the subset checkpoints from the same run and the same epoch as the
projection's selected training-pool checkpoint. It computes Spearman correlation
across subsets for each query and saves their arithmetic mean as the scalar
`lds.pt` beside the selected `attributions.pt`. Missing subset checkpoints and
existing outputs are errors.
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
per-epoch adapter and AdamW checkpoint serialization, per-example gradient
collection, AdamW second-moment correction, Qwen LoRA block projection,
TrackStar Hessian correction, projected cosine, unit normalization, and attribution
scoring are also implemented, together with cohesive training runs, persisted
projected training/query pools, and same-run, same-epoch LDS evaluation over subset
models.
