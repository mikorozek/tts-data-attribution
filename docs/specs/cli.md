# CLI

Status: **experiment initialization, encoding, and training implemented; query commands planned**

## Command tree

```text
tda
├── experiment
│   ├── init    <name> --dataset <dataset> --data-root <dir>
│   │                  --model <model> --model-path <dir>
│   │                  --training-pool-size N --validation-pool-size N
│   │                  --subset-count N --subset-size N
│   │                  --speaker-count N --seed N [--root]
│   └── encode  <name> [--device] [--batch-size] [--root]
└── training
    ├── configure <experiment> --lora-rank N --lora-alpha N
    │                        --learning-rate R --epochs N
    │                        --batch-size N --seed N [options]
    └── start     <experiment>
                  (--training-pool | --subset ID | --all-training-sets)
                  [--device]
```

## `tda experiment init`

`init` records a raw dataset and base model, samples the training experiment,
and writes no model-derived data. It does not load the model or require a GPU.

Validation completes before the experiment directory is created:

- the named dataset must load from `--data-root`;
- `speaker_count` must fit the available speakers;
- the requested pools must fit the eligible dialogue-disjoint utterances;
- `subset_size` must not exceed `training_pool_size`;
- the experiment directory must not already exist.

`manifest.yaml` records the dataset, base model, and sampling request:

```yaml
data_root: data/raw/dailytalk
dataset: dailytalk
model: qwen3-tts
model_path: artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254
seed: 1234
speaker_count: 2
subset_count: 50
subset_size: 1000
training_pool_size: 2000
validation_pool_size: 200
```

`plan.json` records reference utterances, training and validation pools, and
training subsets. These selections are disjoint at dialogue level. The same
manifest produces the same byte-stable plan. Query sets are sampled later and
are not part of the initial plan.

## `tda experiment encode`

`encode` reads the base model from the immutable experiment manifest. For
Qwen3-TTS, one loaded `Qwen3TTSModel` provides:

- the processor for `<|im_start|>assistant\n{text}` text IDs;
- the bundled speech tokenizer for 16-codebook audio codes;
- the speaker encoder for one vector per planned speaker reference.

The command writes:

```text
experiments/<name>/
├── manifest.yaml
├── plan.json
├── sampled_utterances_encoded.jsonl
└── speaker_embeddings.pt
```

`sampled_utterances_encoded.jsonl` contains only training and validation
utterances. Reference audio is converted only into entries in
`speaker_embeddings.pt`; reference utterances are not materialized as text and
codec-token records.

Encoding appends only complete batches. A repeated invocation skips encoded IDs
and already stored speaker embeddings.

## `tda training configure`

`configure` writes one immutable training recipe shared by the training-pool
checkpoint and every subset checkpoint. It requires an initialized experiment
and refuses to overwrite an existing `training.yaml`. The command records all
resolved defaults as well as explicitly provided values.

A project-level [`training.example.yaml`](../../training.example.yaml) uses the
same strict schema. It may be copied manually to
`experiments/<name>/training.yaml`; future training commands validate manually
created files before loading a model.

The initial implementation fixes AdamW, Qwen LoRA target modules, the training
objective, and final-checkpoint-only serialization. These are not exposed as
configuration switches until an experiment requires alternatives.

## `tda training start`

`start` requires exactly one data-selection mode:

```text
--training-pool       train one checkpoint on plan.training_pool
--subset ID           train one checkpoint on the named subset
--all-training-sets   train the training pool and every named subset
```

Subset names are stable keys such as `subset-0007` in `plan.json`. Checkpoints
are written to:

```text
experiments/<name>/checkpoints/
├── training-pool/
└── subsets/<subset-id>/
```

Single-run modes refuse to replace an existing checkpoint. The all-training-sets
mode skips complete checkpoints so an interrupted collection can be continued,
but rejects incomplete checkpoint directories. Each selected training set gets
a newly loaded base model, identically seeded LoRA initialization, fresh AdamW
state, and the shared validation pool. Epoch metrics and checkpoint events are
written as JSON lines to stdout.

All experiment files and the selected device are validated before the model is
loaded. Training reads the model path from `manifest.yaml` and all recipe values
from `training.yaml`; the command exposes no recipe overrides.

## Shared rules

- Errors go to stderr with exit code 1; argument misuse uses argparse errors.
- Experiment workspaces are local and never tracked.
- Dataset integrations own validation of their source layouts.
- Model-derived commands consume the model recorded in `manifest.yaml`.
- Training, attribution, and evaluation commands consume persisted plans and
  encoded data rather than resampling them implicitly.
