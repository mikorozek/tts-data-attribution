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
├── training
│   ├── configure <experiment> (--training-pool | --subset ID)
│   │                        --lora-rank N --lora-alpha N
│   │                        --learning-rate R --epochs N
│   │                        --batch-size N --seed N [options]
│   └── start     <experiment> <training-run-name> [--device]
└── projection
    ├── init      <experiment> <projection-name>
    │             --training-run NAME --output-dimension N --seed N
    └── apply     <experiment> <projection-name> --training-pool [--device]
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

`configure` requires exactly one data-selection mode:

```text
--training-pool       configure a run on plan.training_pool
--subset ID           configure a run on the named subset
```

It creates an immutable run named `<set>-<UTC timestamp>` under
`experiments/<name>/training-runs/`. The generated `manifest.yaml` records the
selected set, dtype, LoRA configuration, AdamW configuration, epochs, batch
size, and seed. Resolved defaults are persisted alongside explicitly provided
values.

A project-level
[`training-run.example.yaml`](../../training-run.example.yaml) demonstrates the
same strict schema. Configure another named run to change the recipe or train a
different subset.

## `tda training start`

`start` receives the generated run name rather than another data-selection
flag:

```bash
tda training start EXPERIMENT TRAINING_RUN_NAME --device cuda:0
```

The selected training set and complete recipe come from the run manifest. A run
workspace has the following shape:

```text
experiments/<name>/training-runs/<run-name>/
├── manifest.yaml
├── metrics.jsonl
└── target/
    ├── adapter/
    ├── optimizer.pt
    └── metadata.json
```

Epoch metrics are appended to `metrics.jsonl` and written as JSON lines to
stdout. `target/` is created atomically only after successful completion and
contains the one final adapter and matching optimizer state. A run with metrics
or a target cannot be started again; configure a new run instead.

All experiment files and the selected device are validated before the model is
loaded. The command reads the base model from the experiment manifest and all
training choices from the immutable run manifest.

## `tda projection init`

`init` reads the exact ordered LoRA parameter layout from a completed
training-pool run target. It creates an independent two-sided random map for
every trainable parameter matrix and stores the immutable artifact at:

```text
experiments/<name>/trackstar/projections/<projection-name>/
├── manifest.yaml
└── matrices.pt
```

The manifest records the projection type, associated training run, output
dimension, seed, and ordered parameter names and shapes. `matrices.pt` stores
all left and right matrices on CPU in float32. The output dimension must be a
positive square. The command does not load model weights, optimizer state,
encoded data, or gradients and never replaces an existing projection.

## `tda projection apply`

`apply --training-pool` reloads the final adapter and matching AdamW state from
the training run named in the projection manifest. It validates the ordered
trainable parameter layout, evaluates the per-example objective in model eval
mode, corrects every gradient with the saved AdamW second moment, and applies
the saved two-sided matrices.

The result is stored at:

```text
trackstar/projections/<projection-name>/projected/training-pool.pt
```

It contains the ordered training-pool utterance IDs and a float32 matrix shaped
`[number of training examples, output dimension]`. Projection application is
not defined for subset training targets. Query-set application will be added
with named query-set encoding.

## Shared rules

- Errors go to stderr with exit code 1; argument misuse uses argparse errors.
- Experiment workspaces are local and never tracked.
- Dataset integrations own validation of their source layouts.
- Model-derived commands consume the model recorded in `manifest.yaml`.
- Training, attribution, and evaluation commands consume persisted plans and
  encoded data rather than resampling them implicitly.
