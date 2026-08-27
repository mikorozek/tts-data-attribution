# CLI

Status: **end-to-end experiment, training, attribution, and LDS pipeline implemented**

## Command tree

```text
tda
├── experiment
│   ├── init    <name> [dataset, model, pool, subset, and seed options]
│   └── encode  <name> [--device] [--batch-size]
├── training
│   ├── configure <experiment> [training recipe]
│   └── start     <experiment> <run> (--training-pool | --subset ID) [--device]
├── projection
│   ├── init      <experiment> <run> <projection>
│   │             --training-checkpoint training-pool/checkpoints/epoch-NNNN
│   │             --output-dimension N --seed N
│   └── apply     <experiment> <run> <projection>
│                 (--training-pool | --query-pool) [--device]
├── attribution
│   └── compute   <experiment> <run> <projection> --task-weight W [--device]
├── projected-cosine
│   └── compute   <experiment> <run> <projection> [--device]
└── lds
    └── compute   <attributions-path> [--device]
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
query_pool_size: 200
seed: 1234
speaker_count: 2
subset_count: 50
subset_size: 1000
training_pool_size: 2000
validation_pool_size: 200
```

`plan.json` records reference utterances, training, validation, and query
pools, and training subsets. These selections are disjoint at dialogue level.
The same manifest produces the same byte-stable plan.

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

`sampled_utterances_encoded.jsonl` contains the training, validation, and query
pools. Reference audio is converted only into entries in
`speaker_embeddings.pt`; reference utterances are not materialized as text and
codec-token records.

Encoding appends only complete batches. A repeated invocation skips encoded IDs
and already stored speaker embeddings.

## `tda training configure`

`configure` creates one training run for one complete numerical recipe. It does
not select training data:

```bash
tda training configure EXPERIMENT \
  --lora-rank 16 --lora-alpha 32 --learning-rate 6e-5 \
  --epochs 2 --batch-size 12 --seed 5678
```

The generated `training-runs/run-<UTC timestamp>/manifest.yaml` records dtype,
LoRA, AdamW, epoch, batch-size, and seed settings. Different recipes or seeds
require different runs.

## `tda training start`

`start` selects one model inside the configured run:

```bash
tda training start EXPERIMENT RUN --training-pool --device cuda:0
tda training start EXPERIMENT RUN --subset subset-0007 --device cuda:0
```

The training-pool model must start first. It resolves the ordered trainable LoRA
parameters and optimizer groups once in the run manifest. Every subset model
must match that layout and inherits the same recipe.

```text
training-runs/<run>/
├── manifest.yaml
├── training-pool/
│   ├── metrics.jsonl
│   └── checkpoints/epoch-NNNN/
│       ├── adapter/
│       ├── optimizer.pt
│       └── metrics.json
└── subsets/<subset-id>/
    ├── metrics.jsonl
    └── checkpoints/epoch-NNNN/
        ├── adapter/
        ├── optimizer.pt
        └── metrics.json
```

Training and validation loss are measured before optimization and after each
epoch. Each trained epoch creates one immutable checkpoint. The CLI performs no
automatic checkpoint selection and stores no duplicated final target.

## `tda projection init`

`init` requires an explicit complete-pool checkpoint from the same run:

```bash
tda projection init EXPERIMENT RUN PROJECTION \
  --training-checkpoint training-pool/checkpoints/epoch-0002 \
  --output-dimension 4096 --seed 1234
```

The checkpoint path is relative to the run and must have the exact
`training-pool/checkpoints/epoch-NNNN` shape. The command initializes the
random matrices from the run-level ordered parameter layout and writes:

```text
training-runs/<run>/projections/<projection>/
├── manifest.yaml
└── matrices.pt
```

The output dimension must be a positive square. Initialization loads no model,
optimizer, encoded data, or gradients.

## `tda projection apply`

`apply --training-pool` and `apply --query-pool` reload the adapter and AdamW
state selected in the projection manifest. They validate the parameter order,
AdamW hyperparameters, optimizer step, and moment state before computing
per-example gradients and applying second-moment correction. They write:

```text
projections/<projection>/projected/
├── training-pool.pt
└── query-pool.pt
```

Each artifact contains ordered utterance IDs and a floating matrix shaped
`[examples, projection dimension]`.

## `tda attribution compute`

The attribution command forms the task-weighted Gauss--Newton approximation and applies its
pseudoinverse square root to both projected pools, normalizes each row, and
writes:

```text
projections/<projection>/attribution/
├── config.yaml
└── attributions.pt
```

`config.yaml` stores the required task weight. `attributions.pt` is the floating
attribution tensor shaped `[training examples, query examples]`.

## `tda projected-cosine compute`

Projected cosine directly normalizes the same corrected projected gradients and
writes its attribution tensor without loading a model or repeating gradients:

```text
projections/<projection>/projected-cosine/attributions.pt
```

Both attribution commands reject non-finite, zero-norm, incorrectly shaped, or
misordered projected inputs and never replace an existing method directory.

## `tda lds compute`

`compute` receives exactly one attribution path:

```bash
tda lds compute \
  experiments/EXPERIMENT/training-runs/RUN/projections/PROJECTION/attribution/attributions.pt --device cuda:0
```

The path determines the experiment, run, projection, method, and output. The
projection manifest determines the selected full-model epoch. LDS requires the
matching checkpoint for every planned subset at:

```text
training-runs/<run>/subsets/<subset-id>/checkpoints/epoch-NNNN
```

The command evaluates each adapter on the ordered query pool, uses negative
per-query objective as the observed response, sums attribution values over each
subset, computes Spearman correlation across subsets for every query, and takes
the arithmetic mean across queries. It writes only that finite scalar beside the
input:

```text
attribution/lds.pt
projected-cosine/lds.pt
```

The command has no response cache, automatic method discovery, statistical
payload, or alternate artifact format.

The canonical LDS definition is TRAK Definition 2.4:
<https://proceedings.mlr.press/v202/park23c.html>.

## Shared rules

- Errors go to stderr with exit code 1; argument misuse uses argparse errors.
- Experiment workspaces are local and never tracked.
- Dataset integrations own validation of their source layouts.
- Model-derived commands consume the model recorded in `manifest.yaml`.
- Training, attribution, and evaluation commands consume persisted plans and
  encoded data rather than resampling them implicitly.
