# CLI

Status: **`tda experiment init` and `tda experiment encode` implemented; training and status commands planned**

## Scope

The `tda` command is the user-facing composition layer. Dataset, encoding, and
experiment behavior remains importable outside the CLI.

## Command tree

```text
tda
└── experiment
    ├── init    <name> --dataset <dataset> --data-root <dir>
    │                  --training-pool-size N --validation-pool-size N
    │                  --query-pool-size N --subset-count N --subset-size N
    │                  --speaker-count N --seed N [--root]
    ├── encode  <name> --model <model> --model-path <dir>
    │                  [--device] [--batch-size] [--root]
    ├── train   <name>                                           (planned)
    └── status  <name>                                           (planned)
```

## `tda experiment init`

`init` loads a raw dataset, samples the experiment, and writes no model-derived
data. It never loads a model or requires a GPU.

Validation completes before the experiment directory is created:

- the named dataset must load from `--data-root`;
- `speaker_count` must fit the available speakers;
- the requested pools must fit the eligible dialogue-disjoint utterances;
- `subset_size` must not exceed `training_pool_size`;
- the experiment directory must not already exist.

`manifest.yaml` records the dataset and sampling request:

```yaml
data_root: data/raw/dailytalk
dataset: dailytalk
query_pool_size: 100
seed: 1234
speaker_count: 2
subset_count: 50
subset_size: 1000
training_pool_size: 2000
validation_pool_size: 200
```

`plan.json` records reference utterances, training, validation, and query
pools, and subsets. These partitions are disjoint at dialogue level. The same
manifest produces the same byte-stable plan.

## `tda experiment encode`

`encode` is an explicit model-dependent materialization step. It reads the
experiment plan and encodes the union of the training, validation, and query
pools and reference utterances. It does not encode unsampled dataset records.

For Qwen3-TTS, one loaded `Qwen3TTSModel` provides:

- `processor` for `<|im_start|>assistant\n{text}` text IDs;
- the bundled speech tokenizer for 16-codebook audio codes;
- the speaker encoder for one vector per planned speaker reference.

The command writes:

```text
experiments/<name>/
├── manifest.yaml
├── plan.json
├── encoding.yaml
├── sampled_utterances_encoded.jsonl
└── speaker_embeddings.pt
```

`sampled_utterances_encoded.jsonl` contains `id`, `speaker`, `dialogue`,
`text_ids`, and `audio_codes`. It intentionally omits raw text and audio paths.
`encoding.yaml` records the model name and path used for the materialization.

Encoding appends only complete batches. A repeated invocation with the same
model skips encoded IDs and already stored speaker embeddings; an invocation
with a different recorded model is rejected.

## Shared rules

- Errors go to stderr with exit code 1; argument misuse uses argparse errors.
- Experiment workspaces are local and never tracked.
- Dataset integrations own validation of their source layouts.
- A model contributes one focused experiment encoder and one entry in
  `models.EXPERIMENT_ENCODERS`.
- Training, attribution, and evaluation commands consume the immutable plan
  and encoded experiment data rather than resampling or re-encoding it.
