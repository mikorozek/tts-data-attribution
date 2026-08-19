# CLI

Status: **`tda experiment init` and `tda experiment encode` implemented; training and query commands planned**

## Command tree

```text
tda
└── experiment
    ├── init    <name> --dataset <dataset> --data-root <dir>
    │                  --model <model> --model-path <dir>
    │                  --training-pool-size N --validation-pool-size N
    │                  --subset-count N --subset-size N
    │                  --speaker-count N --seed N [--root]
    ├── encode  <name> [--device] [--batch-size] [--root]
    └── train   <name>                                           (planned)
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

## Shared rules

- Errors go to stderr with exit code 1; argument misuse uses argparse errors.
- Experiment workspaces are local and never tracked.
- Dataset integrations own validation of their source layouts.
- Model-derived commands consume the model recorded in `manifest.yaml`.
- Training, attribution, and evaluation commands consume persisted plans and
  encoded data rather than resampling them implicitly.
