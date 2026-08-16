# CLI

Status: **`tda data encode` implemented; experiment commands specified, not yet implemented**

## Scope

The package installs one console command, `tda`. It is the only user surface of
the framework. The CLI layer parses arguments and composes importable modules;
it contains no dataset, model, or training logic of its own.

Obtaining datasets is out of scope. The user provides the extracted dataset on
disk and verifies it against `references/sources.yaml`. The CLI starts at data
that already exists.

## Command tree

```text
tda
├── data
│   └── encode   <dataset>   [--data-root] [--output] [--tokenizer-path]
│                            [--device] [--batch-size]
└── experiment                                                    (planned)
    ├── init        <name>   --dataset <dataset>
    ├── plan        <name>
    ├── materialize <name>   [--device]
    ├── train       <name>   (--target | --subset <number>)
    └── status      <name>
```

## tda data encode

One command prepares a dataset end to end: it parses the source layout,
validates it, and encodes every utterance with the pinned 12Hz tokenizer.

The output is one JSONL file:

```text
data/processed/dailytalk_qwen3tts.jsonl
```

Each line is one utterance with this shape:

```json
{"audio_codes":[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]],"audio_path":"data/2/0_0_d2.wav","dialogue":"2","id":"2-0","speaker":"0","text":"Hello"}
```

`audio_codes` contains one list of 16 integers per frame.

Behavior:

- fails with an actionable message when the dataset layout is not at
  `--data-root`;
- writes the output file line by line and, on restart, skips IDs that are
  already encoded;
- writes a manifest sidecar with the tokenizer path, utterance count, and
  output content hash.

Encoding needs the vendored `qwen-tts` package and a GPU:

```bash
uv run --group qwen tda data encode dailytalk
```

## Experiment commands (planned)

One experiment is one directory `experiments/<name>/`, never tracked by git.

- `init` writes `config.yaml` from a template with the dataset filled in and
  refuses to overwrite an existing experiment. The config uses descriptive
  keys: `training_pool_size`, `subset_count`, `subset_size`, `speaker_count`,
  `sampling_seed`.
- `plan` makes every decision and no computation: it samples the training
  pool and subset ID lists by utterance, plus one reference utterance per
  speaker (excluded from the pool), all from `sampling_seed`, and writes
  `plan.json`. Dialogue is an analysis label, not a sampling constraint. The
  same config always regenerates an identical plan; a changed config refuses
  to overwrite an existing plan without `--force`.
- `materialize` makes every computation and no decision: it selects the encoded
  records for the plan IDs, computes one speaker embedding per reference with
  the model's speaker encoder, and writes `train_data.json` with a `speakers`
  mapping and a `data` list.
- `train` executes one run per call: the target run, or the subset run
  selected by `--subset`. Checkpoints and the resolved run manifest go to
  `experiments/<name>/runs/`. Training internals belong to the fine-tuning
  specification.
- `status` prints stage completion: config, plan, train_data, and runs done
  versus `subset_count`.

## Shared rules

- Every command checks its prerequisites and names the command to run first
  instead of raising a stack trace.
- Errors go to stderr with exit code 1; argument misuse exits with the
  argparse convention.
- Paths resolve relative to the repository root and come from defaults or the
  experiment config, never from hard-coded logic downstream.
- A new dataset integration adds one module and one name accepted by
  `tda data encode`; experiment commands stay unchanged.
