# CLI

Status: **`tda data encode` and `tda experiment init` implemented; `train` and `status` specified, not yet implemented**

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
│   └── encode   <dataset> <model>   [--data-root] [--output] [--tokenizer-path]
│                                    [--device] [--batch-size]
└── experiment
    ├── init        <name>   --dataset <encoded.jsonl> --audio-root <dir>
    │                        --model <model> --model-path <dir>
    │                        --training-pool-size N --subset-count N --subset-size N
    │                        --speaker-count N --seed N [--device] [--root]
    ├── train       <name>   (--target | --subset <number>)           (planned)
    └── status      <name>                                            (planned)
```

## tda data encode

One command prepares a dataset end to end: it loads the named dataset,
validates its layout, and encodes every utterance with the named model's
tokenizer. Datasets and models are independent choices; any pair works.

`--data-root`, `--tokenizer-path`, and `--output` are required: they name a
dataset and a model, so the generic command has no default for them. The
output is one JSONL file.

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

Encoding needs a GPU:

```bash
uv run tda data encode dailytalk qwen3-tts \
  --data-root data/raw/dailytalk \
  --tokenizer-path artifacts/models/Qwen3-TTS-Tokenizer-12Hz-7dd38ad \
  --output data/processed/dailytalk_qwen3tts.jsonl
```

## tda experiment init

One experiment is one directory `experiments/<name>/`, never tracked by git,
and one command defines it completely. `init` takes the identity of the
experiment (one encoded dataset with its raw audio root, one model) and its
sampling, validates everything, and writes three files. A different sampling
is a different experiment.

Validation, in order, before anything is created:

- the encoded dataset must load through `UtteranceDataset.from_jsonl`;
- the sampling must fit the dataset: `speaker_count` ≤ speakers present,
  `training_pool_size` ≤ candidate utterances, `subset_size` ≤
  `training_pool_size`;
- the model must load through the upstream `Qwen3TTSModel.from_pretrained`
  on the given device;
- the experiment directory must not exist yet.

`manifest.yaml` records what was asked for:

```yaml
audio_root: data/raw/dailytalk
dataset: data/processed/dailytalk_qwen3tts.jsonl
model: qwen3-tts
model_path: artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254
seed: 1234
speaker_count: 2
subset_count: 50
subset_size: 1000
training_pool_size: 2000
```

`plan.json` records what the sampling produced, deterministically from `seed`:

```json
{
  "references": {"0": "870-6", "1": "1201-3"},
  "training_pool": ["0-0", "0-3", "..."],
  "subsets": [["0-0", "..."], ["..."]]
}
```

`references` holds one reference utterance per speaker; those utterances are
excluded from the pool. `training_pool` is drawn from the remaining utterances
of the chosen speakers, and every subset is drawn from the pool. Speakers are
the first `speaker_count` speaker IDs in sorted order. Sampling is by utterance;
`dialogue` is not a constraint. The same manifest always yields a
byte-identical plan.

`speaker_embeddings.pt` holds one speaker-encoder vector per reference
utterance, keyed by speaker, computed by the loaded model from the reference
wav resampled to the speaker encoder's sample rate. Training conditions every
utterance of a speaker on this fixed vector; the reference wavs are never read
again.

Every later command reads these three files and never takes a data, model, or
sampling value on the command line again.

## Experiment commands (planned)

- `train` executes one run per call: the target run, or the subset run
  selected by `--subset`. Checkpoints and the resolved run manifest go to
  `experiments/<name>/runs/`. Training internals belong to the fine-tuning
  specification.
- `status` prints stage completion: experiment files present and runs done
  versus `subset_count`.

## Shared rules

- Every command checks its prerequisites and names the command to run first
  instead of raising a stack trace.
- Errors go to stderr with exit code 1; argument misuse exits with the
  argparse convention.
- Paths resolve relative to the repository root and come from defaults or the
  experiment config, never from hard-coded logic downstream.
- A new dataset adds one class and one name in `DATASETS`; a new model adds
  one encoder class and one name in `ENCODERS`. Experiment commands stay
  unchanged.
