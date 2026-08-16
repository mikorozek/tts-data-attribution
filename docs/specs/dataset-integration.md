# Dataset Integration

Status: **implemented; intentionally small**

## Scope

A dataset integration turns its source into `Utterance` values. `UtteranceDataset`
stores them, implements the PyTorch dataset interface, and reads or writes JSONL.
Encoding fills the `audio_codes` of every utterance and appends the result to the
encoded JSONL that training later consumes.

```text
raw dataset → load_<dataset>() → UtteranceDataset → encode_utterances() → <dataset>_encoded.jsonl
```

## API

```python
@dataclass(frozen=True)
class Utterance:
    id: str
    text: str
    speaker: str
    dialogue: str
    audio_path: str
    audio_codes: list[list[int]] | None = None

    def to_json(self) -> str: ...


class UtteranceDataset(torch.utils.data.Dataset[Utterance]):
    def __init__(self, utterances: Iterable[Utterance]) -> None: ...

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Self: ...

    def to_jsonl(self, path: str | Path) -> None: ...

    def ids(self) -> set[str]: ...


def load_dailytalk(root: str | Path) -> UtteranceDataset: ...


def encode_utterances(
    dataset: UtteranceDataset,
    audio_root: Path,
    encoder: Callable[[list[Path]], list[torch.Tensor]],
    output: Path,
    batch_size: int,
) -> None: ...
```

## Utterance contract

- `id` identifies the same utterance across repeated loads and runs.
- `text` is the transcript the model must speak.
- `speaker` selects the reference voice at experiment time.
- `dialogue` is an analysis label; sampling does not group by it.
- `audio_path` is portable and relative to the dataset root.
- `audio_codes` is `None` after loading and one list of 16 integers per
  12.5 Hz frame after encoding.

One JSONL line per utterance, keys sorted, UTF-8 preserved:

```json
{"audio_codes":[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]],"audio_path":"data/2/0_0_d2.wav","dialogue":"2","id":"2-0","speaker":"0","text":"Hello"}
```

## Encoding behavior

`encode_utterances()` reads the IDs already present in the output, encodes only
the missing utterances in batches, and appends one line per utterance after
each batch. An interrupted run resumes by rerunning the same command. The
encoder is any callable from audio paths to one `(frames, 16)` tensor per path;
`CodesEncoder` in `models.qwen3_tts` wraps the pinned 12Hz tokenizer.

There is no schema validator. `JSONDecodeError`, `TypeError`, and `OSError`
propagate unchanged.

## DailyTalk

`load_dailytalk()` reads the official `metadata.json`, orders dialogues and
utterances numerically, and takes each transcript from the per-utterance text
file next to the audio. Eight metadata transcripts differ from those files, so
the text files win. Exact counts are recorded in `references/sources.yaml`.

## Deliberately outside this interface

- integration registries and plugin discovery;
- dataset downloading;
- split orchestration and reference-voice selection (experiment plan);
- model-specific collation and tensor construction (training).
