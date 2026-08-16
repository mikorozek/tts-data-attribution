# Dataset Integration

Status: **implemented; intentionally small**

## Scope

`UtteranceDataset` is the data ready for an experiment: a tuple of `Utterance`
values behind the PyTorch dataset interface, read from and written to JSONL.
A dataset×model class such as `DailyTalkQwen3TTSDataset` extends it with two
duties: `load()` turns the raw source layout into utterances, and `encode()`
fills their `audio_codes` with that model's tokenizer and appends the result to
the encoded JSONL.

```text
raw layout → DailyTalkQwen3TTSDataset(data_root) → .encode(...) → dailytalk_qwen3tts.jsonl
                                                                          ↓
                                                     UtteranceDataset.from_jsonl()  (experiments)
```

A new dataset or model pairing is a new subclass; nothing existing changes.

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


class DailyTalkQwen3TTSDataset(UtteranceDataset):
    def __init__(self, data_root: str | Path) -> None: ...

    def load(self) -> list[Utterance]: ...

    def encode(self, tokenizer_path: Path, output: Path, device: str, batch_size: int) -> None: ...
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

`encode()` reads the IDs already present in `output`, encodes only the missing
utterances in `DataLoader` batches of `batch_size`, and appends one line per
utterance after each batch. An interrupted run resumes by rerunning the same
command. The tokenizer is loaded lazily inside `encode()`, so constructing the
dataset stays cheap. `CodesEncoder` in `models.qwen3_tts` wraps the pinned 12Hz
tokenizer and returns one `(frames, 16)` tensor per audio path.

There is no schema validator. `JSONDecodeError`, `TypeError`, and `OSError`
propagate unchanged.

## DailyTalk

`load()` reads the official `metadata.json`, orders dialogues and utterances
numerically, and takes each transcript from the per-utterance text file next to
the audio. Eight metadata transcripts differ from those files, so the text files
win. Exact counts are recorded in `references/sources.yaml`.

## Deliberately outside this interface

- integration registries and plugin discovery;
- dataset downloading;
- split orchestration and reference-voice selection (experiment plan);
- model-specific collation and tensor construction (training).
