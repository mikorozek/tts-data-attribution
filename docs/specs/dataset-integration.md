# Dataset Integration

Status: **implemented; intentionally small**

## Scope

Two independent axes compose at the command line: a dataset class loads a raw
source layout into `Utterance` values, and an encoder class fills their
`audio_codes` with one model's tokenizer. Any dataset works with any encoder,
so a new dataset or a new model is one new class each, never a pairing.

```text
DailyTalkDataset(data_root)  ─┐
                              ├─ Qwen3TTSEncoder.encode(dataset, audio_root, output, batch_size)
UtteranceDataset (any)       ─┘                     ↓
                                     dailytalk_qwen3tts.jsonl  →  UtteranceDataset.from_jsonl()
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


class UtteranceDataset(torch.utils.data.Dataset[Utterance]):
    def __init__(self, utterances: Iterable[Utterance]) -> None: ...

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Self: ...

    def to_jsonl(self, path: str | Path, append: bool = False) -> None: ...

    def ids(self) -> set[str]: ...


class DailyTalkDataset(UtteranceDataset):
    def __init__(self, data_root: str | Path) -> None: ...   # keeps self.data_root


class Qwen3TTSEncoder:                                       # models.qwen3_tts
    def __init__(self, tokenizer: AudioCodesTokenizer) -> None: ...

    @classmethod
    def from_pretrained(cls, tokenizer_path: str | Path, device: str) -> Self: ...

    def encode(
        self, dataset: UtteranceDataset, audio_root: Path, output: Path, batch_size: int
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

`to_jsonl` writes one line per utterance, keys sorted, UTF-8 preserved; it is
the only place that serializes utterances:

```json
{"audio_codes":[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]],"audio_path":"data/2/0_0_d2.wav","dialogue":"2","id":"2-0","speaker":"0","text":"Hello"}
```

## Encoding behavior

`Qwen3TTSEncoder.encode()` reads the IDs already present in `output`, encodes
only the missing utterances in `DataLoader` batches of `batch_size`, and appends
one line per utterance after each batch through `to_jsonl(append=True)`. An
interrupted run resumes by rerunning the same command. The tokenizer is
constructed by `from_pretrained`, the only place that imports `qwen_tts`; tests
inject a fake tokenizer through the constructor.

There is no schema validator. `JSONDecodeError`, `TypeError`, and `OSError`
propagate unchanged.

## DailyTalk

`DailyTalkDataset` reads the official `metadata.json`, orders dialogues and
utterances numerically, and takes each transcript from the per-utterance text
file next to the audio. Eight metadata transcripts differ from those files, so
the text files win. Exact counts are recorded in `references/sources.yaml`.

## Deliberately outside this interface

- integration registries and plugin discovery beyond the two CLI mappings;
- dataset downloading;
- split orchestration and reference-voice selection (experiment plan);
- model-specific collation and tensor construction (training).
