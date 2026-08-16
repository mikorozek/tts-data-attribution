# Dataset integration

Status: **implemented for transient DailyTalk loading and experiment-scoped Qwen3-TTS encoding**

## Data flow

```text
DailyTalkDataset
        ↓ experiment init samples IDs
Plan
        ↓ experiment encode reloads only planned records
Qwen3TTSEncoder
        ↓
UtteranceDataset.from_jsonl / to_jsonl
```

## Source datasets

A source integration such as `DailyTalkDataset` reads its native on-disk
layout and keeps transient records keyed by stable ID. `get_records()` supports
sampling and `get_records_by_ids()` retrieves the records selected by the plan.
Source records contain text, speaker, dialogue, and relative audio paths, but
have no common persisted JSONL representation.

`DailyTalkDataset` uses per-utterance transcript files rather than duplicate
metadata text and orders numeric dialogue and utterance IDs deterministically.

## Encoded utterances

```python
@dataclass(frozen=True)
class Utterance:
    id: str
    speaker: str
    dialogue: str
    text_ids: list[int]
    audio_codes: list[list[int]]
```

`text_ids` are produced by the selected model's text processor.
`audio_codes` contains one row of 16 integer codebook IDs per 12 Hz frame.
The encoded record intentionally omits raw text and paths; the experiment
manifest and stable ID identify its source.

`UtteranceDataset` is the only serialized dataset type. It implements
`torch.utils.data.Dataset`, deterministic JSONL loading and writing, append
mode, and ID lookup.

## Qwen3-TTS encoding

`Qwen3TTSEncoder` loads one `Qwen3TTSModel` and exposes:

```python
encode_text(text)
encode_audio(audio_paths)
encode_utterances(source_records, data_root)
encode_speaker(reference_audio_path)
```

Text encoding uses `<|im_start|>assistant\n{text}`, which is the exact prefix
left by the released fine-tuning code after it removes its five-token ChatML
suffix. Audio encoding uses the speech tokenizer bundled with the model.
Speaker encoding loads mono audio at the model's speaker-encoder sample rate
and stores the resulting CPU vector.

The experiment CLI owns plan selection, batching, progress, restart behavior,
and artifact serialization. The encoder owns conversion from transient source
records to persisted `Utterance` values.
