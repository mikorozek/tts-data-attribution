# Qwen3-TTS + DailyTalk TrackStar/LDS experiment

This directory contains the first concrete study built with the reusable attribution framework. Nothing in this experiment is a framework-wide default.

## Goal

Validate TrackStar for controlled LoRA fine-tuning of Qwen3-TTS using counterfactual subset retraining and LDS. Attribution claims apply only to examples included in this fine-tuning study, never to unknown Qwen pretraining data.

## Current candidate assets

- Qwen3-TTS 1.7B Base: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- Qwen speech tokenizer: https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz
- Qwen source: https://github.com/QwenLM/Qwen3-TTS
- DailyTalk: https://github.com/keonlee9420/DailyTalk
- Exact revisions and checksums: `sources.yaml` and `../../references/sources.yaml`
- Pinned vendored source: `../../third_party/Qwen3-TTS/`; direct project modifications must be documented in its `PATCHES.md`.
- Ignored local model snapshots: `../../artifacts/models/`

DailyTalk contains two fixed speakers, 2,541 dialogues, and 23,773 utterance-level clips (approximately 21.7 hours). The dataset is CC BY-SA 4.0; the repository code has a separate MIT license.

## Intended integration

The Qwen integration must provide a fresh `torch.nn.Module`, reference/target preprocessing, explicit per-example teacher-forced loss, serialization, and configurable parameter selection. The DailyTalk integration must convert raw utterances to stable records, preserve speaker/dialogue metadata, and create leakage-safe configurable partitions.

The codec and reference-voice encoder are expected to remain frozen in this study. Adapter placement and all hyperparameters are experiment configuration rather than framework assumptions.

## Known feasibility issue

The public fine-tuning script is a reference, not a trusted objective implementation. Its manual shifts combined with the pinned Transformers causal loss appear capable of shifting labels twice, and the within-frame predictor alignment needs a target-leakage audit. Implement and unit-test explicit per-example masked loss before any scaled run.

Required pilot: encode/decode, reference-conditioned inference, loss-index tests, parameter-selection audit, finite gradients, adapter save/reload, one-example and small-batch overfit, then a tiny end-to-end TrackStar/LDS experiment.

## Files

```text
experiment.yaml            preliminary full experiment composition
lora.yaml                  preliminary Qwen-specific adapter selection
sources.yaml               experiment-specific source pins
```

These configurations are placeholders until the corresponding framework integrations and schemas are implemented. Revise them through validated configuration, not by introducing Qwen or DailyTalk assumptions into framework-core code.
