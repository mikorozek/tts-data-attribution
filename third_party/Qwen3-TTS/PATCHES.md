# Project modifications to vendored Qwen3-TTS

Upstream baseline: `022e286b98fbec7e1e916cb940cdf532cd9f488e` from https://github.com/QwenLM/Qwen3-TTS.

## Policy

- Framework behavior is composed around `qwen_tts` objects in `src/tts_data_attribution/models/qwen3_tts/` whenever a wrapper can reach what it needs; edit the upstream classes here only for training-forward, loss, or gradient behavior a wrapper cannot reach.
- Add only what the framework needs; do not restructure upstream code around it.
- Keep modifications minimal and do not remove or alter upstream licensing notices.
- For every direct edit, record the affected files, purpose, behavioral change, and validating tests below.
- Never place model weights or generated artifacts in this directory.

## Applied modifications

- `assets/Qwen3_TTS.pdf` is removed from Git tracking but kept on disk to slim the repository; its provenance is pinned in `references/sources.yaml`. This tracking change does not alter upstream runtime behavior.
