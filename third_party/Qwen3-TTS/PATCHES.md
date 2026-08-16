# Project modifications to vendored Qwen3-TTS

Upstream baseline: `022e286b98fbec7e1e916cb940cdf532cd9f488e` from https://github.com/QwenLM/Qwen3-TTS.

## Policy

- Framework behavior for this model — training forward passes, per-example objectives, gradient collection, adapter injection, speaker conditioning, serialization — is added directly to the upstream classes here, so `src/` uses `qwen_tts` classes as they are.
- Add only what the framework needs; do not restructure upstream code around it.
- Keep modifications minimal and do not remove or alter upstream licensing notices.
- For every direct edit, record the affected files, purpose, behavioral change, and validating tests below.
- Never place model weights or generated artifacts in this directory.

## Applied modifications

- `assets/Qwen3_TTS.pdf` is removed from Git tracking but kept on disk to slim the repository; its provenance is pinned in `references/sources.yaml`. This tracking change does not alter upstream runtime behavior.
