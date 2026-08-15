# Project modifications to vendored Qwen3-TTS

Upstream baseline: `022e286b98fbec7e1e916cb940cdf532cd9f488e` from https://github.com/QwenLM/Qwen3-TTS.

## Policy

- Prefer framework/integration wrappers when they can implement the required behavior cleanly.
- Direct edits are allowed for training-forward, loss-alignment, gradient, or serialization behavior that cannot be safely provided by a wrapper.
- Keep modifications minimal and do not remove or alter upstream licensing notices.
- For every direct edit, record the affected files, purpose, behavioral change, and validating tests below.
- Never place model weights or generated artifacts in this directory.

## Applied modifications

None yet. `UPSTREAM.md` and this file are repository metadata, not changes to upstream runtime behavior.
