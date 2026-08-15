#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_DIR="$ROOT/artifacts/models/Qwen3-TTS-12Hz-1.7B-Base-fd4b254"
TOKENIZER_DIR="$ROOT/artifacts/models/Qwen3-TTS-Tokenizer-12Hz-7dd38ad"

mkdir -p "$ROOT/artifacts/models"

uvx --from 'huggingface-hub>=0.34,<1' hf download \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --revision fd4b254389122332181a7c3db7f27e918eec64e3 \
  --local-dir "$MODEL_DIR"

uvx --from 'huggingface-hub>=0.34,<1' hf download \
  Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --revision 7dd38ad4e9bad454aae9cd937d0cd577604fe229 \
  --local-dir "$TOKENIZER_DIR"

printf 'Model snapshot: %s\nTokenizer snapshot: %s\n' "$MODEL_DIR" "$TOKENIZER_DIR"
