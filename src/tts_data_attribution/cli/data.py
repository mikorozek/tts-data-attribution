from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from ..dataset import DailyTalkQwen3TTSDataset
from .errors import CommandError


def register(subparsers: argparse._SubParsersAction) -> None:
    data_parser = subparsers.add_parser("data", help="dataset preparation")
    data_subparsers = data_parser.add_subparsers(required=True)
    encode_parser = data_subparsers.add_parser(
        "encode", help="encode every utterance into audio codes"
    )
    encode_parser.add_argument("dataset", choices=["dailytalk"])
    encode_parser.add_argument("--data-root", type=Path, default=Path("data/raw/dailytalk"))
    encode_parser.add_argument(
        "--output", type=Path, default=Path("data/processed/dailytalk_qwen3tts.jsonl")
    )
    encode_parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("artifacts/models/Qwen3-TTS-Tokenizer-12Hz-7dd38ad"),
    )
    encode_parser.add_argument("--device", default="cuda:0")
    encode_parser.add_argument("--batch-size", type=int, default=16)
    encode_parser.set_defaults(run=run_encode)


def run_encode(arguments: argparse.Namespace) -> None:
    if not (arguments.data_root / "metadata.json").is_file():
        raise CommandError(
            f"no dataset found at {arguments.data_root}: metadata.json is missing; "
            "provide the extracted dataset there and verify it against references/sources.yaml"
        )
    if importlib.util.find_spec("qwen_tts") is None:
        raise CommandError(
            "encoding needs the vendored qwen-tts package; "
            "run: uv run --group qwen tda data encode dailytalk"
        )
    if not arguments.tokenizer_path.exists():
        raise CommandError(f"tokenizer not found at {arguments.tokenizer_path}")
    dataset = DailyTalkQwen3TTSDataset(arguments.data_root)
    dataset.encode(
        tokenizer_path=arguments.tokenizer_path,
        output=arguments.output,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    write_manifest(arguments.output, arguments.tokenizer_path, len(dataset))
    print(f"encoded data ready at {arguments.output}")


def write_manifest(output: Path, tokenizer_path: Path, utterance_count: int) -> None:
    manifest = {
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "tokenizer_path": tokenizer_path.as_posix(),
        "utterance_count": utterance_count,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
