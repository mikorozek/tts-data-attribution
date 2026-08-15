from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Protocol, TypedDict, cast

from ..dataset import AttributionDataset, DailyTalkDataset, DatasetExample
from .errors import CommandError

DAILYTALK = "dailytalk"
DEFAULT_DATA_ROOT = Path("data/raw/dailytalk")
DEFAULT_OUTPUT = Path("data/processed/dailytalk_encoded.jsonl")
DEFAULT_TOKENIZER_PATH = Path("artifacts/models/Qwen3-TTS-Tokenizer-12Hz-7dd38ad")


class AudioCodesEncoder(Protocol):
    def encode(self, audio_paths: list[str]) -> list[list[list[int]]]: ...


class EncodedRecord(TypedDict):
    audio_codes: list[list[int]]
    dialogue: str
    id: str
    speaker: str
    text: str


def register(subparsers: argparse._SubParsersAction) -> None:
    data_parser = subparsers.add_parser("data", help="dataset preparation")
    data_subparsers = data_parser.add_subparsers(required=True)
    encode_parser = data_subparsers.add_parser(
        "encode",
        help="encode every dataset utterance into audio codes",
    )
    encode_parser.add_argument("dataset", choices=[DAILYTALK])
    encode_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    encode_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    encode_parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    encode_parser.add_argument("--device", default="cuda:0")
    encode_parser.add_argument("--batch-size", type=int, default=16)
    encode_parser.set_defaults(run=run_encode)


def run_encode(arguments: argparse.Namespace) -> None:
    dataset = load_dataset(arguments.data_root)
    encode_missing_examples(
        dataset=dataset,
        data_root=arguments.data_root,
        output=arguments.output,
        tokenizer_path=arguments.tokenizer_path,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    write_manifest(
        output=arguments.output,
        tokenizer_path=arguments.tokenizer_path,
        example_count=len(dataset),
    )
    print(f"encoded data ready at {arguments.output}")


def load_dataset(data_root: Path) -> AttributionDataset:
    if not (data_root / "metadata.json").is_file():
        raise CommandError(
            f"no dataset found at {data_root}: metadata.json is missing; "
            "provide the extracted dataset there and verify it against references/sources.yaml"
        )
    return DailyTalkDataset.from_directory(data_root)


def encode_missing_examples(
    dataset: AttributionDataset,
    data_root: Path,
    output: Path,
    tokenizer_path: Path,
    device: str,
    batch_size: int,
) -> None:
    encoded_ids = read_encoded_ids(output)
    pending = [example for example in dataset.examples if example.id not in encoded_ids]
    if not pending:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch(exist_ok=True)
        print(f"all {len(dataset)} utterances are already encoded")
        return
    if not tokenizer_path.exists():
        raise CommandError(f"tokenizer not found at {tokenizer_path}")
    encoder = load_encoder(tokenizer_path, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for batch in batched(pending, batch_size):
            for example, audio_codes in zip(batch, encode_batch(encoder, batch, data_root), strict=True):
                json.dump(
                    encoded_record(example, audio_codes),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
            stream.flush()
            completed += len(batch)
            print(f"encoded {completed}/{len(pending)}", flush=True)


def encoded_record(
    example: DatasetExample,
    audio_codes: list[list[int]],
) -> EncodedRecord:
    if example.groups is None:
        raise ValueError(f"groups are missing for {example.id}")
    return {
        "audio_codes": audio_codes,
        "dialogue": example.groups["dialogue"],
        "id": example.id,
        "speaker": example.groups["speaker"],
        "text": cast(str, example.payload["text"]),
    }


def load_encoder(tokenizer_path: Path, device: str) -> AudioCodesEncoder:
    from ..models.qwen3_tts import CodesEncoder

    try:
        return CodesEncoder(tokenizer_path=tokenizer_path, device=device)
    except ModuleNotFoundError as error:
        raise CommandError(
            "encoding needs the vendored qwen-tts package; "
            "run: uv run --group qwen tda data encode dailytalk"
        ) from error


def encode_batch(
    encoder: AudioCodesEncoder,
    batch: list[DatasetExample],
    data_root: Path,
) -> list[list[list[int]]]:
    audio_paths = [str(data_root / str(example.payload["audio_path"])) for example in batch]
    return encoder.encode(audio_paths)


def read_encoded_ids(output: Path) -> set[str]:
    if not output.is_file():
        return set()
    with output.open(encoding="utf-8") as stream:
        return {json.loads(line)["id"] for line in stream if line.strip()}


def batched(examples: list[DatasetExample], batch_size: int) -> list[list[DatasetExample]]:
    return [examples[start : start + batch_size] for start in range(0, len(examples), batch_size)]


def write_manifest(
    output: Path,
    tokenizer_path: Path,
    example_count: int,
) -> None:
    manifest = {
        "encoded_count": len(read_encoded_ids(output)),
        "example_count": example_count,
        "output_sha256": file_sha256(output),
        "tokenizer_path": tokenizer_path.as_posix(),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
