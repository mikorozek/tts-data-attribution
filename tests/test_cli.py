from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tts_data_attribution.cli import build_parser, main
from tts_data_attribution.cli.data import run_encode
from tts_data_attribution.models.qwen3_tts import validate_codes


def write_utterance(
    root: Path,
    dialogue_id: int,
    utterance_id: int,
    speaker: int,
    text: str,
) -> None:
    directory = root / "data" / str(dialogue_id)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{utterance_id}_{speaker}_d{dialogue_id}"
    (directory / f"{stem}.txt").write_text(text, encoding="utf-8")
    (directory / f"{stem}.wav").write_bytes(b"")


def write_dailytalk_fixture(root: Path) -> None:
    metadata = {
        "2": {
            "0": {
                "index": "2-0",
                "turn": 1,
                "topic": 1,
                "emotion": "no emotion",
                "act": "inform",
                "speaker": 0,
                "text": "First",
                "dialog_idx": 2,
                "utterance_idx": 0,
            },
            "1": {
                "index": "2-1",
                "turn": 1,
                "topic": 1,
                "emotion": "no emotion",
                "act": "question",
                "speaker": 1,
                "text": "Second",
                "dialog_idx": 2,
                "utterance_idx": 1,
            },
        }
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_utterance(root, 2, 0, 0, "First")
    write_utterance(root, 2, 1, 1, "Second")


class FakeEncoder:
    def __init__(self, tokenizer_path: Path, device: str) -> None:
        self.device = device

    def encode(self, audio_paths: list[str]) -> list[list[list[int]]]:
        return [[[7] * 16] * 2 for _ in audio_paths]


def encode_arguments(tmp_path: Path) -> list[str]:
    return [
        "data",
        "encode",
        "dailytalk",
        "--data-root",
        str(tmp_path / "raw"),
        "--output",
        str(tmp_path / "encoded.jsonl"),
        "--tokenizer-path",
        str(tmp_path / "tokenizer"),
        "--device",
        "cpu",
    ]


def test_parser_maps_data_encode_to_its_command() -> None:
    arguments = build_parser().parse_args(["data", "encode", "dailytalk"])

    assert arguments.run is run_encode
    assert arguments.dataset == "dailytalk"
    assert arguments.data_root == Path("data/raw/dailytalk")
    assert arguments.output == Path("data/processed/dailytalk_encoded.jsonl")
    assert arguments.tokenizer_path == Path(
        "artifacts/models/Qwen3-TTS-Tokenizer-12Hz-7dd38ad"
    )
    assert arguments.device == "cuda:0"
    assert arguments.batch_size == 16


def test_main_fails_cleanly_without_a_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(encode_arguments(tmp_path))

    assert exit_code == 1
    assert "metadata.json is missing" in capsys.readouterr().err


def test_encode_writes_output_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_dailytalk_fixture(tmp_path / "raw")
    (tmp_path / "tokenizer").mkdir()
    monkeypatch.setattr(
        "tts_data_attribution.models.qwen3_tts.CodesEncoder",
        FakeEncoder,
    )

    exit_code = main(encode_arguments(tmp_path))

    assert exit_code == 0
    output = tmp_path / "encoded.jsonl"
    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((tmp_path / "encoded.manifest.json").read_text(encoding="utf-8"))
    assert records == [
        {
            "audio_codes": [[7] * 16] * 2,
            "dialogue": "2",
            "id": "2-0",
            "speaker": "0",
            "text": "First",
        },
        {
            "audio_codes": [[7] * 16] * 2,
            "dialogue": "2",
            "id": "2-1",
            "speaker": "1",
            "text": "Second",
        },
    ]
    assert manifest == {
        "encoded_count": 2,
        "example_count": 2,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "tokenizer_path": (tmp_path / "tokenizer").as_posix(),
    }


def test_encode_skips_ids_that_are_already_encoded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_dailytalk_fixture(tmp_path / "raw")
    (tmp_path / "tokenizer").mkdir()
    existing_record = {
        "audio_codes": [[1] * 16],
        "dialogue": "2",
        "id": "2-0",
        "speaker": "0",
        "text": "First",
    }
    (tmp_path / "encoded.jsonl").write_text(
        json.dumps(existing_record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tts_data_attribution.models.qwen3_tts.CodesEncoder",
        FakeEncoder,
    )

    exit_code = main(encode_arguments(tmp_path))

    assert exit_code == 0
    records = [
        json.loads(line)
        for line in (tmp_path / "encoded.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0] == existing_record
    assert records[1] == {
        "audio_codes": [[7] * 16] * 2,
        "dialogue": "2",
        "id": "2-1",
        "speaker": "1",
        "text": "Second",
    }
    assert "encoded 1/1" in capsys.readouterr().out


def test_validate_codes_accepts_a_matching_tensor() -> None:
    codes = torch.zeros((25, 16), dtype=torch.long)

    assert validate_codes(codes, expected_frames=25, audio_path="a.wav") == [[0] * 16] * 25


def test_validate_codes_rejects_a_wrong_codebook_count() -> None:
    codes = torch.zeros((25, 8), dtype=torch.long)

    with pytest.raises(ValueError, match="unexpected code shape"):
        validate_codes(codes, expected_frames=25, audio_path="a.wav")


def test_validate_codes_rejects_an_implausible_frame_count() -> None:
    codes = torch.zeros((40, 16), dtype=torch.long)

    with pytest.raises(ValueError, match="expected about 25"):
        validate_codes(codes, expected_frames=25, audio_path="a.wav")
