from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import FakeTokenizer, write_dailytalk_fixture

from tts_data_attribution.cli.data import run_encode
from tts_data_attribution.cli.main import build_parser, main


def encode_arguments(tmp_path: Path) -> list[str]:
    return [
        "data",
        "encode",
        "dailytalk",
        "qwen3-tts",
        "--data-root",
        str(tmp_path / "raw"),
        "--output",
        str(tmp_path / "encoded.jsonl"),
        "--tokenizer-path",
        str(tmp_path / "tokenizer"),
        "--device",
        "cpu",
    ]


def test_parser_maps_data_encode_to_its_command(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(encode_arguments(tmp_path))

    assert arguments.run is run_encode
    assert (arguments.dataset, arguments.model) == ("dailytalk", "qwen3-tts")
    assert arguments.data_root == tmp_path / "raw"
    assert arguments.output == tmp_path / "encoded.jsonl"
    assert arguments.tokenizer_path == tmp_path / "tokenizer"
    assert arguments.device == "cpu"
    assert arguments.batch_size == 16


def test_parser_requires_the_three_paths(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["data", "encode", "dailytalk", "qwen3-tts"])
    assert "--data-root" in capsys.readouterr().err


def test_parser_rejects_an_unknown_model(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["data", "encode", "dailytalk", "other-model"])
    assert "invalid choice: 'other-model'" in capsys.readouterr().err


def test_main_fails_cleanly_without_a_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(encode_arguments(tmp_path)) == 1
    assert "metadata.json is missing" in capsys.readouterr().err


def test_main_fails_cleanly_without_the_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_dailytalk_fixture(tmp_path / "raw")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    assert main(encode_arguments(tmp_path)) == 1
    assert "tokenizer not found" in capsys.readouterr().err


def test_encode_command_writes_the_dataset_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_tokenizer: FakeTokenizer
) -> None:
    write_dailytalk_fixture(tmp_path / "raw")
    (tmp_path / "tokenizer").mkdir()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    assert main(encode_arguments(tmp_path)) == 0

    output = tmp_path / "encoded.jsonl"
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == ["2-0", "2-1", "10-0"]
    assert records[0] == {
        "audio_codes": [[7] * 16] * 2,
        "audio_path": "data/2/0_0_d2.wav",
        "dialogue": "2",
        "id": "2-0",
        "speaker": "0",
        "text": "First",
    }
    assert json.loads((tmp_path / "encoded.manifest.json").read_text(encoding="utf-8")) == {
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "tokenizer_path": (tmp_path / "tokenizer").as_posix(),
        "utterance_count": 3,
    }
