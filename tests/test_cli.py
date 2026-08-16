from __future__ import annotations

from pathlib import Path

from tts_data_attribution.cli.experiment import run_encode, run_init
from tts_data_attribution.cli.main import build_parser


def test_parser_maps_experiment_commands() -> None:
    init = build_parser().parse_args(
        [
            "experiment",
            "init",
            "study",
            "--dataset",
            "dailytalk",
            "--data-root",
            "raw",
            "--training-pool-size",
            "8",
            "--subset-count",
            "3",
            "--subset-size",
            "4",
            "--speaker-count",
            "2",
            "--seed",
            "7",
        ]
    )
    encode = build_parser().parse_args(
        [
            "experiment",
            "encode",
            "study",
            "--model",
            "qwen3-tts",
            "--model-path",
            "model",
        ]
    )

    assert init.run is run_init
    assert init.data_root == Path("raw")
    assert not hasattr(init, "model")
    assert not hasattr(init, "model_path")
    assert not hasattr(init, "device")
    assert encode.run is run_encode
    assert encode.model_path == Path("model")
    assert encode.batch_size == 16
