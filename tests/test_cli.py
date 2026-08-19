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
            "--model",
            "qwen3-tts",
            "--model-path",
            "model",
            "--training-pool-size",
            "8",
            "--validation-pool-size",
            "2",
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
    encode = build_parser().parse_args(["experiment", "encode", "study"])

    assert init.run is run_init
    assert init.data_root == Path("raw")
    assert init.model == "qwen3-tts"
    assert init.model_path == Path("model")
    assert not hasattr(init, "device")
    assert encode.run is run_encode
    assert encode.batch_size == 16
    assert not hasattr(encode, "model")
    assert not hasattr(encode, "model_path")
