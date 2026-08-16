from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from ..dataset import UtteranceDataset
from ..experiment import ExperimentManifest, Plan
from .errors import CommandError

MODELS = ["qwen3-tts"]


def register(subparsers: argparse._SubParsersAction) -> None:
    experiment_parser = subparsers.add_parser("experiment", help="experiment workspace")
    experiment_subparsers = experiment_parser.add_subparsers(required=True)
    init_parser = experiment_subparsers.add_parser(
        "init", help="define an experiment: dataset, model, and sampled subsets"
    )
    init_parser.add_argument("name")
    init_parser.add_argument("--dataset", type=Path, required=True)
    init_parser.add_argument("--model", choices=MODELS, required=True)
    init_parser.add_argument("--model-path", type=Path, required=True)
    init_parser.add_argument("--training-pool-size", type=int, required=True)
    init_parser.add_argument("--subset-count", type=int, required=True)
    init_parser.add_argument("--subset-size", type=int, required=True)
    init_parser.add_argument("--speaker-count", type=int, required=True)
    init_parser.add_argument("--seed", type=int, required=True)
    init_parser.add_argument("--device", default="cuda:0")
    init_parser.add_argument("--root", type=Path, default=Path("experiments"))
    init_parser.set_defaults(run=run_init)


def run_init(arguments: argparse.Namespace) -> None:
    directory = arguments.root / arguments.name
    if directory.exists():
        raise CommandError(f"experiment {arguments.name} already exists at {directory}")
    manifest = ExperimentManifest(
        dataset=arguments.dataset,
        model=arguments.model,
        model_path=arguments.model_path,
        training_pool_size=arguments.training_pool_size,
        subset_count=arguments.subset_count,
        subset_size=arguments.subset_size,
        speaker_count=arguments.speaker_count,
        seed=arguments.seed,
    )
    dataset = load_dataset(manifest.dataset)
    try:
        plan = Plan.sample(manifest, dataset)
    except ValueError as error:
        raise CommandError(str(error)) from error
    check_model(manifest.model, manifest.model_path, arguments.device)
    directory.mkdir(parents=True)
    manifest.to_yaml(directory / "manifest.yaml")
    plan.to_json(directory / "plan.json")
    print(f"experiment {arguments.name} created at {directory}")


def load_dataset(dataset: Path) -> UtteranceDataset:
    if not dataset.is_file():
        raise CommandError(
            f"encoded dataset not found at {dataset}; run: tda data encode ..."
        )
    try:
        return UtteranceDataset.from_jsonl(dataset)
    except (TypeError, json.JSONDecodeError) as error:
        raise CommandError(
            f"{dataset} is not an encoded utterance file: {error}"
        ) from error


def check_model(model: str, model_path: Path, device: str) -> None:
    if importlib.util.find_spec("qwen_tts") is None:
        raise CommandError(
            "loading the model needs the vendored qwen-tts package; "
            "run: uv run --group qwen tda experiment init ..."
        )
    if not model_path.is_dir():
        raise CommandError(f"model directory not found at {model_path}")
    from qwen_tts import Qwen3TTSModel

    try:
        Qwen3TTSModel.from_pretrained(str(model_path), device_map=device)
    except (OSError, TypeError, ValueError) as error:
        raise CommandError(f"cannot load {model} from {model_path}: {error}") from error
