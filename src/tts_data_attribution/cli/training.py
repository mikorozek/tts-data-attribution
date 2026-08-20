from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from peft import LoraConfig, PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..dataset import UtteranceDataset
from ..experiment import ExperimentManifest, Plan, TrainingRunManifest
from ..models import apply_lora, save_lora_checkpoint, train
from ..models.qwen3_tts import (
    LORA_TARGET_MODULES,
    collate,
    load_model,
    objective,
)
from .errors import CommandError


def register(subparsers: argparse._SubParsersAction) -> None:
    training_parser = subparsers.add_parser("training", help="model training")
    training_subparsers = training_parser.add_subparsers(required=True)

    configure_parser = training_subparsers.add_parser(
        "configure", help="configure a training run"
    )
    configure_parser.add_argument("experiment_name", metavar="experiment-name")
    selection = configure_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--training-pool",
        action="store_true",
        help="train the complete training pool",
    )
    selection.add_argument(
        "--subset",
        metavar="subset-id",
        help="train one named subset",
    )
    configure_parser.add_argument(
        "--dtype", choices=["bfloat16", "float32"], default="bfloat16"
    )
    configure_parser.add_argument("--lora-rank", type=int, required=True)
    configure_parser.add_argument("--lora-alpha", type=int, required=True)
    configure_parser.add_argument("--lora-dropout", type=float, default=0.0)
    configure_parser.add_argument("--learning-rate", type=float, required=True)
    configure_parser.add_argument("--adam-beta1", type=float, default=0.9)
    configure_parser.add_argument("--adam-beta2", type=float, default=0.999)
    configure_parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    configure_parser.add_argument("--weight-decay", type=float, default=0.01)
    configure_parser.add_argument("--epochs", type=int, required=True)
    configure_parser.add_argument("--batch-size", type=int, required=True)
    configure_parser.add_argument("--seed", type=int, required=True)
    configure_parser.set_defaults(run=run_configure)

    start_parser = training_subparsers.add_parser(
        "start", help="start a configured training run"
    )
    start_parser.add_argument("experiment_name", metavar="experiment-name")
    start_parser.add_argument("training_run_name", metavar="training-run-name")
    start_parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    start_parser.set_defaults(run=run_start)


def run_configure(arguments: argparse.Namespace) -> None:
    experiment_directory = Path("experiments") / arguments.experiment_name
    manifest_path = experiment_directory / "manifest.yaml"
    plan_path = experiment_directory / "plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        raise CommandError(
            f"experiment {arguments.experiment_name} is incomplete; "
            "run experiment init first"
        )

    try:
        plan = Plan.from_json(plan_path)
    except (OSError, TypeError, ValueError) as error:
        raise CommandError(f"cannot read experiment plan: {error}") from error

    if arguments.training_pool:
        training_set = "training-pool"
    else:
        training_set = arguments.subset
        if training_set not in plan.subsets:
            raise CommandError(f"unknown training subset: {training_set}")

    try:
        manifest = TrainingRunManifest(
            training_set=training_set,
            dtype=arguments.dtype,
            lora_rank=arguments.lora_rank,
            lora_alpha=arguments.lora_alpha,
            lora_dropout=arguments.lora_dropout,
            learning_rate=arguments.learning_rate,
            adam_betas=(arguments.adam_beta1, arguments.adam_beta2),
            adam_epsilon=arguments.adam_epsilon,
            weight_decay=arguments.weight_decay,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
        )
    except ValueError as error:
        raise CommandError(str(error)) from error

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    training_run_name = f"{training_set}-{timestamp}"
    training_run_directory = experiment_directory / "training-runs" / training_run_name
    try:
        training_run_directory.mkdir(parents=True)
        manifest.to_yaml(training_run_directory / "manifest.yaml")
    except OSError as error:
        raise CommandError(f"cannot create training run: {error}") from error
    print(f"training run {training_run_name} configured at {training_run_directory}")


def run_start(arguments: argparse.Namespace) -> None:
    if Path(arguments.training_run_name).name != arguments.training_run_name:
        raise CommandError("training run name must be a single path component")
    experiment_directory = Path("experiments") / arguments.experiment_name
    training_run_directory = (
        experiment_directory / "training-runs" / arguments.training_run_name
    )
    paths = {
        "manifest": experiment_directory / "manifest.yaml",
        "plan": experiment_directory / "plan.json",
        "training run manifest": training_run_directory / "manifest.yaml",
        "encoded utterances": experiment_directory / "sampled_utterances_encoded.jsonl",
        "speaker embeddings": experiment_directory / "speaker_embeddings.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise CommandError(f"experiment is missing: {', '.join(missing)}")

    target = training_run_directory / "target"
    metrics_path = training_run_directory / "metrics.jsonl"
    if target.exists() or metrics_path.exists():
        raise CommandError(
            f"training run {arguments.training_run_name} has already started"
        )

    try:
        experiment = ExperimentManifest.from_yaml(paths["manifest"])
        plan = Plan.from_json(paths["plan"])
        config = TrainingRunManifest.from_yaml(paths["training run manifest"])
        encoded = UtteranceDataset.from_jsonl(paths["encoded utterances"])
        speaker_embeddings = torch.load(
            paths["speaker embeddings"],
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as error:
        raise CommandError(f"cannot read experiment: {error}") from error

    if experiment.model != "qwen3-tts":
        raise CommandError(f"training is not implemented for model {experiment.model}")
    if not isinstance(speaker_embeddings, dict) or not all(
        isinstance(name, str) and isinstance(embedding, torch.Tensor)
        for name, embedding in speaker_embeddings.items()
    ):
        raise CommandError("speaker embeddings must map speaker names to tensors")

    if config.training_set == "training-pool":
        training_ids = plan.training_pool
    elif config.training_set in plan.subsets:
        training_ids = plan.subsets[config.training_set]
    else:
        raise CommandError(f"unknown training set: {config.training_set}")

    required_ids = set(plan.validation_pool) | set(training_ids)
    missing_ids = sorted(required_ids - encoded.ids())
    if missing_ids:
        raise CommandError(f"encoded utterances missing for: {missing_ids}")
    missing_speakers = sorted(
        {utterance.speaker for utterance in encoded} - set(speaker_embeddings)
    )
    if missing_speakers:
        raise CommandError(f"speaker embeddings missing for: {missing_speakers}")

    training_dataset = UtteranceDataset(encoded.get_utterances_by_ids(training_ids))
    validation_dataset = UtteranceDataset(
        encoded.get_utterances_by_ids(plan.validation_pool)
    )

    try:
        device = torch.device(arguments.device)
    except RuntimeError as error:
        raise CommandError(f"invalid training device: {arguments.device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA is not available")

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=lambda utterances: collate(utterances, speaker_embeddings),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda utterances: collate(utterances, speaker_embeddings),
    )
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]

    model = load_model(
        experiment.model_path,
        device=device,
        dtype=dtype,
    )
    model.talker = cast(
        Any,
        apply_lora(
            model.talker,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules=list(LORA_TARGET_MODULES),
            ),
            seed=config.seed,
        ),
    )
    optimizer = AdamW(
        (
            parameter
            for parameter in model.talker.parameters()
            if parameter.requires_grad
        ),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    def report_epoch(metrics: dict[str, int | float]) -> None:
        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            json.dumps(
                {"training_run": arguments.training_run_name, **metrics},
                sort_keys=True,
            ),
            flush=True,
        )

    history = train(
        model,
        training_loader,
        validation_loader,
        optimizer,
        objective,
        config.epochs,
        device,
        report_epoch,
    )
    save_lora_checkpoint(
        target,
        cast(PeftModel, model.talker),
        optimizer,
        epoch=config.epochs,
        step=int(history[-1]["step"]),
    )
    print(f"training target saved at {target}")
