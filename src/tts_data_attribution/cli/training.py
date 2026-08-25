from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from peft import LoraConfig, PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..dataset import UtteranceDataset
from ..experiment import (
    ExperimentManifest,
    Plan,
    TrainingParameter,
    TrainingRunManifest,
)
from ..models import (
    apply_lora,
    lora_parameter_layout,
    save_lora_checkpoint,
    train,
)
from ..models.qwen3_tts import (
    LORA_TARGET_MODULES,
    collate,
    load_model,
    objective,
)
from .errors import CommandError, require_flat_name


def register(subparsers: argparse._SubParsersAction) -> None:
    training_parser = subparsers.add_parser("training", help="model training")
    training_subparsers = training_parser.add_subparsers(required=True)

    configure_parser = training_subparsers.add_parser(
        "configure", help="configure a training run"
    )
    configure_parser.add_argument("experiment_name", metavar="experiment-name")
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
        "start", help="train one model in a configured training run"
    )
    start_parser.add_argument("experiment_name", metavar="experiment-name")
    start_parser.add_argument("training_run_name", metavar="training-run-name")
    selection = start_parser.add_mutually_exclusive_group(required=True)
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
    start_parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    start_parser.set_defaults(run=run_start)


def run_configure(arguments: argparse.Namespace) -> None:
    require_flat_name(arguments.experiment_name, "experiment name")
    experiment_directory = Path("experiments") / arguments.experiment_name
    manifest_path = experiment_directory / "manifest.yaml"
    plan_path = experiment_directory / "plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        raise CommandError(
            f"experiment {arguments.experiment_name} is incomplete; "
            "run experiment init first"
        )

    try:
        manifest = TrainingRunManifest(
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
        Plan.from_json(plan_path)
    except (OSError, TypeError, ValueError) as error:
        raise CommandError(f"cannot configure training run: {error}") from error

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    training_run_name = f"run-{timestamp}"
    training_run_directory = experiment_directory / "training-runs" / training_run_name
    try:
        training_run_directory.mkdir(parents=True)
        manifest.to_yaml(training_run_directory / "manifest.yaml")
    except OSError as error:
        raise CommandError(f"cannot create training run: {error}") from error
    print(f"training run {training_run_name} configured at {training_run_directory}")


def _write_resolved_manifest(
    path: Path,
    config: TrainingRunManifest,
    model: PeftModel,
    optimizer: AdamW,
) -> TrainingRunManifest:
    parameters, parameter_groups = lora_parameter_layout(model, optimizer)
    resolved = replace(
        config,
        parameters=tuple(
            TrainingParameter(
                name=cast(str, parameter["name"]),
                shape=tuple(cast(list[int], parameter["shape"])),
                dtype=cast(str, parameter["dtype"]),
            )
            for parameter in parameters
        ),
        parameter_groups=tuple(tuple(group) for group in parameter_groups),
    )
    if config.parameters:
        if config != resolved:
            raise ValueError("model parameter layout differs from the training run")
        return config
    temporary_path = path.with_name(f".{path.name}.tmp")
    resolved.to_yaml(temporary_path)
    temporary_path.replace(path)
    return resolved


def run_start(arguments: argparse.Namespace) -> None:
    require_flat_name(arguments.experiment_name, "experiment name")
    require_flat_name(arguments.training_run_name, "training run name")
    experiment_directory = Path("experiments") / arguments.experiment_name
    training_run_directory = (
        experiment_directory / "training-runs" / arguments.training_run_name
    )
    paths = {
        "experiment manifest": experiment_directory / "manifest.yaml",
        "plan": experiment_directory / "plan.json",
        "training run manifest": training_run_directory / "manifest.yaml",
        "encoded utterances": experiment_directory / "sampled_utterances_encoded.jsonl",
        "speaker embeddings": experiment_directory / "speaker_embeddings.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise CommandError(f"experiment is missing: {', '.join(missing)}")

    try:
        experiment = ExperimentManifest.from_yaml(paths["experiment manifest"])
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

    if arguments.training_pool:
        training_ids = plan.training_pool
        model_name = "training-pool"
        model_directory = training_run_directory / "training-pool"
    else:
        subset_id = arguments.subset
        if subset_id not in plan.subsets:
            raise CommandError(f"unknown training subset: {subset_id}")
        training_ids = plan.subsets[subset_id]
        model_name = subset_id
        model_directory = training_run_directory / "subsets" / subset_id
    if model_directory.exists():
        raise CommandError(f"training model {model_name} has already started")
    if not arguments.training_pool and not config.parameters:
        raise CommandError("the training-pool model must be started before subsets")

    if experiment.model != "qwen3-tts":
        raise CommandError(f"training is not implemented for model {experiment.model}")
    if not isinstance(speaker_embeddings, dict) or not all(
        isinstance(name, str) and isinstance(embedding, torch.Tensor)
        for name, embedding in speaker_embeddings.items()
    ):
        raise CommandError("speaker embeddings must map speaker names to tensors")

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
    training_evaluation_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda utterances: collate(utterances, speaker_embeddings),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda utterances: collate(utterances, speaker_embeddings),
    )
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]

    try:
        model = load_model(experiment.model_path, device=device, dtype=dtype)
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
        config = _write_resolved_manifest(
            paths["training run manifest"],
            config,
            cast(PeftModel, model.talker),
            optimizer,
        )
    except (OSError, TypeError, ValueError) as error:
        raise CommandError(f"cannot prepare training model: {error}") from error

    try:
        model_directory.mkdir(parents=True)
    except OSError as error:
        raise CommandError(
            f"cannot create training model directory: {error}"
        ) from error
    metrics_path = model_directory / "metrics.jsonl"

    def report_epoch(metrics: dict[str, int | float]) -> None:
        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "training_run": arguments.training_run_name,
                    "model": model_name,
                    **metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        epoch = int(metrics["epoch"])
        if epoch == 0:
            return
        checkpoint = model_directory / "checkpoints" / f"epoch-{epoch:04d}"
        save_lora_checkpoint(
            checkpoint,
            cast(PeftModel, model.talker),
            optimizer,
            metrics,
        )

    train(
        model,
        training_loader,
        training_evaluation_loader,
        validation_loader,
        optimizer,
        objective,
        config.epochs,
        device,
        report_epoch,
    )
    print(f"training model saved at {model_directory}")
