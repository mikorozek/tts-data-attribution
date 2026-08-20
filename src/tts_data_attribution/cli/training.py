from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from peft import LoraConfig, PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..dataset import UtteranceDataset
from ..experiment import ExperimentManifest, Plan, TrainingConfig
from ..models import (
    apply_lora,
    is_lora_checkpoint_complete,
    save_lora_checkpoint,
    train,
)
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
        "configure", help="configure training for an experiment"
    )
    configure_parser.add_argument("experiment")
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
        "start", help="train experiment checkpoints"
    )
    start_parser.add_argument("experiment_name", metavar="experiment-name")
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
    selection.add_argument(
        "--all-training-sets",
        action="store_true",
        help="train the training pool and every subset",
    )
    start_parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    start_parser.set_defaults(run=run_start)


def run_configure(arguments: argparse.Namespace) -> None:
    directory = Path("experiments") / arguments.experiment
    if (
        not (directory / "manifest.yaml").is_file()
        or not (directory / "plan.json").is_file()
    ):
        raise CommandError(
            f"experiment {arguments.experiment} is incomplete; "
            "run experiment init first"
        )
    path = directory / "training.yaml"
    if path.exists():
        raise CommandError(f"training is already configured at {path}")

    try:
        config = TrainingConfig(
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

    config.to_yaml(path)
    print(f"training configured at {path}")


def run_start(arguments: argparse.Namespace) -> None:
    directory = Path("experiments") / arguments.experiment_name
    paths = {
        "manifest": directory / "manifest.yaml",
        "plan": directory / "plan.json",
        "training config": directory / "training.yaml",
        "encoded utterances": directory / "sampled_utterances_encoded.jsonl",
        "speaker embeddings": directory / "speaker_embeddings.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise CommandError(f"experiment is missing: {', '.join(missing)}")

    try:
        manifest = ExperimentManifest.from_yaml(paths["manifest"])
        plan = Plan.from_json(paths["plan"])
        config = TrainingConfig.from_yaml(paths["training config"])
        encoded = UtteranceDataset.from_jsonl(paths["encoded utterances"])
        speaker_embeddings = torch.load(
            paths["speaker embeddings"],
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as error:
        raise CommandError(f"cannot read experiment: {error}") from error

    if manifest.model != "qwen3-tts":
        raise CommandError(f"training is not implemented for model {manifest.model}")
    if not isinstance(speaker_embeddings, dict) or not all(
        isinstance(name, str) and isinstance(embedding, torch.Tensor)
        for name, embedding in speaker_embeddings.items()
    ):
        raise CommandError("speaker embeddings must map speaker names to tensors")

    if arguments.training_pool:
        training_sets = {"training-pool": plan.training_pool}
    elif arguments.subset is not None:
        if arguments.subset not in plan.subsets:
            raise CommandError(f"unknown training subset: {arguments.subset}")
        training_sets = {arguments.subset: plan.subsets[arguments.subset]}
    else:
        training_sets = {"training-pool": plan.training_pool, **plan.subsets}

    required_ids = set(plan.validation_pool)
    for identifiers in training_sets.values():
        required_ids.update(identifiers)
    missing_ids = sorted(required_ids - encoded.ids())
    if missing_ids:
        raise CommandError(f"encoded utterances missing for: {missing_ids}")
    missing_speakers = sorted(
        {utterance.speaker for utterance in encoded} - set(speaker_embeddings)
    )
    if missing_speakers:
        raise CommandError(f"speaker embeddings missing for: {missing_speakers}")

    validation_dataset = UtteranceDataset(
        encoded.get_utterances_by_ids(plan.validation_pool)
    )

    try:
        device = torch.device(arguments.device)
    except RuntimeError as error:
        raise CommandError(f"invalid training device: {arguments.device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA is not available")

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda utterances: collate(utterances, speaker_embeddings),
    )
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]

    for name, identifiers in training_sets.items():
        if name == "training-pool":
            checkpoint = directory / "checkpoints" / "training-pool"
        else:
            checkpoint = directory / "checkpoints" / "subsets" / name

        if arguments.all_training_sets and is_lora_checkpoint_complete(checkpoint):
            print(f"checkpoint already complete at {checkpoint}")
            continue
        if checkpoint.exists():
            raise CommandError(f"checkpoint already exists at {checkpoint}")

        training_dataset = UtteranceDataset(encoded.get_utterances_by_ids(identifiers))
        training_loader = DataLoader(
            training_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
            collate_fn=lambda utterances: collate(utterances, speaker_embeddings),
        )

        model = load_model(
            manifest.model_path,
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
            print(
                json.dumps(
                    {"training_data": name, **metrics},
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
            checkpoint,
            cast(PeftModel, model.talker),
            optimizer,
            epoch=config.epochs,
            step=int(history[-1]["step"]),
        )
        print(f"checkpoint saved at {checkpoint}")

        del optimizer
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
