from __future__ import annotations

import argparse
from pathlib import Path

from ..experiment import TrainingConfig
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
    configure_parser.add_argument("--root", type=Path, default=Path("experiments"))
    configure_parser.set_defaults(run=run_configure)


def run_configure(arguments: argparse.Namespace) -> None:
    directory = arguments.root / arguments.experiment
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
