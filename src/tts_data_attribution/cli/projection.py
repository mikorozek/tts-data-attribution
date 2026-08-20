from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..dataset import UtteranceDataset
from ..experiment import ExperimentManifest, Plan, TrainingRunManifest
from ..models import is_lora_checkpoint_complete
from ..models.qwen3_tts import collate, load_model, objective
from ..trackstar import (
    TwoSidedRandomProjection,
    collect_per_example_gradients,
    correct_gradients_with_adamw,
)
from .errors import CommandError


def register(subparsers: argparse._SubParsersAction) -> None:
    projection_parser = subparsers.add_parser(
        "projection", help="random gradient projections"
    )
    projection_subparsers = projection_parser.add_subparsers(required=True)

    init_parser = projection_subparsers.add_parser(
        "init", help="initialize a random projection"
    )
    init_parser.add_argument("experiment_name", metavar="experiment-name")
    init_parser.add_argument("projection_name", metavar="projection-name")
    init_parser.add_argument(
        "--training-run",
        required=True,
        metavar="training-run-name",
    )
    init_parser.add_argument("--output-dimension", type=int, required=True)
    init_parser.add_argument("--seed", type=int, required=True)
    init_parser.set_defaults(run=run_init)

    apply_parser = projection_subparsers.add_parser(
        "apply", help="apply a projection to per-example gradients"
    )
    apply_parser.add_argument("experiment_name", metavar="experiment-name")
    apply_parser.add_argument("projection_name", metavar="projection-name")
    selection = apply_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--training-pool",
        action="store_true",
        help="project the complete training pool",
    )
    apply_parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    apply_parser.set_defaults(run=run_apply)


def _require_flat_name(value: str, name: str) -> None:
    if not value or Path(value).name != value:
        raise CommandError(f"{name} must be a single path component")


def run_init(arguments: argparse.Namespace) -> None:
    _require_flat_name(arguments.training_run, "training run name")
    _require_flat_name(arguments.projection_name, "projection name")
    if arguments.seed < 0:
        raise CommandError("projection seed must not be negative")

    experiment_directory = Path("experiments") / arguments.experiment_name
    training_run_directory = (
        experiment_directory / "training-runs" / arguments.training_run
    )
    target = training_run_directory / "target"
    if not (experiment_directory / "manifest.yaml").is_file():
        raise CommandError(f"experiment {arguments.experiment_name} does not exist")
    training_run_manifest_path = training_run_directory / "manifest.yaml"
    if not training_run_manifest_path.is_file():
        raise CommandError(f"training run {arguments.training_run} does not exist")
    try:
        training_run = TrainingRunManifest.from_yaml(training_run_manifest_path)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise CommandError(f"cannot read training run: {error}") from error
    if training_run.training_set != "training-pool":
        raise CommandError("projection requires a training-pool run")
    if not is_lora_checkpoint_complete(target):
        raise CommandError(f"training target is incomplete at {target}")

    projection_directory = (
        experiment_directory
        / "trackstar"
        / "projections"
        / arguments.projection_name
    )
    if projection_directory.exists():
        raise CommandError(f"projection already exists at {projection_directory}")

    try:
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        if metadata["format_version"] != 1:
            raise ValueError("unsupported training target metadata")
        parameters = metadata["parameters"]
        projector = TwoSidedRandomProjection(
            tuple(parameter["shape"] for parameter in parameters),
            arguments.output_dimension,
            seed=arguments.seed,
            device="cpu",
        )
        projection_parameters = [
            {"name": parameter["name"], "shape": parameter["shape"]}
            for parameter in parameters
        ]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CommandError(f"cannot initialize projection: {error}") from error

    manifest = {
        "type": "two-sided",
        "training_run": arguments.training_run,
        "output_dimension": projector.output_dimension,
        "seed": projector.seed,
        "parameters": projection_parameters,
    }
    try:
        projection_directory.mkdir(parents=True)
        (projection_directory / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        torch.save(
            {
                "left_matrices": projector.left_matrices,
                "right_matrices": projector.right_matrices,
            },
            projection_directory / "matrices.pt",
        )
    except (OSError, RuntimeError, yaml.YAMLError) as error:
        raise CommandError(f"cannot save projection: {error}") from error
    print(f"projection {arguments.projection_name} created at {projection_directory}")


def run_apply(arguments: argparse.Namespace) -> None:
    _require_flat_name(arguments.projection_name, "projection name")
    experiment_directory = Path("experiments") / arguments.experiment_name
    projection_directory = (
        experiment_directory
        / "trackstar"
        / "projections"
        / arguments.projection_name
    )
    projection_manifest_path = projection_directory / "manifest.yaml"
    matrices_path = projection_directory / "matrices.pt"
    output_path = projection_directory / "projected" / "training-pool.pt"
    if not projection_manifest_path.is_file() or not matrices_path.is_file():
        raise CommandError(f"projection {arguments.projection_name} is incomplete")
    if output_path.exists():
        raise CommandError(f"projected training pool already exists at {output_path}")

    try:
        projection_manifest = yaml.safe_load(
            projection_manifest_path.read_text(encoding="utf-8")
        )
        if projection_manifest["type"] != "two-sided":
            raise ValueError("unsupported projection type")
        training_run_name = projection_manifest["training_run"]
        if not isinstance(training_run_name, str):
            raise TypeError("projection training run must be a string")
        _require_flat_name(training_run_name, "training run name")
        projection_parameters = projection_manifest["parameters"]
        matrices = torch.load(matrices_path, map_location="cpu", weights_only=True)
    except CommandError:
        raise
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise CommandError(f"cannot read projection: {error}") from error

    training_run_directory = (
        experiment_directory / "training-runs" / training_run_name
    )
    target = training_run_directory / "target"
    paths = {
        "experiment manifest": experiment_directory / "manifest.yaml",
        "plan": experiment_directory / "plan.json",
        "training run manifest": training_run_directory / "manifest.yaml",
        "encoded utterances": experiment_directory
        / "sampled_utterances_encoded.jsonl",
        "speaker embeddings": experiment_directory / "speaker_embeddings.pt",
        "training target metadata": target / "metadata.json",
        "optimizer": target / "optimizer.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing or not is_lora_checkpoint_complete(target):
        details = ", ".join(missing) if missing else str(target)
        raise CommandError(f"projection input is incomplete: {details}")

    try:
        device = torch.device(arguments.device)
    except RuntimeError as error:
        raise CommandError(f"invalid projection device: {arguments.device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA is not available")

    try:
        experiment = ExperimentManifest.from_yaml(paths["experiment manifest"])
        plan = Plan.from_json(paths["plan"])
        training_run = TrainingRunManifest.from_yaml(paths["training run manifest"])
        if training_run.training_set != "training-pool":
            raise ValueError("projection requires a training-pool run")
        encoded = UtteranceDataset.from_jsonl(paths["encoded utterances"])
        speaker_embeddings = torch.load(
            paths["speaker embeddings"],
            map_location="cpu",
            weights_only=True,
        )
        target_metadata = json.loads(
            paths["training target metadata"].read_text(encoding="utf-8")
        )
        projector = TwoSidedRandomProjection.from_matrices(
            matrices["left_matrices"],
            matrices["right_matrices"],
            seed=projection_manifest["seed"],
            device=device,
        )
        if projector.output_dimension != projection_manifest["output_dimension"]:
            raise ValueError("projection output dimension differs from its manifest")
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        raise CommandError(f"cannot load projection inputs: {error}") from error

    if experiment.model != "qwen3-tts":
        raise CommandError(
            f"projection is not implemented for model {experiment.model}"
        )
    if not isinstance(speaker_embeddings, dict) or not all(
        isinstance(name, str) and isinstance(embedding, torch.Tensor)
        for name, embedding in speaker_embeddings.items()
    ):
        raise CommandError("speaker embeddings must map speaker names to tensors")
    if not plan.training_pool:
        raise CommandError("training pool must not be empty")
    missing_ids = sorted(set(plan.training_pool) - encoded.ids())
    if missing_ids:
        raise CommandError(f"encoded utterances missing for: {missing_ids}")
    training_utterances = encoded.get_utterances_by_ids(plan.training_pool)
    missing_speakers = sorted(
        {utterance.speaker for utterance in training_utterances}
        - set(speaker_embeddings)
    )
    if missing_speakers:
        raise CommandError(f"speaker embeddings missing for: {missing_speakers}")

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[
        training_run.dtype
    ]
    try:
        model = load_model(experiment.model_path, device=device, dtype=dtype)
        model.talker = cast(
            Any,
            PeftModel.from_pretrained(
                model.talker,
                target / "adapter",
                is_trainable=True,
            ),
        )
        trainable_parameters = tuple(
            (name, parameter)
            for name, parameter in model.talker.named_parameters()
            if parameter.requires_grad
        )
        actual_layout = [
            {"name": name, "shape": list(parameter.shape)}
            for name, parameter in trainable_parameters
        ]
        target_layout = [
            {"name": parameter["name"], "shape": parameter["shape"]}
            for parameter in target_metadata["parameters"]
        ]
        if actual_layout != target_layout or actual_layout != projection_parameters:
            raise ValueError("trainable parameter layout differs from the projection")
        parameter_names = [name for name, _ in trainable_parameters]
        target_parameter_names = [
            name
            for group in target_metadata["parameter_groups"]
            for name in group
        ]
        if parameter_names != target_parameter_names:
            raise ValueError("optimizer parameter ordering differs from the model")

        optimizer = AdamW(
            (parameter for _, parameter in trainable_parameters),
            lr=training_run.learning_rate,
            betas=training_run.adam_betas,
            eps=training_run.adam_epsilon,
            weight_decay=training_run.weight_decay,
        )
        optimizer.load_state_dict(
            torch.load(paths["optimizer"], map_location="cpu", weights_only=True)
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise CommandError(f"cannot load training target: {error}") from error

    dataset = UtteranceDataset(training_utterances)
    data_loader = DataLoader(
        dataset,
        batch_size=training_run.batch_size,
        shuffle=False,
        collate_fn=lambda utterances: collate(utterances, speaker_embeddings),
    )
    model.eval()
    projected_gradients = []
    completed = 0
    try:
        for batch in data_loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            losses = objective(model, batch)
            for gradients in collect_per_example_gradients(model.talker, losses):
                corrected = correct_gradients_with_adamw(
                    model.talker,
                    optimizer,
                    gradients,
                )
                projected_gradients.append(projector(corrected).cpu())
            completed += losses.shape[0]
            print(
                f"projected {completed}/{len(dataset)} training examples",
                flush=True,
            )
    except (RuntimeError, TypeError, ValueError) as error:
        raise CommandError(f"cannot project training gradients: {error}") from error

    if len(projected_gradients) != len(dataset):
        raise CommandError("projected gradient count differs from the training pool")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ids": list(plan.training_pool),
                "projected_gradients": torch.stack(projected_gradients),
            },
            output_path,
        )
    except (OSError, RuntimeError) as error:
        raise CommandError(f"cannot save projected gradients: {error}") from error
    print(f"projected training gradients saved at {output_path}")
