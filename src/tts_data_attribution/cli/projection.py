from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..dataset import UtteranceDataset
from ..evaluation import save_immutable_torch_artifact
from ..experiment import ExperimentManifest, Plan, TrainingRunManifest
from ..models import is_lora_checkpoint_complete, load_checkpoint_metrics
from ..models.qwen3_tts import collate, load_model, objective
from ..trackstar import (
    TwoSidedRandomProjection,
    collect_per_example_gradients,
    correct_gradients_with_adamw,
)
from .errors import CommandError, require_flat_name


def register(subparsers: argparse._SubParsersAction) -> None:
    projection_parser = subparsers.add_parser(
        "projection", help="random gradient projections"
    )
    projection_subparsers = projection_parser.add_subparsers(required=True)

    init_parser = projection_subparsers.add_parser(
        "init", help="initialize a random projection"
    )
    init_parser.add_argument("experiment_name", metavar="experiment-name")
    init_parser.add_argument("training_run_name", metavar="training-run-name")
    init_parser.add_argument("projection_name", metavar="projection-name")
    init_parser.add_argument(
        "--training-checkpoint",
        required=True,
        metavar="checkpoint-path",
        help="training-pool checkpoint path relative to the training run",
    )
    init_parser.add_argument("--output-dimension", type=int, required=True)
    init_parser.add_argument("--seed", type=int, required=True)
    init_parser.set_defaults(run=run_init)

    apply_parser = projection_subparsers.add_parser(
        "apply", help="apply a projection to per-example gradients"
    )
    apply_parser.add_argument("experiment_name", metavar="experiment-name")
    apply_parser.add_argument("training_run_name", metavar="training-run-name")
    apply_parser.add_argument("projection_name", metavar="projection-name")
    selection = apply_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--training-pool",
        action="store_true",
        help="project the complete training pool",
    )
    selection.add_argument(
        "--query-pool",
        action="store_true",
        help="project the experiment query pool",
    )
    apply_parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    apply_parser.set_defaults(run=run_apply)


@dataclass(frozen=True)
class ProjectedPools:
    training: torch.Tensor
    queries: torch.Tensor


def load_projected_pools(
    experiment_directory: Path,
    projection_directory: Path,
    device: torch.device | str,
) -> ProjectedPools:
    plan_path = experiment_directory / "plan.json"
    manifest_path = projection_directory / "manifest.yaml"
    training_path = projection_directory / "projected" / "training-pool.pt"
    query_path = projection_directory / "projected" / "query-pool.pt"
    missing = [
        name
        for name, path in [
            ("plan", plan_path),
            ("projection manifest", manifest_path),
            ("projected training pool", training_path),
            ("projected query pool", query_path),
        ]
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"projected pool input is missing: {', '.join(missing)}")

    plan = Plan.from_json(plan_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    training_artifact = torch.load(
        training_path,
        map_location=device,
        weights_only=True,
    )
    query_artifact = torch.load(
        query_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(manifest, dict) or manifest.get("type") != "two-sided":
        raise ValueError("projection manifest is invalid")
    output_dimension = manifest.get("output_dimension")
    if type(output_dimension) is not int or output_dimension < 1:
        raise ValueError("projection output dimension must be a positive integer")
    if not isinstance(training_artifact, dict) or not isinstance(query_artifact, dict):
        raise ValueError("projected pool artifacts must be mappings")
    if training_artifact.get("ids") != plan.training_pool:
        raise ValueError("projected training IDs differ from the experiment plan")
    if query_artifact.get("ids") != plan.query_pool:
        raise ValueError("projected query IDs differ from the experiment plan")

    training = training_artifact.get("projected_gradients")
    queries = query_artifact.get("projected_gradients")
    if (
        not isinstance(training, torch.Tensor)
        or not isinstance(queries, torch.Tensor)
        or not training.is_floating_point()
        or not queries.is_floating_point()
        or training.shape != (len(plan.training_pool), output_dimension)
        or queries.shape != (len(plan.query_pool), output_dimension)
    ):
        raise ValueError("projected gradient dimensions or dtype are invalid")
    if not torch.isfinite(training).all() or not torch.isfinite(queries).all():
        raise ValueError("projected gradients must be finite")
    return ProjectedPools(training=training, queries=queries)


def resolve_training_checkpoint(
    training_run_directory: Path,
    source: dict[str, object],
) -> tuple[Path, TrainingRunManifest, int, int]:
    value = source.get("training_checkpoint")
    if not isinstance(value, str) or not value:
        raise CommandError("projection training checkpoint is missing")
    relative_path = Path(value)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or len(relative_path.parts) != 3
        or relative_path.parts[:2] != ("training-pool", "checkpoints")
    ):
        raise CommandError(
            "training checkpoint must identify training-pool/checkpoints/epoch-NNNN"
        )
    epoch_name = relative_path.parts[2]
    if not epoch_name.startswith("epoch-"):
        raise CommandError("training checkpoint epoch name is invalid")
    try:
        checkpoint_epoch = int(epoch_name.removeprefix("epoch-"))
    except ValueError as error:
        raise CommandError("training checkpoint epoch name is invalid") from error
    if checkpoint_epoch < 1 or epoch_name != f"epoch-{checkpoint_epoch:04d}":
        raise CommandError("training checkpoint epoch name is invalid")

    checkpoint = training_run_directory / relative_path
    manifest_path = training_run_directory / "manifest.yaml"
    if not manifest_path.is_file():
        raise CommandError(f"training run manifest is missing at {manifest_path}")
    if not is_lora_checkpoint_complete(checkpoint):
        raise CommandError(f"training checkpoint is incomplete at {checkpoint}")
    try:
        training_run = TrainingRunManifest.from_yaml(manifest_path)
        metrics = load_checkpoint_metrics(
            checkpoint / "metrics.json",
            expected_epoch=checkpoint_epoch,
            maximum_epoch=training_run.epochs,
        )
        if not training_run.parameters:
            raise ValueError("training run parameter layout is missing")
    except (
        OSError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise CommandError(f"cannot read training checkpoint: {error}") from error
    return checkpoint, training_run, checkpoint_epoch, int(metrics["step"])


def _validate_adamw_state_dict(
    state_dict: object,
    training_run: TrainingRunManifest,
    expected_step: int,
) -> None:
    if not isinstance(state_dict, dict) or set(state_dict) != {"state", "param_groups"}:
        raise ValueError("AdamW checkpoint must contain state and parameter groups")
    states = state_dict["state"]
    serialized_groups = state_dict["param_groups"]
    if not isinstance(states, dict) or not isinstance(serialized_groups, list):
        raise ValueError("AdamW checkpoint state or parameter groups are invalid")
    if len(serialized_groups) != len(training_run.parameter_groups):
        raise ValueError("AdamW parameter group count differs from the training run")

    expected_options = {
        "lr": training_run.learning_rate,
        "betas": training_run.adam_betas,
        "eps": training_run.adam_epsilon,
        "weight_decay": training_run.weight_decay,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    expected_indices = []
    next_index = 0
    for group, parameter_names in zip(
        serialized_groups,
        training_run.parameter_groups,
        strict=True,
    ):
        if not isinstance(group, dict):
            raise ValueError("AdamW parameter group is invalid")
        indices = list(range(next_index, next_index + len(parameter_names)))
        if group.get("params") != indices:
            raise ValueError("AdamW parameter order differs from the training run")
        for name, expected in expected_options.items():
            actual = group.get(name)
            if name == "betas" and isinstance(actual, list | tuple):
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(f"AdamW {name} differs from the training run")
        expected_indices.extend(indices)
        next_index += len(parameter_names)
    if set(states) != set(expected_indices):
        raise ValueError("AdamW state parameters differ from the training run")

    for index, parameter in enumerate(training_run.parameters):
        state = states[index]
        if not isinstance(state, dict):
            raise ValueError(f"AdamW state is invalid for parameter {index}")
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError(f"AdamW state step is invalid for parameter {index}")
            step = step.item()
        if (
            isinstance(step, bool)
            or not isinstance(step, int | float)
            or not math.isfinite(float(step))
            or not float(step).is_integer()
            or int(step) != expected_step
        ):
            raise ValueError(
                "AdamW state step differs from checkpoint metrics for "
                f"parameter {index}"
            )
        for moment_name in ["exp_avg", "exp_avg_sq"]:
            moment = state.get(moment_name)
            if (
                not isinstance(moment, torch.Tensor)
                or not moment.is_floating_point()
                or moment.shape != parameter.shape
                or not torch.isfinite(moment).all()
            ):
                raise ValueError(
                    f"AdamW {moment_name} is invalid for parameter {index}"
                )


def _load_adamw_optimizer(
    trainable_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    training_run: TrainingRunManifest,
    state_path: Path,
    expected_step: int,
) -> AdamW:
    parameters_by_name = dict(trainable_parameters)
    grouped_parameters = [
        [parameters_by_name[name] for name in group]
        for group in training_run.parameter_groups
    ]
    optimizer = AdamW(
        [{"params": parameters} for parameters in grouped_parameters],
        lr=training_run.learning_rate,
        betas=training_run.adam_betas,
        eps=training_run.adam_epsilon,
        weight_decay=training_run.weight_decay,
    )
    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    _validate_adamw_state_dict(state_dict, training_run, expected_step)
    optimizer.load_state_dict(state_dict)
    return optimizer


def run_init(arguments: argparse.Namespace) -> None:
    require_flat_name(arguments.experiment_name, "experiment name")
    require_flat_name(arguments.training_run_name, "training run name")
    require_flat_name(arguments.projection_name, "projection name")
    if arguments.seed < 0:
        raise CommandError("projection seed must not be negative")

    experiment_directory = Path("experiments") / arguments.experiment_name
    training_run_directory = (
        experiment_directory / "training-runs" / arguments.training_run_name
    )
    if not (experiment_directory / "manifest.yaml").is_file():
        raise CommandError(f"experiment {arguments.experiment_name} does not exist")
    source = {"training_checkpoint": arguments.training_checkpoint}
    _, training_run, _, _ = resolve_training_checkpoint(training_run_directory, source)

    projection_directory = (
        training_run_directory / "projections" / arguments.projection_name
    )
    if projection_directory.exists():
        raise CommandError(f"projection already exists at {projection_directory}")

    try:
        projector = TwoSidedRandomProjection(
            tuple(parameter.shape for parameter in training_run.parameters),
            arguments.output_dimension,
            seed=arguments.seed,
            device="cpu",
        )
    except (TypeError, ValueError) as error:
        raise CommandError(f"cannot initialize projection: {error}") from error

    manifest = {
        "type": "two-sided",
        **source,
        "output_dimension": projector.output_dimension,
        "seed": projector.seed,
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
    require_flat_name(arguments.experiment_name, "experiment name")
    require_flat_name(arguments.training_run_name, "training run name")
    require_flat_name(arguments.projection_name, "projection name")
    collection_name = "training-pool" if arguments.training_pool else "query-pool"
    experiment_directory = Path("experiments") / arguments.experiment_name
    training_run_directory = (
        experiment_directory / "training-runs" / arguments.training_run_name
    )
    projection_directory = (
        training_run_directory / "projections" / arguments.projection_name
    )
    projection_manifest_path = projection_directory / "manifest.yaml"
    matrices_path = projection_directory / "matrices.pt"
    output_path = projection_directory / "projected" / f"{collection_name}.pt"
    if not projection_manifest_path.is_file() or not matrices_path.is_file():
        raise CommandError(f"projection {arguments.projection_name} is incomplete")
    if output_path.exists():
        raise CommandError(
            f"projected {collection_name} already exists at {output_path}"
        )

    try:
        projection_manifest = yaml.safe_load(
            projection_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(projection_manifest, dict):
            raise ValueError("projection manifest must be a mapping")
        if projection_manifest.get("type") != "two-sided":
            raise ValueError("unsupported projection type")
        checkpoint, training_run, _, checkpoint_step = resolve_training_checkpoint(
            training_run_directory,
            projection_manifest,
        )
        matrices = torch.load(matrices_path, map_location="cpu", weights_only=True)
    except CommandError:
        raise
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise CommandError(f"cannot read projection: {error}") from error

    paths = {
        "experiment manifest": experiment_directory / "manifest.yaml",
        "plan": experiment_directory / "plan.json",
        "encoded utterances": experiment_directory / "sampled_utterances_encoded.jsonl",
        "speaker embeddings": experiment_directory / "speaker_embeddings.pt",
        "optimizer": checkpoint / "optimizer.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise CommandError(f"projection input is incomplete: {', '.join(missing)}")

    try:
        device = torch.device(arguments.device)
    except RuntimeError as error:
        raise CommandError(f"invalid projection device: {arguments.device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA is not available")

    try:
        experiment = ExperimentManifest.from_yaml(paths["experiment manifest"])
        plan = Plan.from_json(paths["plan"])
        encoded = UtteranceDataset.from_jsonl(paths["encoded utterances"])
        speaker_embeddings = torch.load(
            paths["speaker embeddings"],
            map_location="cpu",
            weights_only=True,
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
    identifiers = plan.training_pool if arguments.training_pool else plan.query_pool
    if not identifiers:
        raise CommandError(f"{collection_name} must not be empty")
    missing_ids = sorted(set(identifiers) - encoded.ids())
    if missing_ids:
        raise CommandError(f"encoded utterances missing for: {missing_ids}")
    utterances = encoded.get_utterances_by_ids(identifiers)
    missing_speakers = sorted(
        {utterance.speaker for utterance in utterances} - set(speaker_embeddings)
    )
    if missing_speakers:
        raise CommandError(f"speaker embeddings missing for: {missing_speakers}")

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[training_run.dtype]
    try:
        model = load_model(experiment.model_path, device=device, dtype=dtype)
        model.talker = cast(
            Any,
            PeftModel.from_pretrained(
                model.talker,
                checkpoint / "adapter",
                is_trainable=True,
            ),
        )
        trainable_parameters = tuple(
            (name, parameter)
            for name, parameter in model.talker.named_parameters()
            if parameter.requires_grad
        )
        actual_layout = [
            (
                name,
                tuple(parameter.shape),
                str(parameter.dtype).removeprefix("torch."),
            )
            for name, parameter in trainable_parameters
        ]
        expected_layout = [
            (parameter.name, parameter.shape, parameter.dtype)
            for parameter in training_run.parameters
        ]
        if actual_layout != expected_layout:
            raise ValueError("trainable parameter layout differs from the training run")
        parameter_names = [name for name, _ in trainable_parameters]
        expected_parameter_names = [
            name for group in training_run.parameter_groups for name in group
        ]
        if parameter_names != expected_parameter_names:
            raise ValueError("optimizer parameter ordering differs from the model")

        optimizer = _load_adamw_optimizer(
            trainable_parameters,
            training_run,
            paths["optimizer"],
            checkpoint_step,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise CommandError(f"cannot load training checkpoint: {error}") from error

    dataset = UtteranceDataset(utterances)
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
                f"projected {completed}/{len(dataset)} {collection_name} examples",
                flush=True,
            )
    except (RuntimeError, TypeError, ValueError) as error:
        raise CommandError(
            f"cannot project {collection_name} gradients: {error}"
        ) from error

    if len(projected_gradients) != len(dataset):
        raise CommandError(
            f"projected gradient count differs from the {collection_name}"
        )

    projected = torch.stack(projected_gradients)
    if not projected.is_floating_point() or not torch.isfinite(projected).all():
        raise CommandError("projected gradients must be finite floating point values")
    try:
        save_immutable_torch_artifact(
            output_path,
            {
                "ids": list(identifiers),
                "projected_gradients": projected,
            },
        )
    except (FileExistsError, OSError, RuntimeError) as error:
        raise CommandError(f"cannot save projected gradients: {error}") from error
    print(f"projected {collection_name} gradients saved at {output_path}")
