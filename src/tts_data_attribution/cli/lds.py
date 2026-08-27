from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from peft import PeftModel
from torch.utils.data import DataLoader

from ..dataset import UtteranceDataset
from ..evaluation import (
    compute_lds,
    create_membership_matrix,
    save_immutable_torch_artifact,
)
from ..experiment import ExperimentManifest, Plan
from ..models import is_lora_checkpoint_complete, load_checkpoint_metrics
from ..models.qwen3_tts import collate, load_model, objective
from .errors import CommandError
from .projection import resolve_training_checkpoint


def register(subparsers: argparse._SubParsersAction) -> None:
    lds_parser = subparsers.add_parser(
        "lds", help="linear datamodeling score evaluation"
    )
    lds_subparsers = lds_parser.add_subparsers(required=True)

    compute_parser = lds_subparsers.add_parser(
        "compute", help="evaluate one attribution artifact"
    )
    compute_parser.add_argument(
        "attributions",
        metavar="attributions-path",
        help="path to attribution/attributions.pt or projected-cosine/attributions.pt",
    )
    compute_parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device used to evaluate subset adapters",
    )
    compute_parser.set_defaults(run=run_compute)


def _resolve_workspace(path: Path) -> tuple[Path, Path, Path, Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    path = path.resolve()
    method_directory = path.parent
    projection_directory = method_directory.parent
    training_run_directory = projection_directory.parent.parent
    if (
        path.name != "attributions.pt"
        or method_directory.name not in {"attribution", "projected-cosine"}
        or projection_directory.parent.name != "projections"
        or training_run_directory.parent.name != "training-runs"
        or not (projection_directory / "manifest.yaml").is_file()
        or not (training_run_directory / "manifest.yaml").is_file()
    ):
        raise ValueError(
            "attribution artifact must belong to a training-run projection method"
        )
    experiment_directory = training_run_directory.parent.parent
    output_path = method_directory / "lds.pt"
    if output_path.exists():
        raise FileExistsError(output_path)
    return (
        experiment_directory,
        training_run_directory,
        projection_directory,
        output_path,
    )


def _resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except RuntimeError as error:
        raise CommandError(f"invalid LDS device: {value}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA is not available")
    return device


def _load_attributions(path: Path, plan: Plan) -> torch.Tensor:
    attributions = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(attributions, torch.Tensor)
        or not attributions.is_floating_point()
        or attributions.shape != (len(plan.training_pool), len(plan.query_pool))
    ):
        raise ValueError("attribution tensor dimensions or dtype are invalid")
    if not torch.isfinite(attributions).all():
        raise ValueError("attribution tensor must be finite")
    return attributions


def _subset_checkpoints(
    training_run_directory: Path,
    plan: Plan,
    epoch: int,
    maximum_epoch: int,
) -> list[Path]:
    checkpoints = [
        training_run_directory
        / "subsets"
        / subset_id
        / "checkpoints"
        / f"epoch-{epoch:04d}"
        for subset_id in plan.subsets
    ]
    missing = [
        subset_id
        for subset_id, checkpoint in zip(plan.subsets, checkpoints, strict=True)
        if not is_lora_checkpoint_complete(checkpoint)
    ]
    if missing:
        raise ValueError(f"subset checkpoints are missing for: {', '.join(missing)}")
    for subset_id, checkpoint in zip(plan.subsets, checkpoints, strict=True):
        try:
            load_checkpoint_metrics(
                checkpoint / "metrics.json",
                expected_epoch=epoch,
                maximum_epoch=maximum_epoch,
            )
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"checkpoint metrics are invalid for {subset_id}: {error}"
            ) from error
    return checkpoints


def _evaluate_subset_responses(
    experiment_directory: Path,
    experiment: ExperimentManifest,
    checkpoint_paths: list[Path],
    query_ids: list[str],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    encoded_path = experiment_directory / "sampled_utterances_encoded.jsonl"
    speaker_embeddings_path = experiment_directory / "speaker_embeddings.pt"
    missing = [
        name
        for name, path in [
            ("encoded utterances", encoded_path),
            ("speaker embeddings", speaker_embeddings_path),
        ]
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"LDS query input is missing: {', '.join(missing)}")

    encoded = UtteranceDataset.from_jsonl(encoded_path)
    speaker_embeddings = torch.load(
        speaker_embeddings_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(speaker_embeddings, dict) or not all(
        isinstance(name, str) and isinstance(embedding, torch.Tensor)
        for name, embedding in speaker_embeddings.items()
    ):
        raise ValueError("speaker embeddings must map speaker names to tensors")
    missing_ids = sorted(set(query_ids) - encoded.ids())
    if missing_ids:
        raise ValueError(f"encoded query utterances missing for: {missing_ids}")
    query_utterances = UtteranceDataset(encoded.get_utterances_by_ids(query_ids))
    missing_speakers = sorted(
        {utterance.speaker for utterance in query_utterances} - set(speaker_embeddings)
    )
    if missing_speakers:
        raise ValueError(f"speaker embeddings missing for: {missing_speakers}")
    if experiment.model != "qwen3-tts":
        raise ValueError(
            f"LDS adapter responses are not implemented for model {experiment.model}"
        )
    query_loader = DataLoader(
        query_utterances,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda utterances: collate(utterances, speaker_embeddings),
    )
    response_rows = []
    for index, checkpoint in enumerate(checkpoint_paths, start=1):
        model = None
        try:
            model = load_model(experiment.model_path, device=device, dtype=dtype)
            model.talker = cast(
                Any,
                PeftModel.from_pretrained(
                    model.talker,
                    checkpoint / "adapter",
                    is_trainable=False,
                ),
            )
            model.eval()
            batch_responses = []
            with torch.no_grad():
                for batch in query_loader:
                    batch = {name: tensor.to(device) for name, tensor in batch.items()}
                    values = -objective(model, batch)
                    if values.ndim != 1:
                        raise ValueError("objective must return one value per query")
                    batch_responses.append(values.float().cpu())
            adapter_responses = torch.cat(batch_responses)
            if adapter_responses.shape != (len(query_utterances),):
                raise ValueError("adapter query responses are incomplete")
            if not torch.isfinite(adapter_responses).all():
                raise ValueError("adapter query responses are non-finite")
            response_rows.append(adapter_responses)
        finally:
            if model is not None:
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(
            f"evaluated {index}/{len(checkpoint_paths)} subset models",
            flush=True,
        )
    responses = torch.stack(response_rows)
    if (
        not isinstance(responses, torch.Tensor)
        or not responses.is_floating_point()
        or responses.shape != (len(checkpoint_paths), len(query_ids))
    ):
        raise ValueError("subset response matrix dimensions or dtype are invalid")
    responses = responses.detach().cpu()
    if not torch.isfinite(responses).all():
        raise ValueError("subset responses must be finite")
    return responses


def run_compute(arguments: argparse.Namespace) -> None:
    attribution_path = Path(arguments.attributions)
    try:
        (
            experiment_directory,
            training_run_directory,
            projection_directory,
            output_path,
        ) = _resolve_workspace(attribution_path)
    except FileNotFoundError as error:
        raise CommandError(f"LDS attribution input is missing: {error}") from error
    except FileExistsError as error:
        raise CommandError(f"LDS evaluation already exists at {error}") from error
    except ValueError as error:
        raise CommandError(str(error)) from error

    experiment_manifest_path = experiment_directory / "manifest.yaml"
    plan_path = experiment_directory / "plan.json"
    projection_manifest_path = projection_directory / "manifest.yaml"
    try:
        experiment = ExperimentManifest.from_yaml(experiment_manifest_path)
        plan = Plan.from_json(plan_path)
        attributions = _load_attributions(attribution_path.resolve(), plan)
        projection_manifest = yaml.safe_load(
            projection_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(projection_manifest, dict):
            raise ValueError("projection manifest must be a mapping")
        _, training_run, epoch, _ = resolve_training_checkpoint(
            training_run_directory,
            projection_manifest,
        )
        checkpoint_paths = _subset_checkpoints(
            training_run_directory,
            plan,
            epoch,
            training_run.epochs,
        )
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
        raise CommandError(f"cannot resolve LDS inputs: {error}") from error

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[training_run.dtype]
    try:
        observed_responses = _evaluate_subset_responses(
            experiment_directory,
            experiment,
            checkpoint_paths,
            plan.query_pool,
            training_run.batch_size,
            dtype,
            _resolve_device(arguments.device),
        )
        membership = create_membership_matrix(
            plan.training_pool,
            list(plan.subsets.values()),
        )
        result = compute_lds(
            attributions,
            membership,
            observed_responses,
        )
        save_immutable_torch_artifact(output_path, result)
    except FileExistsError as error:
        raise CommandError(f"LDS evaluation already exists at {output_path}") from error
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise CommandError(f"cannot compute LDS: {error}") from error

    print(f"LDS {result:.6f} saved at {output_path}")
