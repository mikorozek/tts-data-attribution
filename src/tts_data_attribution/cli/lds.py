from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from ..dataset import UtteranceDataset
from ..evaluation import (
    LDSConfiguration,
    compute_lds,
    create_membership_matrix,
    discover_completed_subset_runs,
    save_immutable_torch_artifact,
)
from ..experiment import ExperimentManifest, Plan, TrainingRunManifest
from ..models import EXPERIMENT_ADAPTER_RESPONSE_EVALUATORS
from .errors import CommandError


def register(subparsers: argparse._SubParsersAction) -> None:
    lds_parser = subparsers.add_parser(
        "lds", help="linear datamodeling score evaluation"
    )
    lds_subparsers = lds_parser.add_subparsers(required=True)

    compute_parser = lds_subparsers.add_parser(
        "compute", help="evaluate completed subset training runs"
    )
    compute_parser.add_argument("experiment_name", metavar="experiment-name")
    compute_parser.add_argument(
        "--projection",
        dest="projection_name",
        required=True,
        metavar="projection-name",
    )
    compute_parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    compute_parser.set_defaults(run=run_compute)


def run_compute(arguments: argparse.Namespace) -> None:
    for value, name in [
        (arguments.experiment_name, "experiment name"),
        (arguments.projection_name, "projection name"),
    ]:
        if not value or Path(value).name != value:
            raise CommandError(f"{name} must be a single path component")
    configuration = LDSConfiguration(
        correlation="spearman",
        aggregation="mean",
        uncertainty="subset-bootstrap",
        bootstrap_samples=1000,
        confidence_level=0.95,
        seed=0,
    )
    try:
        device = torch.device(arguments.device)
    except RuntimeError as error:
        raise CommandError(f"invalid LDS device: {arguments.device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA is not available")

    experiment_directory = Path("experiments") / arguments.experiment_name
    projection_directory = (
        experiment_directory / "trackstar" / "projections" / arguments.projection_name
    )
    output_path = projection_directory / "lds.pt"
    if output_path.exists():
        raise CommandError(f"LDS evaluation already exists at {output_path}")
    paths = {
        "experiment manifest": experiment_directory / "manifest.yaml",
        "plan": experiment_directory / "plan.json",
        "encoded utterances": experiment_directory / "sampled_utterances_encoded.jsonl",
        "speaker embeddings": experiment_directory / "speaker_embeddings.pt",
        "projection manifest": projection_directory / "manifest.yaml",
        "attributions": projection_directory / "attributions.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise CommandError(f"LDS input is missing: {', '.join(missing)}")

    try:
        experiment = ExperimentManifest.from_yaml(paths["experiment manifest"])
        plan = Plan.from_json(paths["plan"])
        projection_manifest = yaml.safe_load(
            paths["projection manifest"].read_text(encoding="utf-8")
        )
        if not isinstance(projection_manifest, dict):
            raise ValueError("projection manifest must be a mapping")
        full_training_run_name = projection_manifest["training_run"]
        if (
            not isinstance(full_training_run_name, str)
            or not full_training_run_name
            or Path(full_training_run_name).name != full_training_run_name
        ):
            raise ValueError("projection training run must be a single path component")
        full_training_run = TrainingRunManifest.from_yaml(
            experiment_directory
            / "training-runs"
            / full_training_run_name
            / "manifest.yaml"
        )
        if full_training_run.training_set != "training-pool":
            raise ValueError("projection must reference a training-pool run")
        attribution_artifact = torch.load(
            paths["attributions"],
            map_location="cpu",
            weights_only=True,
        )
        training_ids = attribution_artifact["training_ids"]
        query_ids = attribution_artifact["query_ids"]
        attributions = attribution_artifact["attributions"]
        if training_ids != plan.training_pool:
            raise ValueError("attribution training IDs differ from the plan")
        if query_ids != plan.query_pool:
            raise ValueError("attribution query IDs differ from the plan")
        if not isinstance(attributions, torch.Tensor) or attributions.shape != (
            len(training_ids),
            len(query_ids),
        ):
            raise ValueError("attribution matrix dimensions are invalid")
        completed_runs = discover_completed_subset_runs(
            experiment_directory,
            plan,
            full_training_run,
        )
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise CommandError(f"cannot resolve LDS inputs: {error}") from error
    if len(completed_runs) < 2:
        raise CommandError("LDS requires at least two completed subset training runs")

    try:
        encoded = UtteranceDataset.from_jsonl(paths["encoded utterances"])
        speaker_embeddings = torch.load(
            paths["speaker embeddings"],
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CommandError(f"cannot load LDS query data: {error}") from error
    if not isinstance(speaker_embeddings, dict) or not all(
        isinstance(name, str) and isinstance(embedding, torch.Tensor)
        for name, embedding in speaker_embeddings.items()
    ):
        raise CommandError("speaker embeddings must map speaker names to tensors")
    missing_ids = sorted(set(query_ids) - encoded.ids())
    if missing_ids:
        raise CommandError(f"encoded query utterances missing for: {missing_ids}")
    query_utterances = UtteranceDataset(encoded.get_utterances_by_ids(query_ids))
    missing_speakers = sorted(
        {utterance.speaker for utterance in query_utterances} - set(speaker_embeddings)
    )
    if missing_speakers:
        raise CommandError(f"speaker embeddings missing for: {missing_speakers}")
    try:
        evaluator_factory = EXPERIMENT_ADAPTER_RESPONSE_EVALUATORS[experiment.model]
    except KeyError as error:
        raise CommandError(
            f"LDS adapter responses are not implemented for model {experiment.model}"
        ) from error
    evaluator = cast(Any, evaluator_factory())
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[
        full_training_run.dtype
    ]
    try:
        observed_responses = evaluator(
            experiment.model_path,
            [run.target / "adapter" for run in completed_runs],
            query_utterances,
            speaker_embeddings,
            batch_size=full_training_run.batch_size,
            device=device,
            dtype=dtype,
            adapter_evaluated=lambda completed, total: print(
                f"evaluated {completed}/{total} subset models",
                flush=True,
            ),
        )
        membership = create_membership_matrix(
            training_ids,
            [plan.subsets[run.subset_id] for run in completed_runs],
        )
        result = compute_lds(
            attributions,
            membership,
            observed_responses,
            configuration,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CommandError(f"cannot compute LDS: {error}") from error

    artifact = {
        "projection_name": arguments.projection_name,
        "response": "negative_objective",
        "configuration": configuration.to_dict(),
        "training_ids": training_ids,
        "query_ids": query_ids,
        "subset_ids": [run.subset_id for run in completed_runs],
        "training_run_names": [run.training_run_name for run in completed_runs],
        "membership": membership,
        "query_losses": -observed_responses,
        "observed_responses": observed_responses,
        "predicted_responses": result.predicted_responses,
        "per_query_correlations": result.per_query_correlations,
        "per_query_lds": result.per_query_correlations,
        "mean_lds": float(result.per_query_correlations.mean()),
        "aggregate_lds": result.aggregate_correlation,
        "bootstrap_aggregate_lds": result.bootstrap_aggregate_correlations,
        "confidence_interval": result.confidence_interval,
    }
    try:
        save_immutable_torch_artifact(output_path, artifact)
    except FileExistsError as error:
        raise CommandError(f"LDS evaluation already exists at {output_path}") from error
    except (OSError, RuntimeError) as error:
        raise CommandError(f"cannot save LDS evaluation: {error}") from error
    lower, upper = cast(tuple[float, float], result.confidence_interval)
    print(
        f"LDS {result.aggregate_correlation:.6f} "
        f"(95% subset-bootstrap CI [{lower:.6f}, {upper:.6f}]) "
        f"saved at {output_path}"
    )
