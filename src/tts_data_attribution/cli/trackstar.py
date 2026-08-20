from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from ..experiment import Plan
from ..trackstar import GaussNewtonHessianApproximation
from .errors import CommandError


def register(subparsers: argparse._SubParsersAction) -> None:
    trackstar_parser = subparsers.add_parser(
        "trackstar", help="TrackStar attribution"
    )
    trackstar_subparsers = trackstar_parser.add_subparsers(required=True)

    compute_parser = trackstar_subparsers.add_parser(
        "compute", help="compute an attribution matrix"
    )
    compute_parser.add_argument("experiment_name", metavar="experiment-name")
    compute_parser.add_argument("projection_name", metavar="projection-name")
    compute_parser.add_argument("--task-weight", type=float, required=True)
    compute_parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    compute_parser.set_defaults(run=run_compute)


def run_compute(arguments: argparse.Namespace) -> None:
    if Path(arguments.projection_name).name != arguments.projection_name:
        raise CommandError("projection name must be a single path component")

    experiment_directory = Path("experiments") / arguments.experiment_name
    projection_directory = (
        experiment_directory
        / "trackstar"
        / "projections"
        / arguments.projection_name
    )
    paths = {
        "plan": experiment_directory / "plan.json",
        "projection manifest": projection_directory / "manifest.yaml",
        "projected training pool": projection_directory
        / "projected"
        / "training-pool.pt",
        "projected query pool": projection_directory
        / "projected"
        / "query-pool.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise CommandError(f"TrackStar input is missing: {', '.join(missing)}")

    output_path = projection_directory / "attributions.pt"
    if output_path.exists():
        raise CommandError(f"attributions already exist at {output_path}")

    try:
        device = torch.device(arguments.device)
    except RuntimeError as error:
        raise CommandError(f"invalid TrackStar device: {arguments.device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA is not available")

    try:
        plan = Plan.from_json(paths["plan"])
        projection_manifest = yaml.safe_load(
            paths["projection manifest"].read_text(encoding="utf-8")
        )
        training_artifact = torch.load(
            paths["projected training pool"],
            map_location=device,
            weights_only=True,
        )
        query_artifact = torch.load(
            paths["projected query pool"],
            map_location=device,
            weights_only=True,
        )
        if not isinstance(projection_manifest, dict):
            raise ValueError("projection manifest must be a mapping")
        if projection_manifest.get("type") != "two-sided":
            raise ValueError("projection type must be two-sided")
        output_dimension = projection_manifest["output_dimension"]
        if type(output_dimension) is not int or output_dimension < 1:
            raise ValueError("projection output dimension must be a positive integer")
        if not isinstance(training_artifact, dict) or not isinstance(
            query_artifact, dict
        ):
            raise ValueError("projected pool artifacts must be mappings")
        training_ids = training_artifact["ids"]
        query_ids = query_artifact["ids"]
        training = training_artifact["projected_gradients"]
        queries = query_artifact["projected_gradients"]
        if training_ids != plan.training_pool:
            raise ValueError("projected training IDs differ from the experiment plan")
        if query_ids != plan.query_pool:
            raise ValueError("projected query IDs differ from the experiment plan")
        if not isinstance(training, torch.Tensor) or not isinstance(
            queries, torch.Tensor
        ):
            raise ValueError("projected gradients must be tensors")
        if (
            training.ndim != 2
            or queries.ndim != 2
            or training.shape != (len(training_ids), output_dimension)
            or queries.shape != (len(query_ids), output_dimension)
        ):
            raise ValueError("projected gradient dimensions are invalid")
        approximation = GaussNewtonHessianApproximation(
            task_weight=arguments.task_weight
        )
        hessian = approximation.compute(training, queries)
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise CommandError(f"cannot load TrackStar inputs: {error}") from error

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        largest_eigenvalue = eigenvalues[-1]
        if largest_eigenvalue <= 0:
            raise ValueError("Hessian approximation has no positive directions")
        tolerance = (
            torch.finfo(eigenvalues.dtype).eps
            * hessian.shape[0]
            * largest_eigenvalue
        )
        retained = eigenvalues > tolerance
        if not torch.any(retained):
            raise ValueError("Hessian approximation has no positive directions")
        basis = eigenvectors[:, retained]
        inverse_values = eigenvalues[retained].rsqrt()
        inverse_hessian_square_root = (basis * inverse_values) @ basis.T

        corrected_training = training @ inverse_hessian_square_root
        corrected_queries = queries @ inverse_hessian_square_root
        training_norms = torch.linalg.vector_norm(
            corrected_training,
            dim=1,
            keepdim=True,
        )
        query_norms = torch.linalg.vector_norm(
            corrected_queries,
            dim=1,
            keepdim=True,
        )
        if (
            torch.any(training_norms == 0)
            or torch.any(query_norms == 0)
            or not torch.all(torch.isfinite(training_norms))
            or not torch.all(torch.isfinite(query_norms))
        ):
            raise ValueError("cannot normalize a zero or non-finite representation")
        normalized_training = corrected_training / training_norms
        normalized_queries = corrected_queries / query_norms
        attributions = normalized_training @ normalized_queries.T
        if not torch.all(torch.isfinite(attributions)):
            raise ValueError("TrackStar attribution contains non-finite values")
    except (RuntimeError, TypeError, ValueError) as error:
        raise CommandError(f"cannot compute TrackStar attribution: {error}") from error

    try:
        torch.save(
            {
                "task_weight": arguments.task_weight,
                "training_ids": training_ids,
                "query_ids": query_ids,
                "attributions": attributions.cpu(),
            },
            output_path,
        )
    except (OSError, RuntimeError) as error:
        raise CommandError(f"cannot save TrackStar attribution: {error}") from error
    print(f"TrackStar attribution saved at {output_path}")
