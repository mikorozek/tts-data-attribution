from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from ..attribution import GaussNewtonHessianApproximation
from ..evaluation import save_immutable_torch_artifact
from .errors import CommandError, require_flat_name
from .projection import load_projected_pools


def register(subparsers: argparse._SubParsersAction) -> None:
    attribution_parser = subparsers.add_parser(
        "attribution", help="autocorrelation-corrected attribution"
    )
    attribution_subparsers = attribution_parser.add_subparsers(required=True)

    compute_parser = attribution_subparsers.add_parser(
        "compute", help="compute an attribution matrix"
    )
    compute_parser.add_argument("experiment_name", metavar="experiment-name")
    compute_parser.add_argument("training_run_name", metavar="training-run-name")
    compute_parser.add_argument("projection_name", metavar="projection-name")
    compute_parser.add_argument("--task-weight", type=float, required=True)
    compute_parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    compute_parser.set_defaults(run=run_compute)


def run_compute(arguments: argparse.Namespace) -> None:
    for value, name in [
        (arguments.experiment_name, "experiment name"),
        (arguments.training_run_name, "training run name"),
        (arguments.projection_name, "projection name"),
    ]:
        require_flat_name(value, name)

    experiment_directory = Path("experiments") / arguments.experiment_name
    projection_directory = (
        experiment_directory
        / "training-runs"
        / arguments.training_run_name
        / "projections"
        / arguments.projection_name
    )
    method_directory = projection_directory / "attribution"
    if method_directory.exists():
        raise CommandError(f"attribution already exists at {method_directory}")

    try:
        device = torch.device(arguments.device)
    except RuntimeError as error:
        raise CommandError(f"invalid attribution device: {arguments.device}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA is not available")

    try:
        pools = load_projected_pools(
            experiment_directory,
            projection_directory,
            device,
        )
        approximation = GaussNewtonHessianApproximation(
            task_weight=arguments.task_weight
        )
        hessian = approximation.compute(pools.training, pools.queries)
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise CommandError(f"cannot load attribution inputs: {error}") from error

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        largest_eigenvalue = eigenvalues[-1]
        if largest_eigenvalue <= 0:
            raise ValueError("Hessian approximation has no positive directions")
        tolerance = (
            torch.finfo(eigenvalues.dtype).eps * hessian.shape[0] * largest_eigenvalue
        )
        retained = eigenvalues > tolerance
        if not torch.any(retained):
            raise ValueError("Hessian approximation has no positive directions")
        basis = eigenvectors[:, retained]
        inverse_values = eigenvalues[retained].rsqrt()
        inverse_hessian_square_root = (basis * inverse_values) @ basis.T

        corrected_training = pools.training @ inverse_hessian_square_root
        corrected_queries = pools.queries @ inverse_hessian_square_root
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
        attributions = (corrected_training / training_norms) @ (
            corrected_queries / query_norms
        ).T
        if not torch.all(torch.isfinite(attributions)):
            raise ValueError("attribution tensor contains non-finite values")
    except (RuntimeError, TypeError, ValueError) as error:
        raise CommandError(f"cannot compute attribution: {error}") from error

    try:
        method_directory.mkdir(parents=True)
        (method_directory / "config.yaml").write_text(
            yaml.safe_dump({"task_weight": arguments.task_weight}, sort_keys=False),
            encoding="utf-8",
        )
        save_immutable_torch_artifact(
            method_directory / "attributions.pt",
            attributions.cpu(),
        )
    except (FileExistsError, OSError, RuntimeError, yaml.YAMLError) as error:
        raise CommandError(f"cannot save attribution: {error}") from error
    print(f"attribution saved at {method_directory / 'attributions.pt'}")
