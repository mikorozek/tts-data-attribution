from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from ..evaluation import save_immutable_torch_artifact
from .errors import CommandError, require_flat_name
from .projection import load_projected_pools


def register(subparsers: argparse._SubParsersAction) -> None:
    cosine_parser = subparsers.add_parser(
        "projected-cosine",
        help="projected cosine attribution",
    )
    cosine_subparsers = cosine_parser.add_subparsers(required=True)

    compute_parser = cosine_subparsers.add_parser(
        "compute",
        help="compute a projected cosine attribution matrix",
    )
    compute_parser.add_argument("experiment_name", metavar="experiment-name")
    compute_parser.add_argument("training_run_name", metavar="training-run-name")
    compute_parser.add_argument("projection_name", metavar="projection-name")
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

    try:
        device = torch.device(arguments.device)
    except RuntimeError as error:
        raise CommandError(
            f"invalid projected cosine device: {arguments.device}"
        ) from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA is not available")

    experiment_directory = Path("experiments") / arguments.experiment_name
    projection_directory = (
        experiment_directory
        / "training-runs"
        / arguments.training_run_name
        / "projections"
        / arguments.projection_name
    )
    output_path = projection_directory / "projected-cosine" / "attributions.pt"
    if output_path.parent.exists():
        raise CommandError(
            f"projected cosine attribution already exists at {output_path.parent}"
        )

    try:
        pools = load_projected_pools(
            experiment_directory,
            projection_directory,
            device,
        )
        training_norms = torch.linalg.vector_norm(
            pools.training,
            dim=1,
            keepdim=True,
        )
        query_norms = torch.linalg.vector_norm(
            pools.queries,
            dim=1,
            keepdim=True,
        )
        if torch.any(training_norms == 0) or torch.any(query_norms == 0):
            raise ValueError("cannot normalize a zero projected gradient")
        if not torch.all(torch.isfinite(training_norms)) or not torch.all(
            torch.isfinite(query_norms)
        ):
            raise ValueError("cannot normalize a non-finite projected gradient norm")
        attributions = (pools.training / training_norms) @ (
            pools.queries / query_norms
        ).T
        if not torch.all(torch.isfinite(attributions)):
            raise ValueError("projected cosine attribution contains non-finite values")
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise CommandError(f"cannot compute projected cosine: {error}") from error

    try:
        output_path.parent.mkdir(parents=True)
        save_immutable_torch_artifact(output_path, attributions.cpu())
    except (FileExistsError, OSError, RuntimeError) as error:
        raise CommandError(
            f"cannot save projected cosine attribution: {error}"
        ) from error
    print(f"projected cosine attribution saved at {output_path}")
