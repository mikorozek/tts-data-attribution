from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import experiment, lds, projected_cosine, projection, trackstar_based, training
from .errors import CommandError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tda", description="Data-attribution experiment commands"
    )
    subparsers = parser.add_subparsers(required=True)
    experiment.register(subparsers)
    lds.register(subparsers)
    projected_cosine.register(subparsers)
    training.register(subparsers)
    projection.register(subparsers)
    trackstar_based.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        arguments.run(arguments)
    except CommandError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
