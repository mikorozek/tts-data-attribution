from __future__ import annotations

from pathlib import Path


class CommandError(Exception):
    pass


def require_flat_name(value: str, name: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise CommandError(f"{name} must be a single path component")
