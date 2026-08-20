from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_immutable_torch_artifact(
    path: str | Path,
    artifact: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            torch.save(artifact, file)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
