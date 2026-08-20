from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from torch.utils.data import DataLoader

from ...dataset import UtteranceDataset
from .collate import collate
from .model import load_model
from .objective import objective


def evaluate_adapter_negative_objectives(
    model_path: str | Path,
    adapter_directories: Sequence[str | Path],
    query_utterances: UtteranceDataset,
    speaker_embeddings: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    adapter_evaluated: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    if not adapter_directories:
        raise ValueError("adapter directories must not be empty")
    if len(query_utterances) == 0:
        raise ValueError("query utterances must not be empty")
    query_loader = DataLoader(
        query_utterances,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda utterances: collate(utterances, speaker_embeddings),
    )
    response_rows = []
    for index, adapter_directory in enumerate(adapter_directories, start=1):
        model = None
        try:
            model = load_model(model_path, device=device, dtype=dtype)
            model.talker = cast(
                Any,
                PeftModel.from_pretrained(
                    model.talker,
                    adapter_directory,
                    is_trainable=False,
                ),
            )
            model.eval()
            responses = []
            with torch.no_grad():
                for batch in query_loader:
                    batch = {name: tensor.to(device) for name, tensor in batch.items()}
                    batch_responses = -objective(model, batch)
                    if batch_responses.ndim != 1:
                        raise ValueError("objective must return one value per query")
                    responses.append(batch_responses.float().cpu())
            adapter_responses = torch.cat(responses)
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
        if adapter_evaluated is not None:
            adapter_evaluated(index, len(adapter_directories))
    return torch.stack(response_rows)
