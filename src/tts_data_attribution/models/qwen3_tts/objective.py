from __future__ import annotations

import torch
import torch.nn.functional as functional
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
)

from .forward import teacher_forced_forward


def loss_components(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    codec_eos_token_id: int,
) -> dict[str, torch.Tensor]:
    talker_logits = logits["talker_logits"]
    residual_logits = logits["residual_logits"]
    codebook_zero_losses = []
    residual_losses = []
    residual_frame_start = 0

    for index in range(batch["text_ids"].shape[0]):
        text_token_count = int(batch["text_mask"][index].sum().item())
        audio_frame_count = int(batch["audio_mask"][index].sum().item())
        if audio_frame_count < 1:
            raise ValueError("Qwen3-TTS objective requires at least one audio frame")

        prediction_start = text_token_count + 6
        prediction_end = prediction_start + audio_frame_count + 1
        example_logits = talker_logits[index, prediction_start:prediction_end]
        audio_codes = batch["audio_codes"][index, :audio_frame_count].to(
            example_logits.device
        )
        eos_target = torch.tensor([codec_eos_token_id], device=example_logits.device)
        codebook_zero_targets = torch.cat([audio_codes[:, 0], eos_target])
        codebook_zero_losses.append(
            functional.cross_entropy(example_logits.float(), codebook_zero_targets)
        )

        residual_frame_end = residual_frame_start + audio_frame_count
        example_residual_logits = residual_logits[
            residual_frame_start:residual_frame_end
        ]
        residual_losses.append(
            functional.cross_entropy(
                example_residual_logits.float().flatten(0, 1),
                audio_codes[:, 1:].to(residual_logits.device).flatten(),
            )
        )
        residual_frame_start = residual_frame_end

    codebook_zero_losses = torch.stack(codebook_zero_losses)
    residual_losses = torch.stack(residual_losses)
    return {
        "codebook_zero_losses": codebook_zero_losses,
        "residual_losses": residual_losses,
        "losses": codebook_zero_losses + 0.3 * residual_losses,
    }


def objective(
    model: Qwen3TTSForConditionalGeneration,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    logits = teacher_forced_forward(model, batch)
    components = loss_components(
        logits,
        batch,
        model.config.talker_config.codec_eos_token_id,
    )
    return components["losses"]
