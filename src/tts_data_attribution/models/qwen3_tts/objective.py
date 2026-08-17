from __future__ import annotations

import torch
import torch.nn.functional as functional
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
)

from .inputs import build_input_embeddings


def objective(
    model: Qwen3TTSForConditionalGeneration,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    inputs = build_input_embeddings(model, batch)
    outputs = model.talker(
        inputs_embeds=inputs["inputs_embeds"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[0][-1]

    codebook_zero_losses = []
    predecessor_hidden_states = []
    target_audio_codes = []
    audio_lengths = []
    for index in range(batch["text_ids"].shape[0]):
        text_length = int(batch["text_mask"][index].sum().item())
        audio_length = int(batch["audio_mask"][index].sum().item())
        if audio_length < 1:
            raise ValueError("Qwen3-TTS objective requires at least one audio frame")

        prediction_start = text_length + 6
        prediction_end = prediction_start + audio_length + 1
        logits = outputs.logits[index, prediction_start:prediction_end]
        audio_codes = batch["audio_codes"][index, :audio_length].to(logits.device)
        eos_target = torch.tensor(
            [model.config.talker_config.codec_eos_token_id],
            device=logits.device,
        )
        codebook_zero_targets = torch.cat([audio_codes[:, 0], eos_target])
        codebook_zero_losses.append(
            functional.cross_entropy(logits.float(), codebook_zero_targets)
        )

        predecessor_hidden_states.append(
            hidden_states[index, prediction_start : prediction_end - 1]
        )
        target_audio_codes.append(audio_codes)
        audio_lengths.append(audio_length)

    predecessor_hidden_states = torch.cat(predecessor_hidden_states)
    target_audio_codes = torch.cat(target_audio_codes)
    mtp_input_embeddings = [predecessor_hidden_states.unsqueeze(1)]
    mtp_input_embeddings.append(
        model.talker.get_input_embeddings()(target_audio_codes[:, :1])
    )
    residual_embedding_layers = model.talker.code_predictor.get_input_embeddings()
    for codebook in range(1, 15):
        mtp_input_embeddings.append(
            residual_embedding_layers[codebook - 1](
                target_audio_codes[:, codebook : codebook + 1]
            )
        )
    mtp_input_embeddings = torch.cat(mtp_input_embeddings, dim=1)

    mtp_outputs = model.talker.code_predictor.forward_finetune(
        inputs_embeds=mtp_input_embeddings,
        use_cache=False,
    )
    residual_targets = target_audio_codes[:, 1:]
    residual_token_losses = functional.cross_entropy(
        mtp_outputs.logits.float().flatten(0, 1),
        residual_targets.flatten(),
        reduction="none",
    ).view(target_audio_codes.shape[0], 15)

    residual_losses = []
    frame_start = 0
    for audio_length in audio_lengths:
        frame_end = frame_start + audio_length
        residual_losses.append(residual_token_losses[frame_start:frame_end].mean())
        frame_start = frame_end

    return torch.stack(codebook_zero_losses) + 0.3 * torch.stack(residual_losses)
