from __future__ import annotations

import torch
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
)

from .inputs import build_input_embeddings


def teacher_forced_forward(
    model: Qwen3TTSForConditionalGeneration,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    inputs = build_input_embeddings(model, batch)
    talker_outputs = model.talker(
        inputs_embeds=inputs["inputs_embeds"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
        output_hidden_states=True,
    )
    hidden_states = talker_outputs.hidden_states[0][-1]

    predecessor_hidden_states = []
    target_audio_codes = []
    for index in range(batch["text_ids"].shape[0]):
        text_token_count = int(batch["text_mask"][index].sum().item())
        audio_frame_count = int(batch["audio_mask"][index].sum().item())
        if audio_frame_count < 1:
            raise ValueError("Qwen3-TTS forward requires at least one audio frame")

        prediction_start = text_token_count + 6
        prediction_end = prediction_start + audio_frame_count
        predecessor_hidden_states.append(
            hidden_states[index, prediction_start:prediction_end]
        )
        target_audio_codes.append(
            batch["audio_codes"][index, :audio_frame_count].to(hidden_states.device)
        )

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

    mtp_outputs = model.talker.code_predictor.forward_finetune(
        inputs_embeds=torch.cat(mtp_input_embeddings, dim=1),
        use_cache=False,
    )
    return {
        "talker_logits": talker_outputs.logits,
        "residual_logits": mtp_outputs.logits,
    }
