from __future__ import annotations

import torch
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
)


def build_input_embeddings(
    model: Qwen3TTSForConditionalGeneration,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    talker = model.talker
    config = model.config
    talker_config = config.talker_config
    device = talker.get_input_embeddings().weight.device

    text_lengths = batch["text_mask"].sum(dim=1)
    audio_lengths = batch["audio_mask"].sum(dim=1)
    sequence_lengths = text_lengths + audio_lengths + 8
    batch_size = batch["text_ids"].shape[0]
    max_sequence_length = int(sequence_lengths.max().item())

    text_input_ids = torch.full(
        (batch_size, max_sequence_length),
        config.tts_pad_token_id,
        dtype=torch.long,
        device=device,
    )
    codec_input_ids = torch.zeros(
        (batch_size, max_sequence_length, 16),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_sequence_length), dtype=torch.bool, device=device
    )
    codec_embedding_mask = torch.zeros_like(attention_mask)
    audio_embedding_mask = torch.zeros_like(attention_mask)

    for index in range(batch_size):
        text_length = int(text_lengths[index].item())
        audio_length = int(audio_lengths[index].item())
        if text_length < 3:
            raise ValueError("Qwen3-TTS input requires at least three text tokens")

        sequence_length = text_length + audio_length + 8
        text_ids = batch["text_ids"][index, :text_length].to(device)
        audio_codes = batch["audio_codes"][index, :audio_length].to(device)

        text_input_ids[index, :3] = text_ids[:3]
        text_input_ids[index, 7] = config.tts_bos_token_id
        text_input_ids[index, 8 : text_length + 5] = text_ids[3:]
        text_input_ids[index, text_length + 5] = config.tts_eos_token_id

        codec_input_ids[index, 3, 0] = talker_config.codec_nothink_id
        codec_input_ids[index, 4, 0] = talker_config.codec_think_bos_id
        codec_input_ids[index, 5, 0] = talker_config.codec_think_eos_id
        codec_input_ids[index, 7 : text_length + 6, 0] = talker_config.codec_pad_id
        codec_input_ids[index, text_length + 6, 0] = talker_config.codec_bos_id

        audio_start = text_length + 7
        audio_end = audio_start + audio_length
        codec_input_ids[index, audio_start:audio_end] = audio_codes
        codec_input_ids[index, audio_end, 0] = talker_config.codec_eos_token_id

        attention_mask[index, :sequence_length] = True
        codec_embedding_mask[index, 3:sequence_length] = True
        codec_embedding_mask[index, 6] = False
        audio_embedding_mask[index, audio_start:audio_end] = True

    text_embedding_layer = talker.get_text_embeddings()
    text_embeddings = talker.text_projection(text_embedding_layer(text_input_ids))
    text_embeddings = text_embeddings * attention_mask.unsqueeze(-1)

    codec_embedding_layer = talker.get_input_embeddings()
    codec_embeddings = codec_embedding_layer(codec_input_ids[..., 0])
    codec_embeddings = codec_embeddings * codec_embedding_mask.unsqueeze(-1)

    residual_embedding_layers = talker.code_predictor.get_input_embeddings()
    for codebook, embedding_layer in enumerate(residual_embedding_layers, start=1):
        residual_embeddings = embedding_layer(codec_input_ids[..., codebook])
        codec_embeddings = codec_embeddings + (
            residual_embeddings * audio_embedding_mask.unsqueeze(-1)
        )

    inputs_embeds = text_embeddings + codec_embeddings
    speaker_embeddings = batch["speaker_embeddings"].to(
        device=device, dtype=inputs_embeds.dtype
    )
    inputs_embeds[:, 6] = inputs_embeds[:, 6] + speaker_embeddings
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }
