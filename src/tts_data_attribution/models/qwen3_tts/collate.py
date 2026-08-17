from __future__ import annotations

import torch

from ...dataset import Utterance


def collate(
    utterances: list[Utterance],
    speaker_embeddings: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not utterances:
        raise ValueError("cannot collate an empty batch")

    batch_size = len(utterances)
    max_text_length = max(len(utterance.text_ids) for utterance in utterances)
    max_audio_length = max(len(utterance.audio_codes) for utterance in utterances)

    text_ids = torch.zeros((batch_size, max_text_length), dtype=torch.long)
    text_mask = torch.zeros((batch_size, max_text_length), dtype=torch.bool)
    audio_codes = torch.zeros((batch_size, max_audio_length, 16), dtype=torch.long)
    audio_mask = torch.zeros((batch_size, max_audio_length), dtype=torch.bool)

    for index, utterance in enumerate(utterances):
        text_length = len(utterance.text_ids)
        audio_length = len(utterance.audio_codes)
        text_ids[index, :text_length] = torch.tensor(
            utterance.text_ids, dtype=torch.long
        )
        text_mask[index, :text_length] = True
        audio_codes[index, :audio_length] = torch.tensor(
            utterance.audio_codes, dtype=torch.long
        )
        audio_mask[index, :audio_length] = True

    batch_speaker_embeddings = torch.stack(
        [speaker_embeddings[utterance.speaker] for utterance in utterances]
    )
    return {
        "text_ids": text_ids,
        "text_mask": text_mask,
        "audio_codes": audio_codes,
        "audio_mask": audio_mask,
        "speaker_embeddings": batch_speaker_embeddings,
    }
