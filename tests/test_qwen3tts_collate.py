from __future__ import annotations

import torch

from tts_data_attribution.dataset import Utterance
from tts_data_attribution.models.qwen3_tts import collate


def test_collate_right_pads_encoded_utterances() -> None:
    first_audio_frame = list(range(100, 116))
    second_audio_frame = list(range(200, 216))
    third_audio_frame = list(range(300, 316))
    utterances = [
        Utterance(
            id="a",
            speaker="speaker-a",
            dialogue="dialogue-a",
            text_ids=[10, 11, 12, 13],
            audio_codes=[first_audio_frame, second_audio_frame],
        ),
        Utterance(
            id="b",
            speaker="speaker-b",
            dialogue="dialogue-b",
            text_ids=[20, 21, 22],
            audio_codes=[third_audio_frame],
        ),
    ]
    first_speaker_embedding = torch.tensor([0.1, 0.2, 0.3])
    second_speaker_embedding = torch.tensor([0.4, 0.5, 0.6])

    batch = collate(
        utterances,
        {
            "speaker-a": first_speaker_embedding,
            "speaker-b": second_speaker_embedding,
        },
    )

    assert batch["text_ids"].tolist() == [
        [10, 11, 12, 13],
        [20, 21, 22, 0],
    ]
    assert batch["text_mask"].tolist() == [
        [True, True, True, True],
        [True, True, True, False],
    ]
    assert batch["audio_codes"].tolist() == [
        [first_audio_frame, second_audio_frame],
        [third_audio_frame, [0] * 16],
    ]
    assert batch["audio_mask"].tolist() == [
        [True, True],
        [True, False],
    ]
    assert torch.equal(
        batch["speaker_embeddings"],
        torch.stack([first_speaker_embedding, second_speaker_embedding]),
    )
