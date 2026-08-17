from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from tts_data_attribution.dataset import Utterance
from tts_data_attribution.models.qwen3_tts import build_input_embeddings, collate


class FakeCodePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(64, 1) for _ in range(15)])
        with torch.no_grad():
            values = torch.arange(64).unsqueeze(1)
            for embedding_layer in self.embedding_layers:
                embedding_layer.weight.copy_(values)

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.embedding_layers


class FakeTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(64, 1)
        self.codec_embedding = nn.Embedding(64, 1)
        self.text_projection = nn.Linear(1, 1, bias=False)
        self.code_predictor = FakeCodePredictor()
        with torch.no_grad():
            self.text_embedding.weight.copy_(torch.arange(64).unsqueeze(1))
            self.codec_embedding.weight.copy_(100 + torch.arange(64).unsqueeze(1))
            self.text_projection.weight.fill_(10)

    def get_text_embeddings(self) -> nn.Embedding:
        return self.text_embedding

    def get_input_embeddings(self) -> nn.Embedding:
        return self.codec_embedding


class FakeModel:
    def __init__(self) -> None:
        self.talker = FakeTalker()
        self.config = SimpleNamespace(
            tts_pad_token_id=20,
            tts_bos_token_id=21,
            tts_eos_token_id=22,
            talker_config=SimpleNamespace(
                codec_nothink_id=30,
                codec_think_bos_id=31,
                codec_think_eos_id=32,
                codec_pad_id=33,
                codec_bos_id=34,
                codec_eos_token_id=35,
            ),
        )


def test_build_input_embeddings_builds_and_pads_qwen_sequences() -> None:
    first_audio_frame = [5] + list(range(1, 16))
    second_audio_frame = [6] + [0] * 15
    third_audio_frame = [10] + [2] * 15
    batch = collate(
        [
            Utterance(
                id="a",
                speaker="a",
                dialogue="d",
                text_ids=[1, 2, 3, 4],
                audio_codes=[first_audio_frame, second_audio_frame],
            ),
            Utterance(
                id="b",
                speaker="b",
                dialogue="d",
                text_ids=[7, 8, 9],
                audio_codes=[third_audio_frame],
            ),
        ],
        {
            "a": torch.tensor([7.0]),
            "b": torch.tensor([8.0]),
        },
    )
    model = FakeModel()

    inputs = build_input_embeddings(model, batch)

    assert inputs["inputs_embeds"].squeeze(-1).tolist() == [
        [10, 20, 30, 330, 331, 332, 207, 343, 173, 353, 334, 425, 306, 335],
        [70, 80, 90, 330, 331, 332, 208, 343, 353, 334, 340, 335, 0, 0],
    ]
    assert inputs["attention_mask"].tolist() == [
        [1] * 14,
        [1] * 12 + [0, 0],
    ]

    inputs["inputs_embeds"].sum().backward()

    assert model.talker.text_embedding.weight.grad is not None
    assert model.talker.codec_embedding.weight.grad is not None
    assert model.talker.text_projection.weight.grad is not None
    assert all(
        embedding_layer.weight.grad is not None
        for embedding_layer in model.talker.code_predictor.embedding_layers
    )
