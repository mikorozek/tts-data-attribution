from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from torch import nn

from tts_data_attribution.models.qwen3_tts import objective


class FakeCodePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(20, 1) for _ in range(15)])
        generator = torch.Generator().manual_seed(1)
        self.base_logits = nn.Parameter(torch.randn(3, 15, 20, generator=generator))
        self.received: dict = {}
        self.last_logits = torch.empty(0)
        with torch.no_grad():
            token_ids = torch.arange(20).unsqueeze(1)
            for codebook, embedding_layer in enumerate(self.embedding_layers, start=1):
                embedding_layer.weight.copy_(1000 * codebook + token_ids)

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.embedding_layers

    def forward_finetune(self, **kwargs):
        self.received = kwargs
        input_signal = kwargs["inputs_embeds"].sum(dim=1).unsqueeze(1)
        class_scale = torch.arange(20).view(1, 1, 20)
        self.last_logits = self.base_logits + input_signal * class_scale
        return SimpleNamespace(logits=self.last_logits)


class FakeTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.logits = nn.Parameter(torch.randn(2, 14, 20, generator=generator))
        self.hidden_states = nn.Parameter(
            torch.arange(28, dtype=torch.float).view(2, 14, 1)
        )
        self.codec_embedding = nn.Embedding(20, 1)
        self.code_predictor = FakeCodePredictor()
        self.received: dict = {}
        with torch.no_grad():
            self.codec_embedding.weight.copy_(100 + torch.arange(20).unsqueeze(1))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.codec_embedding

    def forward(self, **kwargs):
        self.received = kwargs
        return SimpleNamespace(
            logits=self.logits,
            hidden_states=((self.hidden_states,), None),
        )


class FakeModel:
    def __init__(self) -> None:
        self.talker = FakeTalker()
        self.config = SimpleNamespace(
            talker_config=SimpleNamespace(codec_eos_token_id=5)
        )


def test_objective_aligns_codebook_zero_and_residual_targets_per_example(
    monkeypatch,
) -> None:
    objective_module = importlib.import_module(
        "tts_data_attribution.models.qwen3_tts.objective"
    )
    inputs_embeds = torch.zeros(2, 14, 1)
    attention_mask = torch.tensor([[True] * 14, [True] * 12 + [False, False]])
    monkeypatch.setattr(
        objective_module,
        "build_input_embeddings",
        lambda model, batch: {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        },
    )
    audio_codes = torch.zeros(2, 2, 16, dtype=torch.long)
    audio_codes[0, 0] = torch.arange(1, 17)
    audio_codes[0, 1] = torch.arange(2, 18)
    audio_codes[1, 0] = torch.arange(3, 19)
    batch = {
        "text_ids": torch.zeros(2, 4, dtype=torch.long),
        "text_mask": torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        ),
        "audio_codes": audio_codes,
        "audio_mask": torch.tensor([[True, True], [True, False]]),
        "speaker_embeddings": torch.zeros(2, 1),
    }
    model = FakeModel()

    losses = objective(model, batch)

    codebook_zero_losses = torch.stack(
        [
            functional.cross_entropy(
                model.talker.logits[0, 10:13], torch.tensor([1, 2, 5])
            ),
            functional.cross_entropy(
                model.talker.logits[1, 9:11], torch.tensor([3, 5])
            ),
        ]
    )
    residual_targets = torch.cat([audio_codes[0, :2, 1:], audio_codes[1, :1, 1:]])
    residual_token_losses = functional.cross_entropy(
        model.talker.code_predictor.last_logits.flatten(0, 1),
        residual_targets.flatten(),
        reduction="none",
    ).view(3, 15)
    residual_losses = torch.stack(
        [residual_token_losses[:2].mean(), residual_token_losses[2:].mean()]
    )
    assert torch.allclose(losses, codebook_zero_losses + 0.3 * residual_losses)

    expected_mtp_inputs = []
    predecessor_hidden_states = [10, 11, 23]
    target_frames = [audio_codes[0, 0], audio_codes[0, 1], audio_codes[1, 0]]
    for hidden_state, target_frame in zip(
        predecessor_hidden_states, target_frames, strict=True
    ):
        expected_mtp_inputs.append(
            [hidden_state, 100 + target_frame[0].item()]
            + [
                1000 * codebook + target_frame[codebook].item()
                for codebook in range(1, 15)
            ]
        )
    mtp_inputs = model.talker.code_predictor.received["inputs_embeds"]
    assert mtp_inputs.squeeze(-1).tolist() == expected_mtp_inputs
    assert model.talker.code_predictor.received["use_cache"] is False
    assert model.talker.received == {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "use_cache": False,
        "output_hidden_states": True,
    }

    losses.sum().backward()

    codebook_zero_positions = model.talker.logits.grad.abs().sum(dim=-1) > 0
    assert codebook_zero_positions.tolist() == [
        [False] * 10 + [True, True, True, False],
        [False] * 9 + [True, True, False, False, False],
    ]
    predecessor_positions = model.talker.hidden_states.grad.abs().sum(dim=-1) > 0
    assert predecessor_positions.tolist() == [
        [False] * 10 + [True, True, False, False],
        [False] * 9 + [True, False, False, False, False],
    ]
    assert model.talker.code_predictor.base_logits.grad is not None
