from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from torch import nn

from tts_data_attribution.models.qwen3_tts import loss_components, objective


def make_batch() -> dict[str, torch.Tensor]:
    audio_codes = torch.zeros(2, 2, 16, dtype=torch.long)
    audio_codes[0, 0] = torch.arange(1, 17)
    audio_codes[0, 1] = torch.arange(2, 18)
    audio_codes[1, 0] = torch.arange(3, 19)
    return {
        "text_ids": torch.zeros(2, 4, dtype=torch.long),
        "text_mask": torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        ),
        "audio_codes": audio_codes,
        "audio_mask": torch.tensor([[True, True], [True, False]]),
        "speaker_embeddings": torch.zeros(2, 1),
    }


def test_loss_components_aligns_targets_and_reduces_per_example() -> None:
    generator = torch.Generator().manual_seed(0)
    talker_logits = nn.Parameter(torch.randn(2, 14, 20, generator=generator))
    residual_logits = nn.Parameter(torch.randn(3, 15, 20, generator=generator))
    batch = make_batch()

    components = loss_components(
        {
            "talker_logits": talker_logits,
            "residual_logits": residual_logits,
        },
        batch,
        codec_eos_token_id=5,
    )

    expected_codebook_zero_losses = torch.stack(
        [
            functional.cross_entropy(talker_logits[0, 10:13], torch.tensor([1, 2, 5])),
            functional.cross_entropy(talker_logits[1, 9:11], torch.tensor([3, 5])),
        ]
    )
    audio_codes = batch["audio_codes"]
    residual_targets = torch.cat([audio_codes[0, :2, 1:], audio_codes[1, :1, 1:]])
    residual_token_losses = functional.cross_entropy(
        residual_logits.flatten(0, 1),
        residual_targets.flatten(),
        reduction="none",
    ).view(3, 15)
    expected_residual_losses = torch.stack(
        [residual_token_losses[:2].mean(), residual_token_losses[2:].mean()]
    )
    assert torch.allclose(
        components["codebook_zero_losses"], expected_codebook_zero_losses
    )
    assert torch.allclose(components["residual_losses"], expected_residual_losses)
    assert torch.allclose(
        components["losses"],
        expected_codebook_zero_losses + 0.3 * expected_residual_losses,
    )

    components["losses"].sum().backward()

    codebook_zero_positions = talker_logits.grad.abs().sum(dim=-1) > 0
    assert codebook_zero_positions.tolist() == [
        [False] * 10 + [True, True, True, False],
        [False] * 9 + [True, True, False, False, False],
    ]
    assert residual_logits.grad is not None


def test_objective_composes_forward_and_loss_components(monkeypatch) -> None:
    objective_module = importlib.import_module(
        "tts_data_attribution.models.qwen3_tts.objective"
    )
    expected_logits = {
        "talker_logits": torch.empty(0),
        "residual_logits": torch.empty(0),
    }
    expected_losses = torch.tensor([1.0, 2.0])
    received: list = []
    model = SimpleNamespace(
        config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=5))
    )
    batch = make_batch()

    monkeypatch.setattr(
        objective_module,
        "teacher_forced_forward",
        lambda received_model, received_batch: expected_logits,
    )

    def fake_loss_components(logits, received_batch, codec_eos_token_id):
        received.append((logits, received_batch, codec_eos_token_id))
        return {"losses": expected_losses}

    monkeypatch.setattr(objective_module, "loss_components", fake_loss_components)

    losses = objective(model, batch)

    assert losses is expected_losses
    assert received == [(expected_logits, batch, 5)]
