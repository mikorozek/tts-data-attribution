from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from tts_data_attribution.dataset import Utterance, UtteranceDataset
from tts_data_attribution.models.qwen3_tts import evaluation as qwen_evaluation


class TinyTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter_index = 0


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.talker = TinyTalker()


def test_adapter_negative_objectives_preserve_adapter_and_query_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_models: list[TinyModel] = []

    def fake_load_model(*args: object, **kwargs: object) -> TinyModel:
        model = TinyModel()
        loaded_models.append(model)
        return model

    class FakePeftModel:
        @staticmethod
        def from_pretrained(
            talker: TinyTalker,
            path: Path,
            *,
            is_trainable: bool,
        ) -> TinyTalker:
            assert not is_trainable
            talker.adapter_index = int(path.name[-1])
            return talker

    def fake_collate(
        utterances: list[Utterance],
        speaker_embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert "speaker" in speaker_embeddings
        return {
            "query_indices": torch.tensor(
                [utterance.text_ids[0] for utterance in utterances]
            )
        }

    losses = torch.tensor([[3.0, 1.0], [2.0, 4.0]])

    def fake_objective(
        model: TinyModel,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        assert not model.training
        assert not torch.is_grad_enabled()
        return losses[model.talker.adapter_index, batch["query_indices"]]

    monkeypatch.setattr(qwen_evaluation, "load_model", fake_load_model)
    monkeypatch.setattr(qwen_evaluation, "PeftModel", FakePeftModel)
    monkeypatch.setattr(qwen_evaluation, "collate", fake_collate)
    monkeypatch.setattr(qwen_evaluation, "objective", fake_objective)
    queries = UtteranceDataset(
        [
            Utterance(
                id=f"query-{index}",
                speaker="speaker",
                dialogue=f"dialogue-{index}",
                text_ids=[index],
                audio_codes=[[2] * 16],
            )
            for index in range(2)
        ]
    )
    progress = []

    responses = qwen_evaluation.evaluate_adapter_negative_objectives(
        Path("model"),
        [Path("adapter-0"), Path("adapter-1")],
        queries,
        {"speaker": torch.ones(4)},
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        adapter_evaluated=lambda completed, total: progress.append((completed, total)),
    )

    torch.testing.assert_close(responses, -losses)
    assert len(loaded_models) == 2
    assert progress == [(1, 2), (2, 2)]
