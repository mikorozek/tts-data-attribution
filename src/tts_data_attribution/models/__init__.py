from .lora import apply_lora, is_lora_checkpoint_complete, save_lora_checkpoint
from .training import evaluate, train


def qwen3_tts_encoder() -> type:
    from .qwen3_tts import Qwen3TTSEncoder

    return Qwen3TTSEncoder


def qwen3_tts_adapter_response_evaluator() -> object:
    from .qwen3_tts import evaluate_adapter_negative_objectives

    return evaluate_adapter_negative_objectives


EXPERIMENT_ENCODERS = {"qwen3-tts": qwen3_tts_encoder}
EXPERIMENT_ADAPTER_RESPONSE_EVALUATORS = {
    "qwen3-tts": qwen3_tts_adapter_response_evaluator
}

__all__ = [
    "EXPERIMENT_ADAPTER_RESPONSE_EVALUATORS",
    "EXPERIMENT_ENCODERS",
    "apply_lora",
    "evaluate",
    "is_lora_checkpoint_complete",
    "save_lora_checkpoint",
    "train",
]
