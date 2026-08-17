from .gradients import collect_per_example_gradients
from .lora import apply_lora, save_lora_checkpoint


def qwen3_tts_encoder() -> type:
    from .qwen3_tts import Qwen3TTSEncoder

    return Qwen3TTSEncoder


EXPERIMENT_ENCODERS = {"qwen3-tts": qwen3_tts_encoder}

__all__ = [
    "EXPERIMENT_ENCODERS",
    "apply_lora",
    "collect_per_example_gradients",
    "save_lora_checkpoint",
]
