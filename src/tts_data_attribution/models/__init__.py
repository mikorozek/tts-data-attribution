from .gradients import (
    collect_per_example_gradients,
    correct_gradients_with_adamw,
)
from .lora import apply_lora, save_lora_checkpoint
from .projection import BlockDiagonalProjector, TwoSidedProjector
from .trackstar import (
    TrackStar,
    TrackStarTransform,
    attribution_scores,
    stack_projected_gradients,
)


def qwen3_tts_encoder() -> type:
    from .qwen3_tts import Qwen3TTSEncoder

    return Qwen3TTSEncoder


EXPERIMENT_ENCODERS = {"qwen3-tts": qwen3_tts_encoder}

__all__ = [
    "EXPERIMENT_ENCODERS",
    "apply_lora",
    "attribution_scores",
    "BlockDiagonalProjector",
    "collect_per_example_gradients",
    "correct_gradients_with_adamw",
    "save_lora_checkpoint",
    "stack_projected_gradients",
    "TrackStar",
    "TrackStarTransform",
    "TwoSidedProjector",
]
