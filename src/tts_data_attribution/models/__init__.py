from .qwen3_tts import (
    Qwen3TTSSpeakerReferenceAudioEncoder,
    Qwen3TTSUtteranceAudioEncoder,
)
from .speaker_reference_audio_encoder import SpeakerReferenceAudioEncoder
from .utterance_audio_encoder import UtteranceAudioEncoder

SPEAKER_REFERENCE_AUDIO_ENCODERS: dict[str, type[SpeakerReferenceAudioEncoder]] = {
    "qwen3-tts": Qwen3TTSSpeakerReferenceAudioEncoder
}
UTTERANCE_AUDIO_ENCODERS: dict[str, type[UtteranceAudioEncoder]] = {
    "qwen3-tts": Qwen3TTSUtteranceAudioEncoder
}

__all__ = [
    "SPEAKER_REFERENCE_AUDIO_ENCODERS",
    "UTTERANCE_AUDIO_ENCODERS",
    "SpeakerReferenceAudioEncoder",
    "UtteranceAudioEncoder",
]
