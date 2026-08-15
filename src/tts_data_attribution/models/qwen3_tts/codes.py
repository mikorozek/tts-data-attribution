from __future__ import annotations

from pathlib import Path

import torch

FRAMES_PER_SECOND = 12.5
CODEBOOK_COUNT = 16
FRAME_COUNT_TOLERANCE = 5


class CodesEncoder:
    def __init__(self, tokenizer_path: str | Path, device: str) -> None:
        from qwen_tts import Qwen3TTSTokenizer

        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(str(tokenizer_path), device_map=device)

    def encode(self, audio_paths: list[str]) -> list[list[list[int]]]:
        encoded = self.tokenizer.encode(audio_paths)
        return [
            validate_codes(codes, expected_frame_count(audio_path), audio_path)
            for codes, audio_path in zip(encoded.audio_codes, audio_paths, strict=True)
        ]


def validate_codes(
    codes: torch.Tensor,
    expected_frames: int,
    audio_path: str,
) -> list[list[int]]:
    if codes.ndim != 2 or codes.shape[1] != CODEBOOK_COUNT:
        raise ValueError(f"unexpected code shape {tuple(codes.shape)} for {audio_path}")
    if abs(codes.shape[0] - expected_frames) > FRAME_COUNT_TOLERANCE:
        raise ValueError(
            f"{codes.shape[0]} frames for {audio_path}, expected about {expected_frames}"
        )
    return codes.cpu().tolist()


def expected_frame_count(audio_path: str) -> int:
    import soundfile

    info = soundfile.info(audio_path)
    return round(info.frames / info.samplerate * FRAMES_PER_SECOND)
