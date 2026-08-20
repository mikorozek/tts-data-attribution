from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml


def _require_keys(values: Any, expected: set[str], section: str) -> dict[str, Any]:
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError(
            f"{section} must contain exactly: {', '.join(sorted(expected))}"
        )
    return values


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)


@dataclass(frozen=True)
class TrainingRunManifest:
    training_set: str
    dtype: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    adam_betas: tuple[float, float]
    adam_epsilon: float
    weight_decay: float
    epochs: int
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        if not self.training_set:
            raise ValueError("training set must not be empty")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        if self.lora_rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.lora_alpha < 1:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be between zero and one")
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if any(not 0.0 <= beta < 1.0 for beta in self.adam_betas):
            raise ValueError("AdamW betas must be between zero and one")
        if self.adam_epsilon <= 0.0:
            raise ValueError("AdamW epsilon must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight decay must not be negative")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if self.seed < 0:
            raise ValueError("training seed must not be negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        values = _require_keys(
            yaml.safe_load(Path(path).read_text(encoding="utf-8")),
            {
                "training_set",
                "dtype",
                "lora",
                "adamw",
                "training",
            },
            "training run manifest",
        )
        lora = _require_keys(values["lora"], {"rank", "alpha", "dropout"}, "lora")
        adamw = _require_keys(
            values["adamw"],
            {"learning_rate", "betas", "epsilon", "weight_decay"},
            "adamw",
        )
        training = _require_keys(
            values["training"],
            {"epochs", "batch_size", "seed"},
            "training",
        )
        betas = adamw["betas"]
        if not isinstance(betas, list) or len(betas) != 2:
            raise ValueError("adamw betas must contain exactly two values")
        training_set = values["training_set"]
        if not isinstance(training_set, str):
            raise ValueError("training set must be a string")
        return cls(
            training_set=training_set,
            dtype=str(values["dtype"]),
            lora_rank=_integer(lora["rank"], "lora rank"),
            lora_alpha=_integer(lora["alpha"], "lora alpha"),
            lora_dropout=_number(lora["dropout"], "lora dropout"),
            learning_rate=_number(adamw["learning_rate"], "learning rate"),
            adam_betas=(
                _number(betas[0], "adamw beta1"),
                _number(betas[1], "adamw beta2"),
            ),
            adam_epsilon=_number(adamw["epsilon"], "adamw epsilon"),
            weight_decay=_number(adamw["weight_decay"], "weight decay"),
            epochs=_integer(training["epochs"], "epochs"),
            batch_size=_integer(training["batch_size"], "batch size"),
            seed=_integer(training["seed"], "training seed"),
        )

    def to_yaml(self, path: str | Path) -> None:
        values = {
            "training_set": self.training_set,
            "dtype": self.dtype,
            "lora": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
            },
            "adamw": {
                "learning_rate": self.learning_rate,
                "betas": list(self.adam_betas),
                "epsilon": self.adam_epsilon,
                "weight_decay": self.weight_decay,
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "seed": self.seed,
            },
        }
        Path(path).write_text(
            yaml.safe_dump(values, sort_keys=False),
            encoding="utf-8",
        )
