from __future__ import annotations

import math
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
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class TrainingParameter:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("training parameter name must not be empty")
        if not self.shape or any(dimension < 1 for dimension in self.shape):
            raise ValueError("training parameter shape must be positive")
        if not self.dtype:
            raise ValueError("training parameter dtype must not be empty")


@dataclass(frozen=True)
class TrainingRunManifest:
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
    parameters: tuple[TrainingParameter, ...] = ()
    parameter_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        numerical_values = (
            self.lora_dropout,
            self.learning_rate,
            *self.adam_betas,
            self.adam_epsilon,
            self.weight_decay,
        )
        if not all(math.isfinite(value) for value in numerical_values):
            raise ValueError("training numerical values must be finite")
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
        if bool(self.parameters) != bool(self.parameter_groups):
            raise ValueError("training parameter layout must be complete")
        names = [parameter.name for parameter in self.parameters]
        grouped_names = [name for group in self.parameter_groups for name in group]
        if len(names) != len(set(names)):
            raise ValueError("training parameter names must be unique")
        if self.parameters and grouped_names != names:
            raise ValueError("training parameter groups must match parameter order")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("training run manifest must be a mapping")
        expected = {"dtype", "lora", "adamw", "training"}
        if frozenset(values) not in {
            frozenset(expected),
            frozenset(expected | {"parameters"}),
        }:
            raise ValueError(
                "training run manifest must contain dtype, lora, adamw, training, "
                "and optionally parameters"
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
        parameters: tuple[TrainingParameter, ...] = ()
        parameter_groups: tuple[tuple[str, ...], ...] = ()
        if "parameters" in values:
            layout = _require_keys(
                values["parameters"],
                {"ordered", "groups"},
                "parameters",
            )
            ordered = layout["ordered"]
            groups = layout["groups"]
            if not isinstance(ordered, list) or not isinstance(groups, list):
                raise ValueError("training parameter layout must contain lists")
            parsed_parameters = []
            for value in ordered:
                parameter = _require_keys(
                    value, {"name", "shape", "dtype"}, "parameter"
                )
                shape = parameter["shape"]
                if not isinstance(shape, list):
                    raise ValueError("training parameter shape must be a list")
                parsed_parameters.append(
                    TrainingParameter(
                        name=str(parameter["name"]),
                        shape=tuple(
                            _integer(dimension, "training parameter dimension")
                            for dimension in shape
                        ),
                        dtype=str(parameter["dtype"]),
                    )
                )
            if not all(
                isinstance(group, list) and all(isinstance(name, str) for name in group)
                for group in groups
            ):
                raise ValueError("training parameter groups must contain names")
            parameters = tuple(parsed_parameters)
            parameter_groups = tuple(tuple(group) for group in groups)
        return cls(
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
            parameters=parameters,
            parameter_groups=parameter_groups,
        )

    def to_yaml(self, path: str | Path) -> None:
        values: dict[str, object] = {
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
        if self.parameters:
            values["parameters"] = {
                "ordered": [
                    {
                        "name": parameter.name,
                        "shape": list(parameter.shape),
                        "dtype": parameter.dtype,
                    }
                    for parameter in self.parameters
                ],
                "groups": [list(group) for group in self.parameter_groups],
            }
        Path(path).write_text(
            yaml.safe_dump(values, sort_keys=False),
            encoding="utf-8",
        )
