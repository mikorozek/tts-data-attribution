from __future__ import annotations

import re
from collections import defaultdict

import torch
from torch import nn

from ..projection import BlockDiagonalProjector

_PARAMETER_PATTERN = re.compile(
    r"^.*?(?P<predictor>code_predictor\.)?model\.layers\."
    r"(?P<layer>\d+)\.(?P<family>self_attn|mlp)\."
    r"(?P<module>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"lora_(?P<factor>A|B)\.(?P<adapter>[^.]+)\.weight$"
)
_ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
_MODULE_ORDER = {
    name: index
    for modules in (_ATTENTION_MODULES, _MLP_MODULES)
    for index, name in enumerate(modules)
}
_EXPECTED_LAYER_PARAMETERS = {
    (family, module_name, factor)
    for family, modules in (
        ("self_attn", _ATTENTION_MODULES),
        ("mlp", _MLP_MODULES),
    )
    for module_name in modules
    for factor in ("A", "B")
}


class Qwen3TTSGradientProjector:
    def __init__(
        self,
        module: nn.Module,
        *,
        talker_output_dimension: int,
        code_predictor_output_dimension: int,
        talker_layers_per_block: int,
        code_predictor_layers_per_block: int,
        seed: int,
        adapter_name: str = "default",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        output_dimensions = {
            "talker": talker_output_dimension,
            "code_predictor": code_predictor_output_dimension,
        }
        block_sizes = {
            "talker": talker_layers_per_block,
            "code_predictor": code_predictor_layers_per_block,
        }
        if any(size < 1 for size in block_sizes.values()):
            raise ValueError("layers per projection block must be positive")
        parameters = {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        if not parameters:
            raise ValueError("module has no trainable parameters")
        devices = {parameter.device for parameter in parameters.values()}
        if len(devices) != 1:
            raise ValueError("trainable parameters must be on one device")
        device = next(iter(devices))

        parsed = []
        observed_by_layer = defaultdict(set)
        parameter_shapes = {}
        for name, parameter in parameters.items():
            match = _PARAMETER_PATTERN.fullmatch(name)
            if match is None:
                raise ValueError(f"unsupported Qwen3-TTS trainable parameter: {name}")
            if match["adapter"] != adapter_name:
                raise ValueError(f"unexpected LoRA adapter in parameter: {name}")
            family = match["family"]
            module_name = match["module"]
            if family == "self_attn" and module_name not in _ATTENTION_MODULES:
                raise ValueError(f"invalid attention LoRA parameter: {name}")
            if family == "mlp" and module_name not in _MLP_MODULES:
                raise ValueError(f"invalid MLP LoRA parameter: {name}")

            stack = "code_predictor" if match["predictor"] else "talker"
            layer = int(match["layer"])
            factor = match["factor"]
            parsed.append((stack, layer, family, module_name, factor, name))
            observed_by_layer[(stack, layer)].add((family, module_name, factor))
            parameter_shapes[name] = tuple(parameter.shape)

        maximum_layers = {}
        for stack in block_sizes:
            layers = sorted(
                layer
                for observed_stack, layer in observed_by_layer
                if observed_stack == stack
            )
            if not layers or layers != list(range(layers[-1] + 1)):
                raise ValueError(f"{stack} LoRA layers must be consecutive from zero")
            for layer in layers:
                if observed_by_layer[(stack, layer)] != _EXPECTED_LAYER_PARAMETERS:
                    raise ValueError(
                        f"incomplete LoRA parameters for {stack} layer {layer}"
                    )
            maximum_layers[stack] = layers[-1]

        grouped = defaultdict(list)
        for stack, layer, family, module_name, factor, name in parsed:
            block_size = block_sizes[stack]
            block_start = layer // block_size * block_size
            block_family = "attention" if family == "self_attn" else "mlp"
            grouped[(stack, block_start, block_family)].append(
                (layer, _MODULE_ORDER[module_name], factor, name)
            )

        stack_order = {"talker": 0, "code_predictor": 1}
        family_order = {"attention": 0, "mlp": 1}
        group_keys = sorted(
            grouped,
            key=lambda key: (stack_order[key[0]], key[1], family_order[key[2]]),
        )
        block_parameter_names = {}
        block_projectors = {}
        parameter_seed_offset = 0
        for stack, block_start, family in group_keys:
            block_end = min(
                block_start + block_sizes[stack] - 1,
                maximum_layers[stack],
            )
            block_name = f"{stack}.layers.{block_start}-{block_end}.{family}"
            entries = sorted(
                grouped[(stack, block_start, family)],
                key=lambda entry: (entry[0], entry[1], entry[2]),
            )
            names = tuple(entry[3] for entry in entries)
            shapes = {name: parameter_shapes[name] for name in names}
            block_parameter_names[block_name] = names
            block_projectors[block_name] = BlockDiagonalProjector(
                shapes,
                output_dimensions[stack],
                seed=seed + parameter_seed_offset,
                device=device,
                dtype=dtype,
            )
            parameter_seed_offset += len(names)

        self.parameter_shapes = parameter_shapes
        self.block_names = tuple(block_parameter_names)
        self.block_parameter_names = block_parameter_names
        self.output_dimensions = output_dimensions
        self._block_projectors = block_projectors
        self._dtype = dtype

    def __call__(
        self,
        gradients: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if gradients.keys() != self.parameter_shapes.keys():
            raise ValueError("gradient names do not match the projection parameters")
        for name, gradient in gradients.items():
            if tuple(gradient.shape) != self.parameter_shapes[name]:
                raise ValueError(f"gradient shape does not match parameter {name}")

        projected = {}
        for block_name in self.block_names:
            matrices = {
                name: gradients[name].to(dtype=self._dtype)
                for name in self.block_parameter_names[block_name]
            }
            projected[block_name] = self._block_projectors[block_name](matrices)
        return projected
