"""Function-preserving expansion of Llama/Qwen-style SwiGLU FFNs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SplitRecord:
    layer_index: int
    original_width: int
    expanded_width: int
    original_indices: tuple[int, ...]
    duplicate_indices: tuple[int, ...]
    added_parameters: int

    def to_dict(self) -> dict:
        return asdict(self)


def _linear_like(original: nn.Linear, in_features: int, out_features: int) -> nn.Linear:
    replacement = nn.Linear(
        in_features,
        out_features,
        bias=original.bias is not None,
        device=original.weight.device,
        dtype=original.weight.dtype,
    )
    replacement.train(original.training)
    return replacement


def split_swiglu_layer(model: nn.Module, layer_index: int, indices: list[int]) -> SplitRecord:
    """Duplicate selected FFN neurons while preserving the represented function.

    Each neuron is a coordinated gate_proj row, up_proj row, and down_proj column.
    The incoming rows are copied exactly. The old and copied outgoing columns each
    receive half of the original column, so their summed contribution is unchanged.
    """
    if not indices:
        raise ValueError("At least one neuron index is required")
    unique = tuple(sorted(set(indices)))
    if len(unique) != len(indices):
        raise ValueError("Neuron indices must be unique")

    mlp = model.model.layers[layer_index].mlp
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    width = gate.out_features
    if up.out_features != width or down.in_features != width:
        raise ValueError("The selected MLP does not have coordinated SwiGLU dimensions")
    if unique[0] < 0 or unique[-1] >= width:
        raise IndexError(f"Neuron index outside [0, {width})")

    count = len(unique)
    selected = torch.tensor(unique, device=gate.weight.device, dtype=torch.long)
    new_gate = _linear_like(gate, gate.in_features, width + count)
    new_up = _linear_like(up, up.in_features, width + count)
    new_down = _linear_like(down, width + count, down.out_features)

    with torch.no_grad():
        new_gate.weight.copy_(torch.cat((gate.weight, gate.weight.index_select(0, selected)), dim=0))
        new_up.weight.copy_(torch.cat((up.weight, up.weight.index_select(0, selected)), dim=0))
        if gate.bias is not None:
            new_gate.bias.copy_(torch.cat((gate.bias, gate.bias.index_select(0, selected))))
        if up.bias is not None:
            new_up.bias.copy_(torch.cat((up.bias, up.bias.index_select(0, selected))))

        original_down = down.weight.clone()
        original_down[:, selected] *= 0.5
        duplicate_down = down.weight.index_select(1, selected) * 0.5
        new_down.weight.copy_(torch.cat((original_down, duplicate_down), dim=1))
        if down.bias is not None:
            new_down.bias.copy_(down.bias)

    mlp.gate_proj = new_gate
    mlp.up_proj = new_up
    mlp.down_proj = new_down
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = width + count

    added_parameters = count * (
        gate.in_features + up.in_features + down.out_features
    )
    if gate.bias is not None:
        added_parameters += count
    if up.bias is not None:
        added_parameters += count
    return SplitRecord(
        layer_index=layer_index,
        original_width=width,
        expanded_width=width + count,
        original_indices=unique,
        duplicate_indices=tuple(range(width, width + count)),
        added_parameters=added_parameters,
    )

