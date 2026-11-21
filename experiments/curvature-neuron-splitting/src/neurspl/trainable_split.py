"""A parameter-efficient trainable view of selected split SwiGLU neurons."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as functional
from torch import nn

from neurspl.split import SplitRecord, _linear_like, split_swiglu_layer


class TrainableSplitMLP(nn.Module):
    """Freeze an MLP except for both members of selected sibling pairs.

    The selected neurons are physically removed from ``base_mlp`` and represented
    by the small trainable matrices below. Consequently this module has exactly
    the same parameter count and forward function as the expanded MLP; it does
    not retain a second, hidden copy of the trainable coordinates.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        gate_rows: torch.Tensor,
        up_rows: torch.Tensor,
        down_columns: torch.Tensor,
        gate_bias: torch.Tensor | None = None,
        up_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.gate_rows = nn.Parameter(gate_rows)
        self.up_rows = nn.Parameter(up_rows)
        self.down_columns = nn.Parameter(down_columns)
        self.gate_bias = (
            nn.Parameter(gate_bias) if gate_bias is not None else None
        )
        self.up_bias = nn.Parameter(up_bias) if up_bias is not None else None

    @classmethod
    def from_expanded_mlp(
        cls, mlp: nn.Module, selected_positions: Iterable[int]
    ) -> "TrainableSplitMLP":
        positions = tuple(int(value) for value in selected_positions)
        if not positions:
            raise ValueError("At least one selected position is required")
        if len(set(positions)) != len(positions):
            raise ValueError("Selected positions must be unique")

        gate = mlp.gate_proj
        up = mlp.up_proj
        down = mlp.down_proj
        width = gate.out_features
        if up.out_features != width or down.in_features != width:
            raise ValueError("The MLP does not have coordinated SwiGLU dimensions")
        if min(positions) < 0 or max(positions) >= width:
            raise IndexError(f"Selected position outside [0, {width})")

        selected = torch.tensor(positions, device=gate.weight.device, dtype=torch.long)
        selected_set = set(positions)
        kept_positions = [index for index in range(width) if index not in selected_set]
        kept = torch.tensor(kept_positions, device=gate.weight.device, dtype=torch.long)

        gate_rows = gate.weight.index_select(0, selected).detach().clone()
        up_rows = up.weight.index_select(0, selected).detach().clone()
        down_columns = down.weight.index_select(1, selected).detach().clone()
        gate_bias = (
            gate.bias.index_select(0, selected).detach().clone()
            if gate.bias is not None
            else None
        )
        up_bias = (
            up.bias.index_select(0, selected).detach().clone()
            if up.bias is not None
            else None
        )

        reduced_width = len(kept_positions)
        new_gate = _linear_like(gate, gate.in_features, reduced_width)
        new_up = _linear_like(up, up.in_features, reduced_width)
        new_down = _linear_like(down, reduced_width, down.out_features)
        with torch.no_grad():
            new_gate.weight.copy_(gate.weight.index_select(0, kept))
            new_up.weight.copy_(up.weight.index_select(0, kept))
            new_down.weight.copy_(down.weight.index_select(1, kept))
            if gate.bias is not None:
                new_gate.bias.copy_(gate.bias.index_select(0, kept))
            if up.bias is not None:
                new_up.bias.copy_(up.bias.index_select(0, kept))
            if down.bias is not None:
                new_down.bias.copy_(down.bias)

        mlp.gate_proj = new_gate
        mlp.up_proj = new_up
        mlp.down_proj = new_down
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = reduced_width
        mlp.requires_grad_(False)
        return cls(
            mlp,
            gate_rows,
            up_rows,
            down_columns,
            gate_bias,
            up_bias,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base_output = self.base_mlp(values)
        gate = functional.linear(values, self.gate_rows, self.gate_bias)
        up = functional.linear(values, self.up_rows, self.up_bias)
        selected_output = functional.linear(
            self.base_mlp.act_fn(gate) * up, self.down_columns
        )
        return base_output + selected_output

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def trainable_state(self) -> dict[str, torch.Tensor]:
        names = ("gate_rows", "up_rows", "down_columns", "gate_bias", "up_bias")
        return {
            name: value.detach().cpu().clone()
            for name in names
            if (value := getattr(self, name)) is not None
        }

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        expected = {
            name
            for name in ("gate_rows", "up_rows", "down_columns", "gate_bias", "up_bias")
            if getattr(self, name) is not None
        }
        if set(state) != expected:
            raise ValueError(
                f"Adapter state keys do not match: saved={sorted(state)}, "
                f"expected={sorted(expected)}"
            )
        with torch.no_grad():
            for name in expected:
                destination = getattr(self, name)
                source = state[name].to(
                    device=destination.device, dtype=destination.dtype
                )
                if source.shape != destination.shape:
                    raise ValueError(
                        f"Adapter tensor {name} has shape {tuple(source.shape)}; "
                        f"expected {tuple(destination.shape)}"
                    )
                destination.copy_(source)


def _set_initial_separation(
    mlp: nn.Module,
    record: SplitRecord,
    gate_directions: torch.Tensor,
    up_directions: torch.Tensor,
    epsilon: float,
) -> None:
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    original = torch.tensor(
        record.original_indices, device=mlp.gate_proj.weight.device, dtype=torch.long
    )
    duplicate = torch.tensor(
        record.duplicate_indices, device=mlp.gate_proj.weight.device, dtype=torch.long
    )
    gate_directions = gate_directions.to(
        device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype
    )
    up_directions = up_directions.to(
        device=mlp.up_proj.weight.device, dtype=mlp.up_proj.weight.dtype
    )
    expected = (len(record.original_indices), mlp.gate_proj.in_features)
    if tuple(gate_directions.shape) != expected or tuple(up_directions.shape) != expected:
        raise ValueError(
            f"Direction tensors must both have shape {expected}; got "
            f"{tuple(gate_directions.shape)} and {tuple(up_directions.shape)}"
        )
    with torch.no_grad():
        gate_centers = mlp.gate_proj.weight.index_select(0, original).clone()
        up_centers = mlp.up_proj.weight.index_select(0, original).clone()
        mlp.gate_proj.weight[original] = gate_centers + epsilon * gate_directions
        mlp.gate_proj.weight[duplicate] = gate_centers - epsilon * gate_directions
        mlp.up_proj.weight[original] = up_centers + epsilon * up_directions
        mlp.up_proj.weight[duplicate] = up_centers - epsilon * up_directions


def install_trainable_split(
    model: nn.Module,
    layer_index: int,
    indices: list[int],
    gate_directions: torch.Tensor | None = None,
    up_directions: torch.Tensor | None = None,
    epsilon: float = 0.0,
) -> tuple[TrainableSplitMLP, SplitRecord]:
    """Expand a layer and expose only the two members of each pair for training."""
    model.requires_grad_(False)
    record = split_swiglu_layer(model, layer_index, indices)
    mlp = model.model.layers[layer_index].mlp
    if (gate_directions is None) != (up_directions is None):
        raise ValueError("Gate and up directions must be supplied together")
    if gate_directions is None:
        if epsilon != 0.0:
            raise ValueError("Nonzero epsilon requires direction tensors")
    else:
        _set_initial_separation(
            mlp, record, gate_directions, up_directions, epsilon
        )
    selected_positions = (*record.original_indices, *record.duplicate_indices)
    adapter = TrainableSplitMLP.from_expanded_mlp(mlp, selected_positions)
    model.model.layers[layer_index].mlp = adapter
    return adapter, record


def save_adapter_checkpoint(
    path: Path, adapter: TrainableSplitMLP, metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "metadata": metadata,
            "adapter_state": adapter.trainable_state(),
        },
        temporary,
    )
    temporary.replace(path)


def load_adapter_checkpoint(
    model: nn.Module, path: Path
) -> tuple[TrainableSplitMLP, SplitRecord, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema_version") != 1:
        raise ValueError(f"Unsupported adapter schema in {path}")
    metadata = checkpoint["metadata"]
    layer_index = int(metadata["layer"])
    indices = [int(value) for value in metadata["selected_neurons"]]
    adapter, record = install_trainable_split(model, layer_index, indices)
    adapter.load_trainable_state(checkpoint["adapter_state"])
    return adapter, record, metadata


def split_metadata(record: SplitRecord) -> dict[str, Any]:
    return asdict(record)
