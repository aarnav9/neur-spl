"""Exact flow scores on channel-coarsened two-convolution residual branches.

This is the first neural graph used by the diagnostic. It intentionally models
the residual branch alone: deleting ``bn1`` removes a hidden branch channel and
deleting ``bn2`` removes a branch-output channel before residual addition. Skip
transport is not assigned an arbitrary capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping

import torch
from torch import Tensor

from .flow import Edge, FlowResult, max_flow_min_cut
from .model import CifarResNet


@dataclass(frozen=True)
class ChannelFlowScores:
    """Normalized topological scores aligned to individual output channels."""

    marginal_flow: Tensor
    group_marginal_flow: Tensor
    cut_incident: Tensor
    group_cut_incident: Tensor
    base_flow: float
    channel_group_size: int


@dataclass(frozen=True)
class ResidualBlockFlowResult:
    base_flow: float
    cut_edge_ids: tuple[str, ...]
    hidden: ChannelFlowScores
    output: ChannelFlowScores


@dataclass(frozen=True)
class ResidualGateCapacities:
    """Measured per-channel transport on both paths and after residual merge."""

    branch: Tensor
    skip: Tensor
    merged: Tensor
    examples: int


@dataclass(frozen=True)
class ClassConditionalGateCapacities:
    """Per-commodity gate transport for a residual branch."""

    hidden: Mapping[int, Tensor]
    output: Mapping[int, Tensor]
    examples: Mapping[int, int]


@dataclass(frozen=True)
class ClassConditionedFlowScores:
    """Robust aggregation of normalized deletion damage across commodities."""

    worst_class_marginal: Tensor
    top3_mean_marginal: Tensor
    class_mean_marginal: Tensor
    commodity_count: int
    channel_group_size: int


def _as_channel_matrix(weight_scores: Tensor) -> Tensor:
    if weight_scores.ndim != 4:
        raise ValueError("convolution transport scores must have [out, in, kh, kw] shape")
    if bool((weight_scores < 0).any()):
        raise ValueError("transport capacities must be nonnegative")
    return weight_scores.to(dtype=torch.float64, device="cpu").reshape(
        weight_scores.shape[0], weight_scores.shape[1], -1
    ).sum(dim=2)


def _coarsen(matrix: Tensor, group_size: int) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError("capacity matrix must be two-dimensional")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    rows = ceil(matrix.shape[0] / group_size)
    columns = ceil(matrix.shape[1] / group_size)
    output = torch.zeros(rows, columns, dtype=torch.float64)
    for row in range(rows):
        row_slice = slice(row * group_size, min((row + 1) * group_size, matrix.shape[0]))
        for column in range(columns):
            column_slice = slice(
                column * group_size,
                min((column + 1) * group_size, matrix.shape[1]),
            )
            output[row, column] = matrix[row_slice, column_slice].sum()
    return output


def _build_graph(conv1: Tensor, conv2: Tensor) -> tuple[list[str], list[Edge]]:
    hidden_groups, input_groups = conv1.shape
    output_groups, conv2_hidden_groups = conv2.shape
    if hidden_groups != conv2_hidden_groups:
        raise ValueError("conv1 output groups must match conv2 input groups")

    nodes = ["source", "sink"]
    nodes.extend(f"input:{index}" for index in range(input_groups))
    nodes.extend(f"hidden:{index}" for index in range(hidden_groups))
    nodes.extend(f"output:{index}" for index in range(output_groups))
    internal_total = float(conv1.sum() + conv2.sum())
    boundary = max(1.0, internal_total + 1.0)
    edges: list[Edge] = []
    for index in range(input_groups):
        edges.append(Edge(f"source:{index}", "source", f"input:{index}", boundary))
    for hidden in range(hidden_groups):
        for source in range(input_groups):
            capacity = float(conv1[hidden, source])
            if capacity > 0.0:
                edges.append(
                    Edge(
                        f"conv1:{hidden}:{source}",
                        f"input:{source}",
                        f"hidden:{hidden}",
                        capacity,
                    )
                )
    for output in range(output_groups):
        for hidden in range(hidden_groups):
            capacity = float(conv2[output, hidden])
            if capacity > 0.0:
                edges.append(
                    Edge(
                        f"conv2:{output}:{hidden}",
                        f"hidden:{hidden}",
                        f"output:{output}",
                        capacity,
                    )
                )
    for index in range(output_groups):
        edges.append(Edge(f"sink:{index}", f"output:{index}", "sink", boundary))
    return nodes, edges


def _build_skip_aware_graph(
    conv1: Tensor,
    conv2: Tensor,
    branch: Tensor,
    skip: Tensor,
    merged: Tensor,
) -> tuple[list[str], list[Edge]]:
    """Build a residual graph in which measured skip and branch flow compete."""

    hidden_groups, input_groups = conv1.shape
    output_groups, conv2_hidden_groups = conv2.shape
    if hidden_groups != conv2_hidden_groups:
        raise ValueError("conv1 output groups must match conv2 input groups")
    for name, capacity in (("branch", branch), ("skip", skip), ("merged", merged)):
        if capacity.ndim != 1 or capacity.numel() != output_groups:
            raise ValueError(f"{name} capacities must align to output groups")
        if bool((capacity < 0).any()) or not bool(torch.isfinite(capacity).all()):
            raise ValueError(f"{name} capacities must be finite and nonnegative")

    nodes = ["source", "sink"]
    nodes.extend(f"input:{index}" for index in range(input_groups))
    nodes.extend(f"hidden:{index}" for index in range(hidden_groups))
    nodes.extend(f"branch:{index}" for index in range(output_groups))
    nodes.extend(f"merged:{index}" for index in range(output_groups))
    internal_total = float(
        conv1.sum() + conv2.sum() + branch.sum() + skip.sum() + merged.sum()
    )
    boundary = max(1.0, internal_total + 1.0)
    edges: list[Edge] = []
    for index in range(input_groups):
        edges.append(Edge(f"source:{index}", "source", f"input:{index}", boundary))
    for hidden in range(hidden_groups):
        for source in range(input_groups):
            capacity = float(conv1[hidden, source])
            if capacity > 0.0:
                edges.append(
                    Edge(
                        f"conv1:{hidden}:{source}",
                        f"input:{source}",
                        f"hidden:{hidden}",
                        capacity,
                    )
                )
    for output in range(output_groups):
        for hidden in range(hidden_groups):
            capacity = float(conv2[output, hidden])
            if capacity > 0.0:
                edges.append(
                    Edge(
                        f"conv2:{output}:{hidden}",
                        f"hidden:{hidden}",
                        f"branch:{output}",
                        capacity,
                    )
                )
        if float(branch[output]) > 0.0:
            edges.append(
                Edge(
                    f"main:{output}",
                    f"branch:{output}",
                    f"merged:{output}",
                    float(branch[output]),
                )
            )
        if float(skip[output]) > 0.0:
            edges.append(
                Edge(
                    f"skip:{output}",
                    "source",
                    f"merged:{output}",
                    float(skip[output]),
                )
            )
        if float(merged[output]) > 0.0:
            edges.append(
                Edge(
                    f"merged:{output}",
                    f"merged:{output}",
                    "sink",
                    float(merged[output]),
                )
            )
    return nodes, edges


def _without_node(edges: list[Edge], node: str) -> list[Edge]:
    return [edge for edge in edges if edge.source != node and edge.target != node]


def _incident_cut_capacities(
    result: FlowResult,
    edges: list[Edge],
    hidden_groups: int,
    output_groups: int,
) -> tuple[Tensor, Tensor]:
    cut = set(result.cut_edge_ids)
    hidden = torch.zeros(hidden_groups, dtype=torch.float64)
    output = torch.zeros(output_groups, dtype=torch.float64)
    for edge in edges:
        if edge.edge_id not in cut:
            continue
        if edge.edge_id.startswith("conv1:"):
            hidden_index = int(edge.edge_id.split(":")[1])
            hidden[hidden_index] += edge.capacity
        elif edge.edge_id.startswith("conv2:"):
            _, output_index, hidden_index = edge.edge_id.split(":")
            hidden[int(hidden_index)] += edge.capacity
            output[int(output_index)] += edge.capacity
    return hidden, output


def _skip_aware_incident_cut_capacities(
    result: FlowResult,
    edges: list[Edge],
    hidden_groups: int,
    output_groups: int,
) -> tuple[Tensor, Tensor]:
    cut = set(result.cut_edge_ids)
    hidden = torch.zeros(hidden_groups, dtype=torch.float64)
    output = torch.zeros(output_groups, dtype=torch.float64)
    for edge in edges:
        if edge.edge_id not in cut:
            continue
        if edge.edge_id.startswith("conv1:"):
            hidden[int(edge.edge_id.split(":")[1])] += edge.capacity
        elif edge.edge_id.startswith("conv2:"):
            _, output_index, hidden_index = edge.edge_id.split(":")
            hidden[int(hidden_index)] += edge.capacity
            output[int(output_index)] += edge.capacity
        elif edge.edge_id.startswith("main:"):
            output[int(edge.edge_id.split(":")[1])] += edge.capacity
    return hidden, output


def _expand_group_values(
    group_values: Tensor,
    local_channel_capacity: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    allocated = torch.zeros_like(local_channel_capacity, dtype=torch.float64)
    broadcast = torch.zeros_like(local_channel_capacity, dtype=torch.float64)
    for group, value in enumerate(group_values):
        start = group * group_size
        end = min(start + group_size, local_channel_capacity.numel())
        local = local_channel_capacity[start:end].to(dtype=torch.float64)
        broadcast[start:end] = value
        total = float(local.sum())
        if total > 0.0:
            allocated[start:end] = value * local / total
        else:
            allocated[start:end] = value / max(1, end - start)
    return allocated, broadcast


def residual_branch_flow_scores(
    conv1_weight_scores: Tensor,
    conv2_weight_scores: Tensor,
    *,
    group_size: int = 16,
) -> ResidualBlockFlowResult:
    """Compute exact coarsened node-deletion damage for one residual branch."""

    conv1_channels = _as_channel_matrix(conv1_weight_scores)
    conv2_channels = _as_channel_matrix(conv2_weight_scores)
    if conv1_channels.shape[0] != conv2_channels.shape[1]:
        raise ValueError("conv1 output channels must match conv2 input channels")
    conv1 = _coarsen(conv1_channels, group_size)
    conv2 = _coarsen(conv2_channels, group_size)
    nodes, edges = _build_graph(conv1, conv2)
    baseline = max_flow_min_cut(nodes, edges, "source", "sink")

    hidden_damage = torch.zeros(conv1.shape[0], dtype=torch.float64)
    for group in range(conv1.shape[0]):
        without = _without_node(edges, f"hidden:{group}")
        result = max_flow_min_cut(nodes, without, "source", "sink")
        hidden_damage[group] = max(0.0, baseline.value - result.value)

    output_damage = torch.zeros(conv2.shape[0], dtype=torch.float64)
    for group in range(conv2.shape[0]):
        without = _without_node(edges, f"output:{group}")
        result = max_flow_min_cut(nodes, without, "source", "sink")
        output_damage[group] = max(0.0, baseline.value - result.value)

    hidden_cut, output_cut = _incident_cut_capacities(
        baseline, edges, conv1.shape[0], conv2.shape[0]
    )
    normalizer = baseline.value if baseline.value > 0.0 else 1.0
    hidden_allocated, hidden_broadcast = _expand_group_values(
        hidden_damage / normalizer,
        conv1_channels.sum(dim=1),
        group_size,
    )
    output_allocated, output_broadcast = _expand_group_values(
        output_damage / normalizer,
        conv2_channels.sum(dim=1),
        group_size,
    )
    hidden_cut_allocated, hidden_cut_broadcast = _expand_group_values(
        hidden_cut / normalizer,
        conv1_channels.sum(dim=1),
        group_size,
    )
    output_cut_allocated, output_cut_broadcast = _expand_group_values(
        output_cut / normalizer,
        conv2_channels.sum(dim=1),
        group_size,
    )
    return ResidualBlockFlowResult(
        base_flow=baseline.value,
        cut_edge_ids=baseline.cut_edge_ids,
        hidden=ChannelFlowScores(
            hidden_allocated,
            hidden_broadcast,
            hidden_cut_allocated,
            hidden_cut_broadcast,
            baseline.value,
            group_size,
        ),
        output=ChannelFlowScores(
            output_allocated,
            output_broadcast,
            output_cut_allocated,
            output_cut_broadcast,
            baseline.value,
            group_size,
        ),
    )


def skip_aware_residual_branch_flow_scores(
    conv1_weight_scores: Tensor,
    conv2_weight_scores: Tensor,
    gate_capacities: ResidualGateCapacities,
    *,
    group_size: int = 16,
) -> ResidualBlockFlowResult:
    """Compute node-deletion flow damage with a measured residual bypass."""

    conv1_channels = _as_channel_matrix(conv1_weight_scores)
    conv2_channels = _as_channel_matrix(conv2_weight_scores)
    if conv1_channels.shape[0] != conv2_channels.shape[1]:
        raise ValueError("conv1 output channels must match conv2 input channels")
    output_channels = conv2_channels.shape[0]
    for name, capacity in (
        ("branch", gate_capacities.branch),
        ("skip", gate_capacities.skip),
        ("merged", gate_capacities.merged),
    ):
        if capacity.ndim != 1 or capacity.numel() != output_channels:
            raise ValueError(f"{name} gate capacities must align to output channels")
    conv1 = _coarsen(conv1_channels, group_size)
    conv2 = _coarsen(conv2_channels, group_size)
    branch = _coarsen(gate_capacities.branch.reshape(-1, 1), group_size).flatten()
    skip = _coarsen(gate_capacities.skip.reshape(-1, 1), group_size).flatten()
    merged = _coarsen(gate_capacities.merged.reshape(-1, 1), group_size).flatten()
    nodes, edges = _build_skip_aware_graph(conv1, conv2, branch, skip, merged)
    baseline = max_flow_min_cut(nodes, edges, "source", "sink")

    hidden_damage = torch.zeros(conv1.shape[0], dtype=torch.float64)
    for group in range(conv1.shape[0]):
        result = max_flow_min_cut(
            nodes, _without_node(edges, f"hidden:{group}"), "source", "sink"
        )
        hidden_damage[group] = max(0.0, baseline.value - result.value)

    output_damage = torch.zeros(conv2.shape[0], dtype=torch.float64)
    for group in range(conv2.shape[0]):
        result = max_flow_min_cut(
            nodes, _without_node(edges, f"branch:{group}"), "source", "sink"
        )
        output_damage[group] = max(0.0, baseline.value - result.value)

    hidden_cut, output_cut = _skip_aware_incident_cut_capacities(
        baseline, edges, conv1.shape[0], conv2.shape[0]
    )
    normalizer = baseline.value if baseline.value > 0.0 else 1.0
    hidden_allocated, hidden_broadcast = _expand_group_values(
        hidden_damage / normalizer, conv1_channels.sum(dim=1), group_size
    )
    output_allocated, output_broadcast = _expand_group_values(
        output_damage / normalizer, conv2_channels.sum(dim=1), group_size
    )
    hidden_cut_allocated, hidden_cut_broadcast = _expand_group_values(
        hidden_cut / normalizer, conv1_channels.sum(dim=1), group_size
    )
    output_cut_allocated, output_cut_broadcast = _expand_group_values(
        output_cut / normalizer, conv2_channels.sum(dim=1), group_size
    )
    return ResidualBlockFlowResult(
        base_flow=baseline.value,
        cut_edge_ids=baseline.cut_edge_ids,
        hidden=ChannelFlowScores(
            hidden_allocated,
            hidden_broadcast,
            hidden_cut_allocated,
            hidden_cut_broadcast,
            baseline.value,
            group_size,
        ),
        output=ChannelFlowScores(
            output_allocated,
            output_broadcast,
            output_cut_allocated,
            output_cut_broadcast,
            baseline.value,
            group_size,
        ),
    )


def class_conditioned_residual_branch_flow_scores(
    conv1_weight_scores: Tensor,
    conv2_weight_scores: Tensor,
    gate_capacities: ClassConditionalGateCapacities,
    *,
    group_size: int = 16,
) -> tuple[ClassConditionedFlowScores, ClassConditionedFlowScores]:
    """Score each deletion across coarse-class-conditioned flow commodities."""

    conv1_channels = _as_channel_matrix(conv1_weight_scores)
    conv2_channels = _as_channel_matrix(conv2_weight_scores)
    if conv1_channels.shape[0] != conv2_channels.shape[1]:
        raise ValueError("conv1 output channels must match conv2 input channels")
    commodities = set(gate_capacities.hidden)
    if commodities != set(gate_capacities.output) or commodities != set(
        gate_capacities.examples
    ):
        raise ValueError("class-conditioned gate commodities do not align")
    if len(commodities) < 2:
        raise ValueError("class-conditioned flow requires at least two commodities")
    if any(gate_capacities.examples[commodity] < 1 for commodity in commodities):
        raise ValueError("every class-conditioned commodity needs observations")

    total_examples = sum(gate_capacities.examples.values())
    hidden_overall = sum(
        gate_capacities.hidden[commodity] * gate_capacities.examples[commodity]
        for commodity in commodities
    ) / total_examples
    output_overall = sum(
        gate_capacities.output[commodity] * gate_capacities.examples[commodity]
        for commodity in commodities
    ) / total_examples
    if hidden_overall.numel() != conv1_channels.shape[0]:
        raise ValueError("hidden class capacities do not align to conv1 outputs")
    if output_overall.numel() != conv2_channels.shape[0]:
        raise ValueError("output class capacities do not align to conv2 outputs")
    epsilon = torch.finfo(torch.float64).eps
    hidden_values: list[Tensor] = []
    output_values: list[Tensor] = []
    for commodity in sorted(commodities):
        hidden_ratio = gate_capacities.hidden[commodity].to(torch.float64) / (
            hidden_overall.to(torch.float64) + epsilon
        )
        output_ratio = gate_capacities.output[commodity].to(torch.float64) / (
            output_overall.to(torch.float64) + epsilon
        )
        class_conv1 = conv1_channels * hidden_ratio.reshape(-1, 1)
        class_conv2 = conv2_channels * output_ratio.reshape(-1, 1)
        result = residual_branch_flow_scores(
            class_conv1.unsqueeze(-1).unsqueeze(-1),
            class_conv2.unsqueeze(-1).unsqueeze(-1),
            group_size=group_size,
        )
        hidden_values.append(result.hidden.group_marginal_flow[::group_size])
        output_values.append(result.output.group_marginal_flow[::group_size])

    def aggregate(values: list[Tensor]) -> ClassConditionedFlowScores:
        stacked = torch.stack(values)
        top_count = min(3, stacked.shape[0])
        return ClassConditionedFlowScores(
            worst_class_marginal=stacked.max(dim=0).values,
            top3_mean_marginal=stacked.topk(top_count, dim=0).values.mean(dim=0),
            class_mean_marginal=stacked.mean(dim=0),
            commodity_count=stacked.shape[0],
            channel_group_size=group_size,
        )

    return aggregate(hidden_values), aggregate(output_values)


def model_residual_flow_scores(
    model: CifarResNet,
    weight_scores: Mapping[str, Tensor],
    *,
    group_size: int = 16,
) -> dict[str, ChannelFlowScores]:
    """Compute branch-local flow scores for every BasicBlock convolution."""

    output: dict[str, ChannelFlowScores] = {}
    for layer_index, layer in enumerate(
        (model.layer1, model.layer2, model.layer3, model.layer4), start=1
    ):
        for block_index, _block in enumerate(layer):
            prefix = f"layer{layer_index}.{block_index}"
            conv1_name = f"{prefix}.conv1"
            conv2_name = f"{prefix}.conv2"
            if conv1_name not in weight_scores or conv2_name not in weight_scores:
                raise KeyError(f"missing transport scores for residual branch {prefix}")
            result = residual_branch_flow_scores(
                weight_scores[conv1_name],
                weight_scores[conv2_name],
                group_size=group_size,
            )
            output[conv1_name] = result.hidden
            output[conv2_name] = result.output
    return output


def model_skip_aware_flow_scores(
    model: CifarResNet,
    weight_scores: Mapping[str, Tensor],
    gate_capacities: Mapping[str, ResidualGateCapacities],
    *,
    group_size: int = 16,
) -> dict[str, ChannelFlowScores]:
    """Compute measured-skip flow scores for every BasicBlock convolution."""

    output: dict[str, ChannelFlowScores] = {}
    for layer_index, layer in enumerate(
        (model.layer1, model.layer2, model.layer3, model.layer4), start=1
    ):
        for block_index, _block in enumerate(layer):
            prefix = f"layer{layer_index}.{block_index}"
            conv1_name = f"{prefix}.conv1"
            conv2_name = f"{prefix}.conv2"
            if conv1_name not in weight_scores or conv2_name not in weight_scores:
                raise KeyError(f"missing transport scores for residual branch {prefix}")
            if prefix not in gate_capacities:
                raise KeyError(f"missing measured gate capacities for residual branch {prefix}")
            result = skip_aware_residual_branch_flow_scores(
                weight_scores[conv1_name],
                weight_scores[conv2_name],
                gate_capacities[prefix],
                group_size=group_size,
            )
            output[conv1_name] = result.hidden
            output[conv2_name] = result.output
    return output


def model_class_conditioned_flow_scores(
    model: CifarResNet,
    weight_scores: Mapping[str, Tensor],
    gate_capacities: Mapping[str, ClassConditionalGateCapacities],
    *,
    group_size: int = 16,
) -> dict[str, ClassConditionedFlowScores]:
    """Compute coarse-class commodity scores for every residual convolution."""

    output: dict[str, ClassConditionedFlowScores] = {}
    for layer_index, layer in enumerate(
        (model.layer1, model.layer2, model.layer3, model.layer4), start=1
    ):
        for block_index, _block in enumerate(layer):
            prefix = f"layer{layer_index}.{block_index}"
            conv1_name = f"{prefix}.conv1"
            conv2_name = f"{prefix}.conv2"
            if conv1_name not in weight_scores or conv2_name not in weight_scores:
                raise KeyError(f"missing transport scores for residual branch {prefix}")
            if prefix not in gate_capacities:
                raise KeyError(f"missing class-conditioned gates for residual branch {prefix}")
            hidden, branch_output = class_conditioned_residual_branch_flow_scores(
                weight_scores[conv1_name],
                weight_scores[conv2_name],
                gate_capacities[prefix],
                group_size=group_size,
            )
            output[conv1_name] = hidden
            output[conv2_name] = branch_output
    return output
