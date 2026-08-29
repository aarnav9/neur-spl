"""Global class-conditioned flow graph for a residual network.

Unlike the branch-local diagnostic, this graph follows paths from the stem to
coarse-class classifier sinks. Residual additions have separately measured
main, skip, and merged gate capacities. Commodities share a topology but use
their own measured gate demands and classifier terminals; deletion damage is
normalized inside each commodity before robust aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .flow import Edge, max_flow_min_cut
from .model import CifarResNet


@dataclass(frozen=True)
class GlobalClassGateCapacities:
    by_gate: Mapping[str, Mapping[int, Tensor]]
    examples: Mapping[int, int]
    fine_to_coarse: tuple[int, ...]


@dataclass(frozen=True)
class GlobalClassFlowScores:
    worst_class_marginal: Tensor
    top3_mean_marginal: Tensor
    class_mean_marginal: Tensor
    commodity_count: int
    channel_group_size: int


def _conv_matrix(score: Tensor) -> Tensor:
    if score.ndim != 4:
        raise ValueError("convolution transport must have [out, in, kh, kw] shape")
    if bool((score < 0).any()) or not bool(torch.isfinite(score).all()):
        raise ValueError("convolution transport must be finite and nonnegative")
    return score.to(dtype=torch.float64, device="cpu").flatten(start_dim=2).sum(dim=2)


def _coarsen_matrix(matrix: Tensor, group_size: int) -> Tensor:
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


def _coarsen_vector(vector: Tensor, group_size: int) -> Tensor:
    if vector.ndim != 1:
        raise ValueError("gate capacity must be a vector")
    return torch.stack(
        [
            vector[start : start + group_size].to(torch.float64).sum()
            for start in range(0, vector.numel(), group_size)
        ]
    )


def _coarsen_columns(matrix: Tensor, group_size: int) -> Tensor:
    return torch.stack(
        [
            matrix[:, start : start + group_size].to(torch.float64).sum(dim=1)
            for start in range(0, matrix.shape[1], group_size)
        ],
        dim=1,
    )


def _without_nodes(edges: Sequence[Edge], removed: set[str]) -> list[Edge]:
    return [
        edge
        for edge in edges
        if edge.source not in removed and edge.target not in removed
    ]


def _build_commodity_graph(
    model: CifarResNet,
    matrices: Mapping[str, Tensor],
    gate_capacities: GlobalClassGateCapacities,
    commodity: int,
    group_size: int,
) -> tuple[list[str], list[Edge], dict[str, list[set[str]]]]:
    nodes = ["source", "sink"]
    edges: list[Edge] = []
    deletion_nodes: dict[str, list[set[str]]] = {}

    def add_node(name: str) -> None:
        nodes.append(name)

    def add_edge(edge_id: str, source: str, target: str, capacity: float) -> None:
        if capacity > 0.0:
            edges.append(Edge(edge_id, source, target, capacity))

    def gate(name: str) -> Tensor:
        if name not in gate_capacities.by_gate:
            raise KeyError(f"missing global gate capacity {name!r}")
        values = gate_capacities.by_gate[name]
        if commodity not in values:
            raise KeyError(f"gate {name!r} is missing commodity {commodity}")
        return _coarsen_vector(values[commodity], group_size)

    stem_matrix = matrices["conv1"]
    stem_gate = gate("bn1")
    boundary: list[str] = []
    for output in range(stem_matrix.shape[0]):
        stem_in = f"stem:in:{output}"
        stem_out = f"stem:out:{output}"
        add_node(stem_in)
        add_node(stem_out)
        add_edge(
            f"stem:weight:{output}",
            "source",
            stem_in,
            float(stem_matrix[output].sum()),
        )
        add_edge(f"stem:gate:{output}", stem_in, stem_out, float(stem_gate[output]))
        boundary.append(stem_out)

    for layer_index, layer in enumerate(
        (model.layer1, model.layer2, model.layer3, model.layer4), start=1
    ):
        for block_index, block in enumerate(layer):
            prefix = f"layer{layer_index}.{block_index}"
            conv1 = matrices[f"{prefix}.conv1"]
            conv2 = matrices[f"{prefix}.conv2"]
            hidden_gate = gate(f"{prefix}.bn1")
            branch_gate = gate(f"{prefix}.branch_transport_gate")
            skip_gate = gate(f"{prefix}.skip_transport_gate")
            merge_gate = gate(f"{prefix}.merge_transport_gate")
            if conv1.shape[1] != len(boundary):
                raise ValueError(f"global input groups do not align for {prefix}")

            hidden_out_nodes: list[str] = []
            hidden_targets: list[set[str]] = []
            for hidden in range(conv1.shape[0]):
                hidden_in = f"{prefix}:hidden_in:{hidden}"
                hidden_out = f"{prefix}:hidden_out:{hidden}"
                add_node(hidden_in)
                add_node(hidden_out)
                for source, source_node in enumerate(boundary):
                    add_edge(
                        f"{prefix}:conv1:{hidden}:{source}",
                        source_node,
                        hidden_in,
                        float(conv1[hidden, source]),
                    )
                add_edge(
                    f"{prefix}:hidden_gate:{hidden}",
                    hidden_in,
                    hidden_out,
                    float(hidden_gate[hidden]),
                )
                hidden_out_nodes.append(hidden_out)
                hidden_targets.append({hidden_in, hidden_out})
            deletion_nodes[f"{prefix}.conv1"] = hidden_targets

            branch_nodes: list[str] = []
            output_targets: list[set[str]] = []
            merge_in_nodes: list[str] = []
            merge_out_nodes: list[str] = []
            for output in range(conv2.shape[0]):
                branch = f"{prefix}:branch:{output}"
                merge_in = f"{prefix}:merge_in:{output}"
                merge_out = f"{prefix}:merge_out:{output}"
                for name in (branch, merge_in, merge_out):
                    add_node(name)
                for hidden, hidden_node in enumerate(hidden_out_nodes):
                    add_edge(
                        f"{prefix}:conv2:{output}:{hidden}",
                        hidden_node,
                        branch,
                        float(conv2[output, hidden]),
                    )
                add_edge(
                    f"{prefix}:branch_gate:{output}",
                    branch,
                    merge_in,
                    float(branch_gate[output]),
                )
                add_edge(
                    f"{prefix}:merge_gate:{output}",
                    merge_in,
                    merge_out,
                    float(merge_gate[output]),
                )
                branch_nodes.append(branch)
                merge_in_nodes.append(merge_in)
                merge_out_nodes.append(merge_out)
                output_targets.append({branch})
            deletion_nodes[f"{prefix}.conv2"] = output_targets

            if block.downsample is None:
                if len(boundary) != len(merge_in_nodes):
                    raise ValueError(f"identity skip groups do not align for {prefix}")
                for output, (source_node, merge_in) in enumerate(
                    zip(boundary, merge_in_nodes, strict=True)
                ):
                    add_edge(
                        f"{prefix}:identity_skip:{output}",
                        source_node,
                        merge_in,
                        float(skip_gate[output]),
                    )
            else:
                projection = matrices[f"{prefix}.downsample.0"]
                if projection.shape != (len(merge_in_nodes), len(boundary)):
                    raise ValueError(f"projection skip groups do not align for {prefix}")
                for output, merge_in in enumerate(merge_in_nodes):
                    skip_node = f"{prefix}:skip:{output}"
                    add_node(skip_node)
                    for source, source_node in enumerate(boundary):
                        add_edge(
                            f"{prefix}:projection:{output}:{source}",
                            source_node,
                            skip_node,
                            float(projection[output, source]),
                        )
                    add_edge(
                        f"{prefix}:skip_gate:{output}",
                        skip_node,
                        merge_in,
                        float(skip_gate[output]),
                    )
            boundary = merge_out_nodes

    pool_gate = gate("pool")
    fc = matrices["fc"]
    fine_classes = [
        fine
        for fine, coarse in enumerate(gate_capacities.fine_to_coarse)
        if coarse == commodity
    ]
    if not fine_classes:
        raise ValueError(f"commodity {commodity} has no classifier outputs")
    if fc.shape[1] != len(boundary):
        raise ValueError("classifier input groups do not align to final boundary")
    for group, source_node in enumerate(boundary):
        pooled = f"pool:{group}"
        add_node(pooled)
        add_edge(f"pool:gate:{group}", source_node, pooled, float(pool_gate[group]))
        classifier_capacity = float(fc[fine_classes, group].sum())
        add_edge(f"classifier:{commodity}:{group}", pooled, "sink", classifier_capacity)
    return nodes, edges, deletion_nodes


def model_global_class_flow_scores(
    model: CifarResNet,
    weight_scores: Mapping[str, Tensor],
    gate_capacities: GlobalClassGateCapacities,
    *,
    target_names: Sequence[str],
    group_size: int = 16,
) -> dict[str, GlobalClassFlowScores]:
    """Compute global source-to-class deletion damage for selected groups."""

    if group_size < 1:
        raise ValueError("group_size must be positive")
    convolution_names = ["conv1"]
    for layer_index, layer in enumerate(
        (model.layer1, model.layer2, model.layer3, model.layer4), start=1
    ):
        for block_index, block in enumerate(layer):
            prefix = f"layer{layer_index}.{block_index}"
            convolution_names.extend((f"{prefix}.conv1", f"{prefix}.conv2"))
            if block.downsample is not None:
                convolution_names.append(f"{prefix}.downsample.0")
    missing = set(convolution_names) - set(weight_scores)
    if missing or "fc" not in weight_scores:
        all_missing = missing | ({"fc"} - set(weight_scores))
        raise KeyError(f"missing global weight capacities: {sorted(all_missing)}")
    matrices = {
        name: _coarsen_matrix(_conv_matrix(weight_scores[name]), group_size)
        for name in convolution_names
    }
    fc_score = weight_scores["fc"]
    if fc_score.ndim != 2 or fc_score.shape[0] != len(gate_capacities.fine_to_coarse):
        raise ValueError("classifier transport does not align to fine classes")
    matrices["fc"] = _coarsen_columns(
        fc_score.to(dtype=torch.float64, device="cpu"), group_size
    )

    requested = set(target_names)
    per_target: dict[str, list[Tensor]] = {name: [] for name in requested}
    commodities = sorted(gate_capacities.examples)
    if len(commodities) < 2:
        raise ValueError("global class flow requires at least two commodities")
    for commodity in commodities:
        nodes, edges, deletion_nodes = _build_commodity_graph(
            model, matrices, gate_capacities, commodity, group_size
        )
        unknown = requested - set(deletion_nodes)
        if unknown:
            raise KeyError(f"unknown global deletion groups: {sorted(unknown)}")
        baseline = max_flow_min_cut(nodes, edges, "source", "sink")
        normalizer = baseline.value if baseline.value > 0.0 else 1.0
        for name in requested:
            damages = torch.zeros(len(deletion_nodes[name]), dtype=torch.float64)
            for group_index, removed in enumerate(deletion_nodes[name]):
                result = max_flow_min_cut(
                    nodes, _without_nodes(edges, removed), "source", "sink"
                )
                damages[group_index] = max(0.0, baseline.value - result.value) / normalizer
            per_target[name].append(damages)

    output: dict[str, GlobalClassFlowScores] = {}
    for name, values in per_target.items():
        stacked = torch.stack(values)
        top_count = min(3, stacked.shape[0])
        output[name] = GlobalClassFlowScores(
            worst_class_marginal=stacked.max(dim=0).values,
            top3_mean_marginal=stacked.topk(top_count, dim=0).values.mean(dim=0),
            class_mean_marginal=stacked.mean(dim=0),
            commodity_count=stacked.shape[0],
            channel_group_size=group_size,
        )
    return output
