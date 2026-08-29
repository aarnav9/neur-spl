"""Shared-capacity maximum concurrent flow and dual deletion scores."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from torch import Tensor

from .flow import Edge
from .global_flow import (
    GlobalClassGateCapacities,
    _build_commodity_graph,
    _coarsen_columns,
    _coarsen_matrix,
    _conv_matrix,
)
from .model import CifarResNet


@dataclass(frozen=True)
class ConcurrentFlowScores:
    dual_deletion_value: Tensor
    concurrent_flow: float
    commodities: int
    shared_edges: int
    solver_seconds: float
    channel_group_size: int


def _weight_matrices(
    model: CifarResNet, weight_scores: Mapping[str, Tensor], group_size: int
) -> dict[str, Tensor]:
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
        raise KeyError(f"missing concurrent-flow weights: {sorted(all_missing)}")
    matrices = {
        name: _coarsen_matrix(_conv_matrix(weight_scores[name]), group_size)
        for name in convolution_names
    }
    matrices["fc"] = _coarsen_columns(
        weight_scores["fc"].to(dtype=torch.float64, device="cpu"), group_size
    )
    return matrices


def _shared_graph(
    model: CifarResNet,
    matrices: Mapping[str, Tensor],
    gates: GlobalClassGateCapacities,
    group_size: int,
) -> tuple[list[str], list[Edge], dict[str, list[set[str]]], list[int]]:
    commodities = sorted(gates.examples)
    total_examples = sum(gates.examples.values())
    topology: dict[str, tuple[str, str]] = {}
    capacity_sums: dict[str, float] = {}
    nodes: set[str] = {"source"}
    deletion_nodes: dict[str, list[set[str]]] | None = None
    for commodity in commodities:
        commodity_nodes, commodity_edges, targets = _build_commodity_graph(
            model, matrices, gates, commodity, group_size
        )
        nodes.update(node for node in commodity_nodes if node != "sink")
        sink = f"sink:{commodity}"
        nodes.add(sink)
        if deletion_nodes is None:
            deletion_nodes = targets
        elif targets != deletion_nodes:
            raise RuntimeError("commodity deletion nodes do not share a topology")
        weight = gates.examples[commodity] / total_examples
        for edge in commodity_edges:
            if edge.edge_id.startswith("classifier:"):
                edge_id = edge.edge_id
                source, target = edge.source, sink
                capacity = edge.capacity
            else:
                edge_id = edge.edge_id
                source, target = edge.source, edge.target
                capacity = edge.capacity * weight
            previous = topology.get(edge_id)
            if previous is not None and previous != (source, target):
                raise RuntimeError(f"edge topology differs across commodities: {edge_id}")
            topology[edge_id] = (source, target)
            capacity_sums[edge_id] = capacity_sums.get(edge_id, 0.0) + capacity
    if deletion_nodes is None:  # pragma: no cover - commodities validated upstream
        raise RuntimeError("no commodity graph was built")
    edges = [
        Edge(edge_id, *topology[edge_id], capacity)
        for edge_id, capacity in capacity_sums.items()
        if capacity > 0.0
    ]
    return sorted(nodes), edges, deletion_nodes, commodities


def model_concurrent_flow_scores(
    model: CifarResNet,
    weight_scores: Mapping[str, Tensor],
    gate_capacities: GlobalClassGateCapacities,
    *,
    target_names: Sequence[str],
    group_size: int = 16,
) -> dict[str, ConcurrentFlowScores]:
    """Solve max-min class flow once and score deletions using capacity duals."""

    matrices = _weight_matrices(model, weight_scores, group_size)
    nodes, edges, deletion_nodes, commodities = _shared_graph(
        model, matrices, gate_capacities, group_size
    )
    requested = set(target_names)
    unknown = requested - set(deletion_nodes)
    if unknown:
        raise KeyError(f"unknown concurrent-flow deletion groups: {sorted(unknown)}")
    node_index = {node: index for index, node in enumerate(nodes)}
    edge_index = {edge.edge_id: index for index, edge in enumerate(edges)}
    commodity_count = len(commodities)
    edge_count = len(edges)
    node_count = len(nodes)
    flow_variables = commodity_count * edge_count
    concurrent_index = flow_variables

    ub_rows: list[int] = []
    ub_columns: list[int] = []
    ub_values: list[float] = []
    for edge_position in range(edge_count):
        for commodity_position in range(commodity_count):
            ub_rows.append(edge_position)
            ub_columns.append(commodity_position * edge_count + edge_position)
            ub_values.append(1.0)
    a_ub = coo_matrix(
        (ub_values, (ub_rows, ub_columns)),
        shape=(edge_count, flow_variables + 1),
    ).tocsr()
    b_ub = np.asarray([edge.capacity for edge in edges], dtype=np.float64)

    eq_rows: list[int] = []
    eq_columns: list[int] = []
    eq_values: list[float] = []
    for commodity_position, commodity in enumerate(commodities):
        row_offset = commodity_position * node_count
        variable_offset = commodity_position * edge_count
        for edge_position, edge in enumerate(edges):
            variable = variable_offset + edge_position
            eq_rows.extend(
                (row_offset + node_index[edge.source], row_offset + node_index[edge.target])
            )
            eq_columns.extend((variable, variable))
            eq_values.extend((1.0, -1.0))
        eq_rows.extend(
            (row_offset + node_index["source"], row_offset + node_index[f"sink:{commodity}"])
        )
        eq_columns.extend((concurrent_index, concurrent_index))
        eq_values.extend((-1.0, 1.0))
    a_eq = coo_matrix(
        (eq_values, (eq_rows, eq_columns)),
        shape=(commodity_count * node_count, flow_variables + 1),
    ).tocsr()
    objective = np.zeros(flow_variables + 1, dtype=np.float64)
    objective[concurrent_index] = -1.0
    start = perf_counter()
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.zeros(commodity_count * node_count, dtype=np.float64),
        bounds=(0.0, None),
        method="highs",
    )
    solver_seconds = perf_counter() - start
    if not result.success:
        raise RuntimeError(f"concurrent-flow LP failed: {result.message}")
    concurrent_flow = float(result.x[concurrent_index])
    dual = np.maximum(0.0, -np.asarray(result.ineqlin.marginals))
    edge_value = {
        edge.edge_id: edge.capacity * dual[position]
        for position, edge in enumerate(edges)
    }
    normalizer = concurrent_flow if concurrent_flow > 0.0 else 1.0
    output: dict[str, ConcurrentFlowScores] = {}
    for name in requested:
        values = torch.zeros(len(deletion_nodes[name]), dtype=torch.float64)
        for group_index, removed in enumerate(deletion_nodes[name]):
            incident = {
                edge.edge_id
                for edge in edges
                if edge.source in removed or edge.target in removed
            }
            values[group_index] = sum(edge_value[edge_id] for edge_id in incident) / normalizer
        output[name] = ConcurrentFlowScores(
            dual_deletion_value=values,
            concurrent_flow=concurrent_flow,
            commodities=commodity_count,
            shared_edges=edge_count,
            solver_seconds=solver_seconds,
            channel_group_size=group_size,
        )
    return output
