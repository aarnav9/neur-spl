"""A small exact directed max-flow/min-cut implementation.

The solver is designed for correctness tests and coarsened pilot graphs. It is
not intended to be the eventual large-model systems implementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source: str
    target: str
    capacity: float

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise ValueError("edge_id must be nonempty")
        if self.source == self.target:
            raise ValueError("self loops are not supported")
        if not isfinite(self.capacity) or self.capacity < 0:
            raise ValueError("capacity must be finite and nonnegative")


@dataclass(frozen=True)
class FlowResult:
    value: float
    reachable: frozenset[str]
    cut_edge_ids: tuple[str, ...]
    edge_flows: dict[str, float]


@dataclass
class _Arc:
    target: int
    reverse: int
    residual: float
    original_capacity: float
    edge_id: str | None
    is_forward: bool


def max_flow_min_cut(
    nodes: Iterable[str],
    edges: Iterable[Edge],
    source: str,
    sink: str,
    *,
    epsilon: float = 1.0e-12,
) -> FlowResult:
    """Compute exact max flow and one corresponding minimum cut with Dinic's method."""

    node_list = list(dict.fromkeys(nodes))
    if source == sink:
        raise ValueError("source and sink must differ")
    if source not in node_list or sink not in node_list:
        raise ValueError("source and sink must be present in nodes")

    edge_list = list(edges)
    edge_ids = [edge.edge_id for edge in edge_list]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("edge_id values must be unique")

    index = {name: position for position, name in enumerate(node_list)}
    graph: list[list[_Arc]] = [[] for _ in node_list]
    forward_locations: dict[str, tuple[int, int]] = {}

    def add_edge(edge: Edge) -> None:
        if edge.source not in index or edge.target not in index:
            raise ValueError(f"edge {edge.edge_id!r} refers to an unknown node")
        u = index[edge.source]
        v = index[edge.target]
        forward_index = len(graph[u])
        reverse_index = len(graph[v])
        graph[u].append(
            _Arc(v, reverse_index, edge.capacity, edge.capacity, edge.edge_id, True)
        )
        graph[v].append(_Arc(u, forward_index, 0.0, 0.0, None, False))
        forward_locations[edge.edge_id] = (u, forward_index)

    for edge in edge_list:
        add_edge(edge)

    source_index = index[source]
    sink_index = index[sink]
    total_flow = 0.0

    while True:
        level = [-1] * len(node_list)
        level[source_index] = 0
        queue: deque[int] = deque([source_index])
        while queue:
            u = queue.popleft()
            for arc in graph[u]:
                if arc.residual > epsilon and level[arc.target] < 0:
                    level[arc.target] = level[u] + 1
                    queue.append(arc.target)
        if level[sink_index] < 0:
            break

        cursor = [0] * len(node_list)

        def send(u: int, amount: float) -> float:
            if u == sink_index:
                return amount
            while cursor[u] < len(graph[u]):
                arc_index = cursor[u]
                arc = graph[u][arc_index]
                if arc.residual > epsilon and level[arc.target] == level[u] + 1:
                    pushed = send(arc.target, min(amount, arc.residual))
                    if pushed > epsilon:
                        arc.residual -= pushed
                        graph[arc.target][arc.reverse].residual += pushed
                        return pushed
                cursor[u] += 1
            return 0.0

        while True:
            pushed = send(source_index, float("inf"))
            if pushed <= epsilon:
                break
            total_flow += pushed

    reachable_indices = {source_index}
    queue = deque([source_index])
    while queue:
        u = queue.popleft()
        for arc in graph[u]:
            if arc.residual > epsilon and arc.target not in reachable_indices:
                reachable_indices.add(arc.target)
                queue.append(arc.target)

    reachable = frozenset(node_list[i] for i in reachable_indices)
    cut_edges = tuple(
        sorted(
            edge.edge_id
            for edge in edge_list
            if edge.capacity > epsilon
            and edge.source in reachable
            and edge.target not in reachable
        )
    )
    edge_flows: dict[str, float] = {}
    for edge_id, (u, arc_index) in forward_locations.items():
        arc = graph[u][arc_index]
        flow = arc.original_capacity - arc.residual
        edge_flows[edge_id] = 0.0 if abs(flow) <= epsilon else flow

    return FlowResult(total_flow, reachable, cut_edges, edge_flows)

