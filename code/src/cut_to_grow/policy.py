"""Reference cut-to-grow policy for small graphs.

This implementation performs exact counterfactual flow recomputation. It is a
research oracle for toy and diagnostic graphs, not a scalable training policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .flow import Edge, max_flow_min_cut


@dataclass(frozen=True)
class CandidateEdge:
    edge_id: str
    source: str
    target: str
    capacity: float
    active: bool = True
    gradient_score: float = 0.0
    loss_saliency: float = 0.0

    def as_edge(self) -> Edge:
        return Edge(self.edge_id, self.source, self.target, self.capacity)


@dataclass(frozen=True)
class SwapDecision:
    remove_edge_id: str
    grow_edge_id: str
    protected_edge_ids: tuple[str, ...]
    flow_before: float
    flow_after_removal: float
    flow_after_growth: float
    removal_flow_damage: float
    growth_flow_gain: float


def _active_edges(candidates: Iterable[CandidateEdge]) -> list[Edge]:
    return [candidate.as_edge() for candidate in candidates if candidate.active]


def propose_swap(
    nodes: Iterable[str],
    candidates: Iterable[CandidateEdge],
    source: str,
    sink: str,
    *,
    loss_weight: float = 0.0,
    gradient_weight: float = 0.0,
) -> SwapDecision:
    """Remove one redundant edge and grow one edge across a weak cut.

    Current minimum-cut edges are protected. Removal uses exact marginal flow
    damage plus optional task saliency. Growth prefers currently inactive edges
    crossing the post-removal minimum cut and ranks them by exact flow gain plus
    an optional missing-gradient score.
    """

    node_list = list(nodes)
    candidate_list = list(candidates)
    active = _active_edges(candidate_list)
    baseline = max_flow_min_cut(node_list, active, source, sink)
    protected = set(baseline.cut_edge_ids)

    removal_options: list[tuple[float, float, str, float]] = []
    for candidate in candidate_list:
        if not candidate.active or candidate.edge_id in protected:
            continue
        without = [edge for edge in active if edge.edge_id != candidate.edge_id]
        result = max_flow_min_cut(node_list, without, source, sink)
        damage = baseline.value - result.value
        score = damage + loss_weight * candidate.loss_saliency
        removal_options.append((score, candidate.capacity, candidate.edge_id, result.value))

    if not removal_options:
        raise ValueError("no removable active edge remains outside the protected cut")

    _, _, remove_edge_id, flow_after_removal = min(removal_options)
    remaining = [edge for edge in active if edge.edge_id != remove_edge_id]
    after_removal = max_flow_min_cut(node_list, remaining, source, sink)

    inactive = [
        candidate
        for candidate in candidate_list
        if not candidate.active and candidate.edge_id != remove_edge_id
    ]
    crossing = [
        candidate
        for candidate in inactive
        if candidate.source in after_removal.reachable
        and candidate.target not in after_removal.reachable
    ]
    growth_pool = crossing or inactive
    if not growth_pool:
        raise ValueError("no inactive edge is available for growth")

    growth_options: list[tuple[float, float, str, float]] = []
    for candidate in growth_pool:
        result = max_flow_min_cut(
            node_list,
            [*remaining, candidate.as_edge()],
            source,
            sink,
        )
        gain = result.value - after_removal.value
        score = gain + gradient_weight * candidate.gradient_score
        growth_options.append((-score, -candidate.gradient_score, candidate.edge_id, result.value))

    _, _, grow_edge_id, flow_after_growth = min(growth_options)
    return SwapDecision(
        remove_edge_id=remove_edge_id,
        grow_edge_id=grow_edge_id,
        protected_edge_ids=tuple(sorted(protected)),
        flow_before=baseline.value,
        flow_after_removal=flow_after_removal,
        flow_after_growth=flow_after_growth,
        removal_flow_damage=baseline.value - flow_after_removal,
        growth_flow_gain=flow_after_growth - flow_after_removal,
    )


def apply_swap(
    candidates: Iterable[CandidateEdge], decision: SwapDecision
) -> tuple[CandidateEdge, ...]:
    """Apply a swap while preserving the number of active edges exactly."""

    output: list[CandidateEdge] = []
    saw_remove = False
    saw_grow = False
    active_before = 0
    for candidate in candidates:
        active_before += int(candidate.active)
        if candidate.edge_id == decision.remove_edge_id:
            output.append(replace(candidate, active=False))
            saw_remove = True
        elif candidate.edge_id == decision.grow_edge_id:
            output.append(replace(candidate, active=True))
            saw_grow = True
        else:
            output.append(candidate)
    if not saw_remove or not saw_grow:
        raise ValueError("decision refers to an unknown candidate edge")
    if sum(int(candidate.active) for candidate in output) != active_before:
        raise AssertionError("cut-to-grow swap changed the active-edge budget")
    return tuple(output)

