#!/usr/bin/env python3
"""Run a deterministic cut-to-grow swap on a small transport graph."""

from __future__ import annotations

import json

from cut_to_grow.policy import CandidateEdge, apply_swap, propose_swap


def main() -> None:
    nodes = ["source", "a", "b", "sink"]
    candidates = [
        CandidateEdge("source_a", "source", "a", 5.0, active=True),
        CandidateEdge("a_sink", "a", "sink", 3.0, active=True),
        CandidateEdge("source_b", "source", "b", 2.0, active=True),
        CandidateEdge("b_sink", "b", "sink", 2.0, active=True),
        CandidateEdge("a_b_redundant", "a", "b", 1.0, active=True),
        CandidateEdge(
            "a_sink_growth",
            "a",
            "sink",
            2.0,
            active=False,
            gradient_score=1.0,
        ),
    ]
    decision = propose_swap(nodes, candidates, "source", "sink", gradient_weight=0.01)
    updated = apply_swap(candidates, decision)
    payload = {
        "decision": decision.__dict__,
        "active_before": [edge.edge_id for edge in candidates if edge.active],
        "active_after": [edge.edge_id for edge in updated if edge.active],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

