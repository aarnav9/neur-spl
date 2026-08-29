import unittest

from cut_to_grow.policy import CandidateEdge, apply_swap, propose_swap


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = ["source", "a", "b", "sink"]
        self.candidates = [
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

    def test_cut_to_grow_improves_flow_and_preserves_budget(self) -> None:
        decision = propose_swap(
            self.nodes,
            self.candidates,
            "source",
            "sink",
            gradient_weight=0.01,
        )
        self.assertEqual(decision.remove_edge_id, "a_b_redundant")
        self.assertEqual(decision.grow_edge_id, "a_sink_growth")
        self.assertEqual(set(decision.protected_edge_ids), {"a_sink", "b_sink"})
        self.assertAlmostEqual(decision.flow_before, 5.0)
        self.assertAlmostEqual(decision.flow_after_growth, 7.0)

        updated = apply_swap(self.candidates, decision)
        self.assertEqual(
            sum(candidate.active for candidate in updated),
            sum(candidate.active for candidate in self.candidates),
        )


if __name__ == "__main__":
    unittest.main()

