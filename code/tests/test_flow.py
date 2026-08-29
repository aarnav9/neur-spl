import unittest

from cut_to_grow.flow import Edge, max_flow_min_cut


class FlowTests(unittest.TestCase):
    def test_max_flow_and_cut(self) -> None:
        nodes = ["s", "a", "b", "t"]
        edges = [
            Edge("s_a", "s", "a", 5.0),
            Edge("a_t", "a", "t", 3.0),
            Edge("s_b", "s", "b", 2.0),
            Edge("b_t", "b", "t", 2.0),
            Edge("a_b", "a", "b", 1.0),
        ]
        result = max_flow_min_cut(nodes, edges, "s", "t")
        self.assertAlmostEqual(result.value, 5.0)
        self.assertEqual(set(result.cut_edge_ids), {"a_t", "b_t"})
        self.assertEqual(result.reachable, frozenset({"s", "a", "b"}))
        self.assertAlmostEqual(result.edge_flows["a_t"], 3.0)
        self.assertAlmostEqual(result.edge_flows["b_t"], 2.0)

    def test_duplicate_edge_ids_rejected(self) -> None:
        edges = [Edge("same", "s", "t", 1.0), Edge("same", "s", "t", 2.0)]
        with self.assertRaises(ValueError):
            max_flow_min_cut(["s", "t"], edges, "s", "t")


if __name__ == "__main__":
    unittest.main()

