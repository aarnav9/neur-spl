import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurspl.split_controls import (  # noqa: E402
    choose_random_indices,
    random_unit_directions,
    summarize_family,
)


class SplitControlTests(unittest.TestCase):
    def test_random_directions_are_deterministic_and_unit_norm(self):
        first = random_unit_directions(4, 11, 123)
        second = random_unit_directions(4, 11, 123)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first.norm(dim=1), torch.ones(4), atol=1e-6))

    def test_random_indices_are_unique_sorted_and_excluded(self):
        values = choose_random_indices(20, 8, {1, 3, 5}, 456)
        self.assertEqual(values, tuple(sorted(values)))
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(set(values).isdisjoint({1, 3, 5}))

    def test_summary_counts_controls_beating_curvature(self):
        arms = [
            {"mean_nll": 1.2, "delta_from_original": -0.1},
            {"mean_nll": 1.4, "delta_from_original": 0.1},
        ]
        summary = summarize_family(arms, curvature_mean_nll=1.25)
        self.assertEqual(summary["beat_or_tied_curvature"], 1)
        self.assertEqual(summary["improved_over_original"], 1)
        self.assertAlmostEqual(summary["plus_one_tail_fraction"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
