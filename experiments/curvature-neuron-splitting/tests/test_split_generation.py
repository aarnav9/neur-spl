import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurspl.evaluate_split_generation import (  # noqa: E402
    exact_mcnemar_two_sided,
    paired_summary,
)
from neurspl.split_generation_sweep import parse_epsilons  # noqa: E402


class SplitGenerationTests(unittest.TestCase):
    def test_epsilon_parser_sorts_and_deduplicates(self):
        self.assertEqual(parse_epsilons("0.2,0,0.1,0.2"), [0.0, 0.1, 0.2])

    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_two_sided(0, 5), 0.0625)
        self.assertEqual(exact_mcnemar_two_sided(2, 2), 1.0)

    def test_paired_summary(self):
        original = [
            {"example_id": "a", "correct": True, "completion": "x", "prediction": "1"},
            {"example_id": "b", "correct": False, "completion": "y", "prediction": "2"},
            {"example_id": "c", "correct": False, "completion": "z", "prediction": "3"},
        ]
        split = [
            {"example_id": "a", "correct": False, "completion": "u", "prediction": "4"},
            {"example_id": "b", "correct": True, "completion": "v", "prediction": "5"},
            {"example_id": "c", "correct": False, "completion": "z", "prediction": "3"},
        ]
        summary = paired_summary(original, split)
        self.assertEqual(summary["original_only_correct"], 1)
        self.assertEqual(summary["split_only_correct"], 1)
        self.assertEqual(summary["both_wrong"], 1)
        self.assertEqual(summary["changed_prediction"], 2)


if __name__ == "__main__":
    unittest.main()
