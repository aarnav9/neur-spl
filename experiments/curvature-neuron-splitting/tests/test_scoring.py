import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurspl.evaluate import (  # noqa: E402
    extract_last_boxed,
    extract_prediction,
    is_correct,
    normalize_math,
)


class ScoringTests(unittest.TestCase):
    def test_nested_boxed_answer(self):
        self.assertEqual(extract_last_boxed(r"work \boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_final_answer_wins(self):
        self.assertEqual(extract_prediction("First 12.\nFinal answer: 18"), "18")

    def test_gsm8k_numeric_normalization(self):
        self.assertTrue(is_correct("gsm8k", "$1,024.", "1024"))

    def test_math_normalization(self):
        self.assertEqual(normalize_math(r"$\left( x \right)$"), r"(x)")


if __name__ == "__main__":
    unittest.main()
