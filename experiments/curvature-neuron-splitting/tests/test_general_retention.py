import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurspl.evaluate_general_retention import summarize  # noqa: E402
from neurspl.prepare_general_retention import (  # noqa: E402
    choose_validation_file,
    deterministic_sample,
    materialize_arc,
    materialize_boolq,
    materialize_hellaswag,
)


class GeneralRetentionTests(unittest.TestCase):
    def test_choose_preferred_validation_file(self):
        files = ["data/train.parquet", "data/validation.parquet"]
        self.assertEqual(
            choose_validation_file(files, "data/validation.parquet"),
            "data/validation.parquet",
        )

    def test_materializers(self):
        hellaswag = materialize_hellaswag(
            [{"ind": 4, "ctx": "context", "endings": ["a", "b"], "label": "1"}]
        )
        self.assertEqual(hellaswag[0]["answer_index"], 1)
        arc = materialize_arc(
            [
                {
                    "id": "x",
                    "question": "q",
                    "choices": {"label": ["A", "B"], "text": ["a", "b"]},
                    "answerKey": "B",
                }
            ]
        )
        self.assertEqual(arc[0]["answer_index"], 1)
        boolq = materialize_boolq(
            [{"passage": "p", "question": "q", "answer": True}]
        )
        self.assertEqual(boolq[0]["choices"], ["No", "Yes"])
        self.assertEqual(boolq[0]["answer_index"], 1)

    def test_sampling_is_deterministic(self):
        rows = [{"value": index} for index in range(20)]
        self.assertEqual(
            deterministic_sample(rows, 5, 42), deterministic_sample(rows, 5, 42)
        )

    def test_paired_summary_by_task(self):
        original = [
            {
                "example_id": "a",
                "benchmark": "one",
                "correct": True,
                "prediction_index": 0,
            },
            {
                "example_id": "b",
                "benchmark": "two",
                "correct": False,
                "prediction_index": 0,
            },
        ]
        split = [
            {
                "example_id": "a",
                "benchmark": "one",
                "correct": False,
                "prediction_index": 1,
            },
            {
                "example_id": "b",
                "benchmark": "two",
                "correct": True,
                "prediction_index": 1,
            },
        ]
        result = summarize(original, split)
        self.assertEqual(result["one"]["original_only_correct"], 1)
        self.assertEqual(result["two"]["split_only_correct"], 1)
        self.assertEqual(result["overall"]["changed_prediction"], 2)


if __name__ == "__main__":
    unittest.main()
