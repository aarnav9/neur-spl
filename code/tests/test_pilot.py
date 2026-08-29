import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from cut_to_grow.pilot import (
    _reconcile_metrics_for_resume,
    checkpoint_epochs,
    stratified_split_indices,
)


class PilotTests(unittest.TestCase):
    def test_stratified_split_is_exact_balanced_and_deterministic(self) -> None:
        labels = [label for label in range(4) for _ in range(10)]
        fit_a, selection_a = stratified_split_indices(labels, 8, seed=17)
        fit_b, selection_b = stratified_split_indices(labels, 8, seed=17)
        self.assertEqual((fit_a, selection_a), (fit_b, selection_b))
        self.assertEqual(len(fit_a), 32)
        self.assertEqual(len(selection_a), 8)
        self.assertEqual(
            Counter(labels[index] for index in selection_a),
            Counter({0: 2, 1: 2, 2: 2, 3: 2}),
        )
        self.assertFalse(set(fit_a) & set(selection_a))

    def test_checkpoint_epochs_collapse_for_short_smoke_runs(self) -> None:
        self.assertEqual(checkpoint_epochs(200, [0.1, 0.5, 1.0]), (20, 100, 200))
        self.assertEqual(checkpoint_epochs(1, [0.1, 0.5, 1.0]), (1,))

    def test_resume_metrics_follow_recovery_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"epoch": epoch}) + "\n" for epoch in (1, 2, 3)
                ),
                encoding="utf-8",
            )
            _reconcile_metrics_for_resume(
                path, {"epoch": 2, "metrics": {"epoch": 2}}
            )
            self.assertEqual(
                [json.loads(line)["epoch"] for line in path.read_text().splitlines()],
                [1, 2],
            )

    def test_resume_metrics_restore_checkpoint_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text(json.dumps({"epoch": 1}) + "\n", encoding="utf-8")
            _reconcile_metrics_for_resume(
                path, {"epoch": 2, "metrics": {"epoch": 2, "accuracy": 0.5}}
            )
            self.assertEqual(
                [json.loads(line)["epoch"] for line in path.read_text().splitlines()],
                [1, 2],
            )


if __name__ == "__main__":
    unittest.main()
