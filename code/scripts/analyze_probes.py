#!/usr/bin/env python3
"""Summarize score-versus-deletion-damage correlations from a probe CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cut_to_grow.diagnostics import (
    binary_auroc,
    binary_average_precision,
    paired_binary_metric_difference_ci,
    summarize_probe_correlations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument(
        "--compare",
        nargs=2,
        action="append",
        metavar=("LEFT_SCORE", "RIGHT_SCORE"),
        help="report paired bootstrap differences for two score columns",
    )
    args = parser.parse_args()
    with args.table.open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))
    scores = summarize_probe_correlations(records)
    output: object = scores
    if args.compare:
        labels = [float(record["accuracy_damage"]) >= 0.02 for record in records]
        comparisons: dict[str, dict[str, float]] = {}
        for pair_index, (left_name, right_name) in enumerate(args.compare):
            missing = {left_name, right_name} - set(records[0])
            if missing:
                raise SystemExit(f"probe table is missing score columns: {sorted(missing)}")
            left = [float(record[left_name]) for record in records]
            right = [float(record[right_name]) for record in records]
            key = f"{left_name}_minus_{right_name}"
            comparisons[key] = {}
            for metric_index, (metric_name, metric) in enumerate(
                (("auroc", binary_auroc), ("average_precision", binary_average_precision))
            ):
                difference, low, high = paired_binary_metric_difference_ci(
                    left,
                    right,
                    labels,
                    metric,
                    seed=20260828 + pair_index * 2 + metric_index,
                )
                comparisons[key][f"{metric_name}_difference"] = difference
                comparisons[key][f"{metric_name}_difference_ci95_low"] = low
                comparisons[key][f"{metric_name}_difference_ci95_high"] = high
        output = {"scores": scores, "paired_comparisons": comparisons}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
