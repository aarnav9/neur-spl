#!/usr/bin/env python3
"""Run the frozen conv2 holdout test of flow beyond Group Fisher."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from cut_to_grow.diagnostics import (
    binary_auroc,
    binary_average_precision,
    paired_binary_metric_difference_ci,
    rank_percentile_scores,
    summarize_probe_correlations,
)


EXPECTED_GROUP_COUNTS = {
    "layer1.0.conv2": 4,
    "layer1.1.conv2": 4,
    "layer2.0.conv2": 8,
    "layer2.1.conv2": 8,
    "layer3.0.conv2": 16,
    "layer3.1.conv2": 16,
    "layer4.0.conv2": 32,
    "layer4.1.conv2": 32,
}


def _protect_k(scores: list[float], labels: list[bool]) -> int | None:
    positives = [index for index, label in enumerate(labels) if label]
    if not positives:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    ranks = {index: rank for rank, index in enumerate(order, start=1)}
    return max(ranks[index] for index in positives)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    with args.table.open(newline="", encoding="utf-8") as source:
        records = [dict(record) for record in csv.DictReader(source)]
    if not records:
        raise SystemExit("holdout table is empty")
    required = {
        "group",
        "channel_group_index",
        "deletion_group_size",
        "flow_marginal",
        "group_fisher",
        "loss_damage",
        "accuracy_damage",
    }
    missing = required - set(records[0])
    if missing:
        raise SystemExit(f"holdout table is missing columns: {sorted(missing)}")
    if {int(record["deletion_group_size"]) for record in records} != {16}:
        raise SystemExit("frozen test requires size-16 deletions")
    counts = Counter(str(record["group"]) for record in records)
    if counts != Counter(EXPECTED_GROUP_COUNTS):
        raise SystemExit(
            "frozen test requires the exhaustive 120-group conv2 table; "
            f"observed counts={dict(sorted(counts.items()))}"
        )
    for name, count in EXPECTED_GROUP_COUNTS.items():
        indices = {
            int(record["channel_group_index"])
            for record in records
            if record["group"] == name
        }
        if indices != set(range(count)):
            raise SystemExit(f"holdout group indices are incomplete for {name}")

    fisher = [float(record["group_fisher"]) for record in records]
    flow = [float(record["flow_marginal"]) for record in records]
    fisher_rank = rank_percentile_scores(fisher)
    flow_rank = rank_percentile_scores(flow)
    hybrid = [
        (fisher_value + flow_value) / 2.0
        for fisher_value, flow_value in zip(fisher_rank, flow_rank, strict=True)
    ]
    for record, value in zip(records, hybrid, strict=True):
        record["fisher_flow_hybrid"] = value

    labels = [float(record["accuracy_damage"]) >= 0.02 for record in records]
    positive_count = sum(labels)
    feasible = 5 <= positive_count <= len(labels) - 5
    result: dict[str, object] = {
        "protocol": "frozen_conv2_conditional_flow_v1",
        "table": str(args.table),
        "probes": len(records),
        "catastrophic_accuracy_drop_threshold": 0.02,
        "catastrophic_positives": positive_count,
        "tail_endpoint_feasible": feasible,
        "hybrid_definition": (
            "0.5 * global_percentile(group_fisher) + "
            "0.5 * global_percentile(flow_marginal)"
        ),
    }
    if feasible:
        summary = summarize_probe_correlations(
            records,
            score_names=("group_fisher", "flow_marginal", "fisher_flow_hybrid"),
        )
        comparisons: dict[str, dict[str, float]] = {}
        for metric_index, (name, metric) in enumerate(
            (("auroc", binary_auroc), ("average_precision", binary_average_precision))
        ):
            difference, low, high = paired_binary_metric_difference_ci(
                hybrid,
                fisher,
                labels,
                metric,
                seed=20260830 + metric_index,
            )
            comparisons[name] = {
                "hybrid_minus_group_fisher": difference,
                "ci95_low": low,
                "ci95_high": high,
            }
        auroc = comparisons["auroc"]
        result.update(
            {
                "scores": summary,
                "paired_hybrid_minus_group_fisher": comparisons,
                "protect_k": {
                    "group_fisher": _protect_k(fisher, labels),
                    "flow_marginal": _protect_k(flow, labels),
                    "fisher_flow_hybrid": _protect_k(hybrid, labels),
                },
                "primary_pass": (
                    float(auroc["hybrid_minus_group_fisher"]) >= 0.05
                    and float(auroc["ci95_low"]) > 0.0
                ),
            }
        )
    else:
        result["primary_pass"] = None
        result["reason"] = "fewer than five positives or negatives; tail test is inconclusive"

    if any(
        not math.isfinite(float(record[name]))
        for record in records
        for name in ("group_fisher", "flow_marginal", "loss_damage")
    ):
        raise FloatingPointError("holdout table contains a non-finite analysis value")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
