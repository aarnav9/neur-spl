#!/usr/bin/env python3
"""Add coarse-class-conditioned flow scores to an existing deletion table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from cut_to_grow.block_flow import model_class_conditioned_flow_scores
from cut_to_grow.diagnostics import collect_class_conditioned_capacities, write_probe_csv
from cut_to_grow.pilot import (
    build_cifar100_loaders,
    load_checkpoint,
    load_cifar100_coarse_mapping,
    model_from_checkpoint,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("pull/raw"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--calibration-batches", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration_batches < 1:
        raise SystemExit("--calibration-batches must be positive")
    with args.table.open(newline="", encoding="utf-8") as source:
        records: list[dict[str, object]] = list(csv.DictReader(source))
    if not records:
        raise SystemExit(f"probe table is empty: {args.table}")
    required = {"group", "channel_group_index", "deletion_group_size"}
    missing = required - set(records[0])
    if missing:
        raise SystemExit(f"probe table is missing columns: {sorted(missing)}")
    group_sizes = {int(record["deletion_group_size"]) for record in records}
    if len(group_sizes) != 1:
        raise SystemExit(f"probe table has mixed group sizes: {sorted(group_sizes)}")
    group_size = group_sizes.pop()

    payload = load_checkpoint(args.checkpoint)
    config = payload["config"]
    if config["data"]["source"] != "huggingface":
        raise SystemExit("frozen coarse commodities require the pinned Hugging Face data")
    seed = int(payload["seed"])
    seed_everything(seed)
    device = resolve_device(args.device)
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    model = model_from_checkpoint(payload).to(device)
    fit_loader, _ = build_cifar100_loaders(
        args.data_root,
        split_seed=int(config["data"]["split_seed"]),
        selection_examples=int(config["data"]["selection_examples"]),
        batch_size=batch_size,
        num_workers=args.num_workers,
        download=False,
        source="huggingface",
    )
    class_to_commodity = load_cifar100_coarse_mapping(args.data_root)
    branch_groups = list(model.channel_groups())[1:]
    group_names = {group.name for group in branch_groups}
    unknown = {str(record["group"]) for record in records} - group_names
    if unknown:
        raise SystemExit(f"probe table contains unknown residual groups: {sorted(unknown)}")

    start = perf_counter()
    weight_scores, conditional_gates = collect_class_conditioned_capacities(
        model,
        fit_loader,
        branch_groups,
        device,
        class_to_commodity=class_to_commodity,
        max_batches=args.calibration_batches,
        decay=float(config["capacity"]["ema_decay"]),
    )
    calibration_seconds = perf_counter() - start
    observed_commodities = set(next(iter(conditional_gates.values())).examples)
    if args.calibration_batches == 8 and observed_commodities != set(range(20)):
        raise RuntimeError(
            f"full calibration did not observe every coarse class: {observed_commodities}"
        )
    start = perf_counter()
    scores = model_class_conditioned_flow_scores(
        model,
        weight_scores,
        conditional_gates,
        group_size=group_size,
    )
    scoring_seconds = perf_counter() - start

    for record in records:
        name = str(record["group"])
        group_index = int(record["channel_group_index"])
        group_scores = scores[name]
        if not 0 <= group_index < group_scores.worst_class_marginal.numel():
            raise SystemExit(f"probe group index is out of range: {record}")
        record["class_flow_worst"] = float(
            group_scores.worst_class_marginal[group_index]
        )
        record["class_flow_top3_mean"] = float(
            group_scores.top3_mean_marginal[group_index]
        )
        record["class_flow_mean"] = float(
            group_scores.class_mean_marginal[group_index]
        )
        record["class_flow_commodities"] = group_scores.commodity_count
        record["class_flow_calibration_seconds"] = calibration_seconds
        record["class_flow_scoring_seconds"] = scoring_seconds

    output = args.output or args.table.with_name(
        args.table.stem + "_class_flow" + args.table.suffix
    )
    write_probe_csv(output, records)
    print(
        f"wrote {len(records)} class-flow probes to {output}; device={device} "
        f"batches={args.calibration_batches} commodities={len(observed_commodities)} "
        f"calibration={calibration_seconds:.3f}s flow={scoring_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
