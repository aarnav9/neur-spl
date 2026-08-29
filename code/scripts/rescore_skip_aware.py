#!/usr/bin/env python3
"""Add measured residual-skip flow scores to an existing deletion table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from cut_to_grow.block_flow import model_residual_flow_scores, model_skip_aware_flow_scores
from cut_to_grow.diagnostics import (
    aggregate_channel_scores,
    collect_skip_aware_capacities,
    write_probe_csv,
)
from cut_to_grow.pilot import (
    build_cifar100_loaders,
    load_checkpoint,
    model_from_checkpoint,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("pull/raw"))
    parser.add_argument("--data-source", choices=("torchvision", "huggingface"))
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
        source=args.data_source or config["data"]["source"],
    )

    branch_groups = list(model.channel_groups())[1:]
    group_by_name = {group.name: group for group in branch_groups}
    unknown = {str(record["group"]) for record in records} - set(group_by_name)
    if unknown:
        raise SystemExit(f"probe table contains unknown residual groups: {sorted(unknown)}")

    start = perf_counter()
    weight_scores, gate_capacities = collect_skip_aware_capacities(
        model,
        fit_loader,
        branch_groups,
        device,
        max_batches=args.calibration_batches,
        decay=float(config["capacity"]["ema_decay"]),
    )
    calibration_seconds = perf_counter() - start
    start = perf_counter()
    branch_flow = model_residual_flow_scores(
        model, weight_scores, group_size=group_size
    )
    skip_flow = model_skip_aware_flow_scores(
        model,
        weight_scores,
        gate_capacities,
        group_size=group_size,
    )
    scoring_seconds = perf_counter() - start

    for record in records:
        name = str(record["group"])
        prefix = name.rsplit(".", 1)[0]
        group_index = int(record["channel_group_index"])
        skip_marginal = aggregate_channel_scores(
            skip_flow[name].marginal_flow, group_size
        )
        skip_cut = aggregate_channel_scores(skip_flow[name].cut_incident, group_size)
        branch_marginal = aggregate_channel_scores(
            branch_flow[name].marginal_flow, group_size
        )
        if not 0 <= group_index < skip_marginal.numel():
            raise SystemExit(f"probe group index is out of range: {record}")
        record["skip_flow_marginal"] = float(skip_marginal[group_index])
        record["skip_cut_incident"] = float(skip_cut[group_index])
        record["skip_block_base_flow"] = skip_flow[name].base_flow
        record["recomputed_branch_flow_marginal"] = float(
            branch_marginal[group_index]
        )
        capacities = gate_capacities[prefix]
        for column, values in (
            ("branch_path_capacity", capacities.branch),
            ("skip_path_capacity", capacities.skip),
            ("merged_path_capacity", capacities.merged),
        ):
            grouped = aggregate_channel_scores(values, group_size)
            record[column] = float(grouped[group_index])
        record["skip_capacity_examples"] = capacities.examples
        record["skip_capacity_calibration_seconds"] = calibration_seconds
        record["skip_flow_scoring_seconds"] = scoring_seconds

    output = args.output or args.table.with_name(
        args.table.stem + "_skip_aware" + args.table.suffix
    )
    write_probe_csv(output, records)
    print(
        f"wrote {len(records)} skip-aware probes to {output}; device={device} "
        f"batches={args.calibration_batches} calibration={calibration_seconds:.3f}s "
        f"flow={scoring_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
