#!/usr/bin/env python3
"""Add true shared-capacity class-commodity dual scores to a probe table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from cut_to_grow.diagnostics import collect_global_class_capacities, write_probe_csv
from cut_to_grow.multicommodity import model_concurrent_flow_scores
from cut_to_grow.pilot import (
    build_cifar100_loaders,
    load_checkpoint,
    load_cifar100_coarse_mapping,
    model_from_checkpoint,
    resolve_device,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("pull/raw"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--calibration-batches", type=int, default=8)
    args = parser.parse_args()
    with args.table.open(newline="", encoding="utf-8") as source:
        records: list[dict[str, object]] = list(csv.DictReader(source))
    if not records or args.calibration_batches < 1:
        raise SystemExit("table and positive calibration-batches are required")
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
        source="huggingface",
    )
    mapping = load_cifar100_coarse_mapping(args.data_root)
    target_names = sorted({str(record["group"]) for record in records})
    start = perf_counter()
    weights, gates = collect_global_class_capacities(
        model,
        fit_loader,
        device,
        class_to_commodity=mapping,
        max_batches=args.calibration_batches,
        decay=float(config["capacity"]["ema_decay"]),
    )
    calibration_seconds = perf_counter() - start
    scores = model_concurrent_flow_scores(
        model,
        weights,
        gates,
        target_names=target_names,
        group_size=group_size,
    )
    first = next(iter(scores.values()))
    for record in records:
        group_scores = scores[str(record["group"])]
        group_index = int(record["channel_group_index"])
        record["concurrent_flow_dual"] = float(
            group_scores.dual_deletion_value[group_index]
        )
        record["concurrent_flow_value"] = group_scores.concurrent_flow
        record["concurrent_flow_commodities"] = group_scores.commodities
        record["concurrent_flow_shared_edges"] = group_scores.shared_edges
        record["concurrent_flow_solver_seconds"] = group_scores.solver_seconds
        record["concurrent_capacity_seconds"] = calibration_seconds
    output = args.output or args.table.with_name(
        args.table.stem + "_concurrent_flow" + args.table.suffix
    )
    write_probe_csv(output, records)
    print(
        f"wrote {len(records)} concurrent-flow probes to {output}; device={device} "
        f"commodities={first.commodities} edges={first.shared_edges} "
        f"calibration={calibration_seconds:.3f}s solver={first.solver_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
