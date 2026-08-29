#!/usr/bin/env python3
"""Add strong gradient baselines to an existing cached deletion table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from cut_to_grow.diagnostics import collect_gate_sensitivity_scores, write_probe_csv
from cut_to_grow.pilot import (
    build_cifar100_loaders,
    build_synthetic_loaders,
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
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-examples", type=int, default=64)
    return parser.parse_args()


def _read_records(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as source:
        records: list[dict[str, object]] = list(csv.DictReader(source))
    if not records:
        raise SystemExit(f"probe table is empty: {path}")
    required = {
        "group",
        "gate",
        "channel_group_index",
        "deletion_group_size",
        "loss_damage",
    }
    missing = required - set(records[0])
    if missing:
        raise SystemExit(f"probe table is missing columns: {sorted(missing)}")
    return records


def main() -> None:
    args = parse_args()
    if args.calibration_batches < 1:
        raise SystemExit("--calibration-batches must be positive")
    records = _read_records(args.table)
    group_sizes = {int(record["deletion_group_size"]) for record in records}
    if len(group_sizes) != 1:
        raise SystemExit(f"probe table has mixed deletion group sizes: {sorted(group_sizes)}")
    group_size = group_sizes.pop()

    payload = load_checkpoint(args.checkpoint)
    config = payload["config"]
    seed = int(payload["seed"])
    seed_everything(seed)
    device = resolve_device(args.device)
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    model = model_from_checkpoint(payload).to(device)

    if args.smoke:
        fit_loader, _ = build_synthetic_loaders(
            examples=args.smoke_examples,
            num_classes=int(payload["num_classes"]),
            batch_size=batch_size,
            seed=seed,
        )
    else:
        fit_loader, _ = build_cifar100_loaders(
            args.data_root,
            split_seed=int(config["data"]["split_seed"]),
            selection_examples=int(config["data"]["selection_examples"]),
            batch_size=batch_size,
            num_workers=args.num_workers,
            download=args.download,
            source=args.data_source or config["data"]["source"],
        )

    requested_names = {str(record["group"]) for record in records}
    all_groups = {group.name: group for group in model.channel_groups()}
    missing_groups = requested_names - set(all_groups)
    if missing_groups:
        raise SystemExit(f"probe table names unknown model groups: {sorted(missing_groups)}")
    groups = [group for group in model.channel_groups() if group.name in requested_names]
    for record in records:
        group = all_groups[str(record["group"])]
        if str(record["gate"]) != group.gate:
            raise SystemExit(f"probe table gate does not match model group {group.name}")

    start = perf_counter()
    scores = collect_gate_sensitivity_scores(
        model,
        fit_loader,
        groups,
        device,
        group_size=group_size,
        max_batches=args.calibration_batches,
    )
    scoring_seconds = perf_counter() - start
    example_counts = {group_scores.examples for group_scores in scores.values()}
    if len(example_counts) != 1:
        raise RuntimeError(f"gates saw inconsistent example counts: {sorted(example_counts)}")

    for record in records:
        group_scores = scores[str(record["group"])]
        group_index = int(record["channel_group_index"])
        if not 0 <= group_index < group_scores.taylor.numel():
            raise SystemExit(f"probe group index is out of range: {record}")
        record["taylor"] = float(group_scores.taylor[group_index])
        record["group_fisher"] = float(group_scores.group_fisher[group_index])
        record["gradient_calibration_examples"] = group_scores.examples
        record["gradient_scoring_seconds"] = scoring_seconds

    output = args.output or args.table.with_name(
        args.table.stem + "_strong_baselines" + args.table.suffix
    )
    write_probe_csv(output, records)
    print(
        f"wrote {len(records)} rescored probes to {output}; "
        f"device={device} batches={args.calibration_batches} "
        f"examples={example_counts.pop()} gradient_scoring={scoring_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
