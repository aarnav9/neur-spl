#!/usr/bin/env python3
"""Add matrix-Fisher spectral leverage scores to a cached deletion table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from cut_to_grow.diagnostics import write_probe_csv
from cut_to_grow.pilot import (
    build_cifar100_loaders,
    build_synthetic_loaders,
    load_checkpoint,
    model_from_checkpoint,
    resolve_device,
    seed_everything,
)
from cut_to_grow.spectral_fisher import (
    collect_gate_gradient_matrices,
    model_spectral_fisher_scores,
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
    parser.add_argument("--ridge", type=float)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-examples", type=int, default=64)
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

    # Score every residual-branch group in one shared operator, even if the
    # supplied table contains only conv1 or conv2 interventions.
    groups = list(model.channel_groups())[1:]
    known_names = {group.name for group in groups}
    unknown = {str(record["group"]) for record in records} - known_names
    if unknown:
        raise SystemExit(f"probe table contains unknown groups: {sorted(unknown)}")

    start = perf_counter()
    matrices = collect_gate_gradient_matrices(
        model,
        fit_loader,
        groups,
        device,
        group_size=group_size,
        max_batches=args.calibration_batches,
    )
    calibration_seconds = perf_counter() - start
    start = perf_counter()
    scores = model_spectral_fisher_scores(
        matrices,
        groups,
        group_size=group_size,
        ridge=args.ridge,
    )
    scoring_seconds = perf_counter() - start

    for record in records:
        key = (str(record["group"]), int(record["channel_group_index"]))
        if key not in scores:
            raise SystemExit(f"probe group is out of range: {key}")
        score = scores[key]
        record["matrix_fisher_trace"] = score.fisher_trace
        record["spectral_fisher_leverage"] = score.ridge_leverage
        record["spectral_fisher_ridge"] = score.ridge
        record["spectral_fisher_examples"] = score.examples
        record["spectral_fisher_calibration_seconds"] = calibration_seconds
        record["spectral_fisher_scoring_seconds"] = scoring_seconds

    output = args.output or args.table.with_name(
        args.table.stem + "_spectral_fisher" + args.table.suffix
    )
    write_probe_csv(output, records)
    print(
        f"wrote {len(records)} matrix-Fisher probes to {output}; device={device} "
        f"batches={args.calibration_batches} examples={next(iter(scores.values())).examples} "
        f"ridge={next(iter(scores.values())).ridge:.8g} "
        f"calibration={calibration_seconds:.3f}s scoring={scoring_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
