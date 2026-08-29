#!/usr/bin/env python3
"""Measure channel scores against actual checkpoint deletion damage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from cut_to_grow.block_flow import model_residual_flow_scores
from cut_to_grow.diagnostics import (
    collect_transport_weight_scores,
    output_channel_transport_scores,
    probe_channel_deletions,
    probe_group_deletions,
    write_probe_csv,
)
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
    parser.add_argument("--data-root", type=Path, default=Path("pull/raw"))
    parser.add_argument("--data-source", choices=("torchvision", "huggingface"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--group", action="append", dest="groups")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--channels-per-score", type=int, default=4)
    parser.add_argument("--deletion-group-size", type=int)
    parser.add_argument("--groups-per-score", type=int)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--evaluation-batches", type=int)
    parser.add_argument("--reuse-probes-from", type=Path, action="append")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-examples", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_checkpoint(args.checkpoint)
    config = payload["config"]
    seed = int(payload["seed"])
    seed_everything(seed)
    device = resolve_device(args.device)
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    model = model_from_checkpoint(payload).to(device)

    if args.smoke:
        fit_loader, selection_loader = build_synthetic_loaders(
            examples=args.smoke_examples,
            num_classes=int(payload["num_classes"]),
            batch_size=batch_size,
            seed=seed,
        )
    else:
        fit_loader, selection_loader = build_cifar100_loaders(
            args.data_root,
            split_seed=int(config["data"]["split_seed"]),
            selection_examples=int(config["data"]["selection_examples"]),
            batch_size=batch_size,
            num_workers=args.num_workers,
            download=args.download,
            source=args.data_source or config["data"]["source"],
        )

    branch_groups = list(model.channel_groups())[1:]
    groups = branch_groups
    if args.groups:
        requested = set(args.groups)
        groups = [group for group in groups if group.name in requested]
        missing = requested - {group.name for group in groups}
        if missing:
            raise SystemExit(f"unknown groups: {sorted(missing)}")
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise SystemExit("no channel groups selected")

    start = perf_counter()
    weight_transport = collect_transport_weight_scores(
        model,
        fit_loader,
        branch_groups,
        device,
        max_batches=args.calibration_batches,
        decay=float(config["capacity"]["ema_decay"]),
    )
    transport_seconds = perf_counter() - start
    transport = output_channel_transport_scores(weight_transport, groups)
    start = perf_counter()
    flow_group_size = int(
        args.deletion_group_size or config["capacity"]["flow_channel_group_size"]
    )
    flow = model_residual_flow_scores(
        model,
        weight_transport,
        group_size=flow_group_size,
    )
    flow_seconds = perf_counter() - start
    start = perf_counter()
    cached_records = None
    if args.reuse_probes_from is not None:
        cached_records = []
        for cache_path in args.reuse_probes_from:
            with cache_path.open(newline="", encoding="utf-8") as source:
                cached_records.extend(csv.DictReader(source))
    if args.deletion_group_size is None:
        records = probe_channel_deletions(
            model,
            selection_loader,
            groups,
            transport,
            flow,
            device,
            channels_per_score=args.channels_per_score,
            max_batches=args.evaluation_batches,
            cached_records=cached_records,
        )
    else:
        records = probe_group_deletions(
            model,
            selection_loader,
            groups,
            transport,
            flow,
            device,
            group_size=args.deletion_group_size,
            groups_per_score=int(
                args.groups_per_score
                or config["diagnostic"]["groups_per_layer_quantile"]
            ),
            max_batches=args.evaluation_batches,
            cached_records=cached_records,
        )
    probe_seconds = perf_counter() - start
    for record in records:
        record["transport_calibration_seconds"] = transport_seconds
        record["flow_scoring_seconds"] = flow_seconds
        record["deletion_probe_seconds"] = probe_seconds
    suffix = (
        "_probes.csv"
        if args.deletion_group_size is None
        else f"_group{args.deletion_group_size:02d}_probes.csv"
    )
    output = args.output or args.checkpoint.with_name(args.checkpoint.stem + suffix)
    write_probe_csv(output, records)
    reused = sum(int(record["measurement_reused"]) for record in records)
    print(
        f"wrote {len(records)} deletion probes to {output}; "
        f"reused={reused} evaluated={len(records) - reused} "
        f"transport={transport_seconds:.3f}s flow={flow_seconds:.3f}s "
        f"interventions={probe_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
