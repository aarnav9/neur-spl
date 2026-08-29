#!/usr/bin/env python3
"""Train the dense CIFAR-100 model used by the deletion diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cut_to_grow.model import cifar_resnet18
from cut_to_grow.pilot import (
    build_cifar100_loaders,
    build_synthetic_loaders,
    load_checkpoint,
    model_from_checkpoint,
    resolve_device,
    run_dense_training,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("code/configs/pilot.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("pull/raw"))
    parser.add_argument("--data-source", choices=("torchvision", "huggingface"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-examples", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.epochs is not None and not args.smoke:
        raise SystemExit("--epochs may only shorten synthetic smoke runs")
    resume = None if args.resume is None else load_checkpoint(args.resume)
    if resume is not None:
        config = resume["config"]
    declared_seeds = [int(seed) for seed in config["training"]["seeds"]]
    if args.seed is None:
        seed = int(resume["seed"]) if resume is not None else declared_seeds[0]
    else:
        seed = args.seed
    if not args.smoke and seed not in declared_seeds:
        raise SystemExit(f"seed {seed} is not declared in {args.config}")
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    output = args.output or (
        args.resume.parent
        if args.resume is not None
        else Path("runs") / f"dense_cifar100_seed_{seed}"
    )
    if args.resume is None and output.exists() and any(output.iterdir()):
        raise SystemExit(f"output is not empty: {output}")

    seed_everything(seed)
    device = resolve_device(args.device)
    if args.smoke:
        fit_loader, selection_loader = build_synthetic_loaders(
            examples=args.smoke_examples,
            num_classes=10,
            batch_size=batch_size,
            seed=seed,
        )
        model = (
            model_from_checkpoint(resume)
            if resume is not None
            else cifar_resnet18(num_classes=10)
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
        model = (
            model_from_checkpoint(resume)
            if resume is not None
            else cifar_resnet18(num_classes=100)
        )

    records = run_dense_training(
        model,
        fit_loader,
        selection_loader,
        output=output,
        config=config,
        seed=seed,
        device=device,
        epochs=args.epochs,
        stop_after_epoch=args.stop_after_epoch,
        resume=resume,
    )
    final = records[-1]
    print(
        f"finished epoch={final['epoch']} device={device} "
        f"selection_accuracy={final['selection']['accuracy']:.4f} output={output}"
    )


if __name__ == "__main__":
    main()
