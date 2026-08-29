#!/usr/bin/env python3
"""List or download explicitly declared vision datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


CIFAR100_HF_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument("--manifest", default="pull/manifest.yaml")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"))
    parser.add_argument("--backend", choices=("torchvision", "huggingface"), default="torchvision")
    parser.add_argument("--root", default="pull/raw")
    return parser.parse_args()


def list_manifest(path: Path) -> None:
    manifest = yaml.safe_load(path.read_text())
    for name, record in manifest["datasets"].items():
        automatic = "automatic" if record["automatic"] else "manual"
        print(f"{name:12s} {automatic:9s} {record['role']}")


def download(dataset: str, root: Path) -> None:
    try:
        from torchvision import datasets
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise SystemExit("torchvision is required; install the code project first") from exc

    root.mkdir(parents=True, exist_ok=True)
    dataset_class = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    dataset_class(root=str(root), train=True, download=True)
    dataset_class(root=str(root), train=False, download=True)
    print(f"downloaded and verified {dataset} under {root}")


def download_huggingface(dataset: str, root: Path) -> None:
    if dataset != "cifar100":
        raise SystemExit("the Hugging Face backend is currently declared only for cifar100")
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:  # pragma: no cover - optional environment
        raise SystemExit("install the project with code[hf]") from exc
    saved_path = root / "cifar100_hf_train"
    if saved_path.exists():
        loaded = load_from_disk(str(saved_path))
    else:
        loaded = load_dataset(
            "uoft-cs/cifar100",
            split="train",
            revision=CIFAR100_HF_REVISION,
            cache_dir=str(root / "huggingface"),
        )
        loaded.save_to_disk(str(saved_path))
    if len(loaded) != 50000:
        raise RuntimeError(f"expected 50000 CIFAR-100 train examples, got {len(loaded)}")
    print(f"downloaded and verified cifar100 from Hugging Face under {saved_path}")


def main() -> None:
    args = parse_args()
    if args.list_only:
        list_manifest(Path(args.manifest))
        return
    if args.dataset is None:
        raise SystemExit("provide --dataset or use --list")
    if args.backend == "huggingface":
        download_huggingface(args.dataset, Path(args.root))
    else:
        download(args.dataset, Path(args.root))


if __name__ == "__main__":
    main()
