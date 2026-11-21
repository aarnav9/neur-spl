#!/usr/bin/env python3
"""Create deterministic, disjoint small subsets while retaining all source data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(row: dict, example_id: str, split: str) -> dict:
    return {
        "example_id": example_id,
        "source": "openai/gsm8k:main",
        "source_split": split,
        "question": row["question"],
        "solution": row["answer"],
        "answer": row["answer"].rsplit("####", 1)[-1].strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "subsets" / "dev_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--curvature-size", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=32)
    args = parser.parse_args()

    train_path = (
        args.root / "datasets" / "gsm8k" / "main" / "train-00000-of-00001.parquet"
    )
    test_path = (
        args.root / "datasets" / "gsm8k" / "main" / "test-00000-of-00001.parquet"
    )
    train_rows = pq.read_table(train_path).to_pylist()
    test_rows = pq.read_table(test_path).to_pylist()
    generator = random.Random(args.seed)
    train_indices = list(range(len(train_rows)))
    test_indices = list(range(len(test_rows)))
    generator.shuffle(train_indices)
    generator.shuffle(test_indices)

    required = args.train_size + args.validation_size
    if required > len(train_rows):
        raise ValueError("Requested train and validation subsets exceed GSM8K train")
    if args.curvature_size > args.train_size:
        raise ValueError("Curvature subset must be contained within the training subset")

    selected_train = train_indices[: args.train_size]
    selected_validation = train_indices[args.train_size : required]
    selected_test = test_indices[: args.test_size]
    selected_curvature = selected_train[: args.curvature_size]

    subsets = {
        f"train_{args.train_size}.jsonl": [
            convert(train_rows[index], f"gsm8k:train:{index}", "train")
            for index in selected_train
        ],
        f"curvature_{args.curvature_size}.jsonl": [
            convert(train_rows[index], f"gsm8k:train:{index}", "train")
            for index in selected_curvature
        ],
        f"validation_{args.validation_size}.jsonl": [
            convert(train_rows[index], f"gsm8k:train:{index}", "train")
            for index in selected_validation
        ],
        f"test_{args.test_size}.jsonl": [
            convert(test_rows[index], f"gsm8k:test:{index}", "test")
            for index in selected_test
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in subsets.items():
        write_jsonl(args.output / filename, rows)

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "source": "openai/gsm8k:main",
        "policy": {
            "curvature_is_subset_of_train": True,
            "validation_is_disjoint_from_train": True,
            "test_uses_original_test_split": True,
            "full_source_dataset_retained": True,
        },
        "files": {filename: len(rows) for filename, rows in subsets.items()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
