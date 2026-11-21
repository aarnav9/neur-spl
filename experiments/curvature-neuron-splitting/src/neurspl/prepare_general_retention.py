#!/usr/bin/env python3
"""Download and materialize small deterministic general-retention subsets."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from neurspl.evaluate import ROOT, progress


DATASETS = {
    "hellaswag": {
        "repo": "Rowan/hellaswag",
        "preferred_file": "data/validation-00000-of-00001.parquet",
    },
    "arc_challenge": {
        "repo": "allenai/ai2_arc",
        "preferred_file": "ARC-Challenge/validation-00000-of-00001.parquet",
    },
    "boolq": {
        "repo": "google/boolq",
        "preferred_file": "data/validation-00000-of-00001.parquet",
    },
}


def choose_validation_file(files: list[str], preferred: str) -> str:
    if preferred in files:
        return preferred
    candidates = [
        name
        for name in files
        if name.endswith(".parquet") and "validation" in name.lower()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not uniquely identify validation parquet; candidates={candidates}"
        )
    return candidates[0]


def materialize_hellaswag(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        label = int(row["label"])
        choices = list(row["endings"])
        if not 0 <= label < len(choices):
            continue
        output.append(
            {
                "example_id": f"hellaswag:validation:{row['ind']}",
                "benchmark": "hellaswag",
                "context": "",
                "question": row["ctx"],
                "choices": choices,
                "answer_index": label,
            }
        )
    return output


def materialize_arc(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        choices = row["choices"]
        labels = [str(value) for value in choices["label"]]
        answer = str(row["answerKey"])
        if answer not in labels:
            continue
        output.append(
            {
                "example_id": f"arc_challenge:validation:{row['id']}",
                "benchmark": "arc_challenge",
                "context": "",
                "question": row["question"],
                "choices": list(choices["text"]),
                "answer_index": labels.index(answer),
            }
        )
    return output


def materialize_boolq(rows: list[dict]) -> list[dict]:
    return [
        {
            "example_id": f"boolq:validation:{index}",
            "benchmark": "boolq",
            "context": row["passage"],
            "question": row["question"],
            "choices": ["No", "Yes"],
            "answer_index": int(bool(row["answer"])),
        }
        for index, row in enumerate(rows)
    ]


MATERIALIZERS = {
    "hellaswag": materialize_hellaswag,
    "arc_challenge": materialize_arc,
    "boolq": materialize_boolq,
}


def deterministic_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    if size <= 0 or size > len(rows):
        raise ValueError(f"Invalid subset size {size} for {len(rows)} available rows")
    generator = random.Random(seed)
    indices = sorted(generator.sample(range(len(rows)), size))
    return [rows[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "datasets" / "general_retention"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "subsets" / "general_retention_v1" / "eval_192.jsonl",
    )
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api = HfApi()
    combined = []
    sources = {}
    for task_index, (task, specification) in enumerate(DATASETS.items()):
        repo = specification["repo"]
        progress(f"resolving {task} dataset revision from {repo}")
        info = api.dataset_info(repo)
        files = api.list_repo_files(repo, repo_type="dataset", revision=info.sha)
        filename = choose_validation_file(files, specification["preferred_file"])
        progress(f"downloading {task}: {filename} at {info.sha[:12]}")
        local_path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type="dataset",
                revision=info.sha,
                local_dir=args.raw_dir / task,
            )
        )
        raw_rows = pq.read_table(local_path).to_pylist()
        usable = MATERIALIZERS[task](raw_rows)
        sampled = deterministic_sample(
            usable, args.examples_per_task, args.seed + task_index
        )
        combined.extend(sampled)
        sources[task] = {
            "repo": repo,
            "revision": info.sha,
            "file": filename,
            "available_examples": len(usable),
            "selected_examples": len(sampled),
            "sample_seed": args.seed + task_index,
        }
        progress(
            f"prepared {task}: selected={len(sampled)} available={len(usable)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "paired general-capability retention development subset",
        "output": str(args.output.resolve()),
        "examples": len(combined),
        "examples_per_task": args.examples_per_task,
        "seed": args.seed,
        "sources": sources,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    progress(f"complete: saved {args.output}")
    progress(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
