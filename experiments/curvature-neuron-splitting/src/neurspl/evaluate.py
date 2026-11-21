#!/usr/bin/env python3
"""Reproducible local baseline evaluation for GSM8K and Hendrycks MATH."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"

SYSTEM_PROMPT = (
    "Solve the mathematical problem accurately and concisely. Use at most five "
    "short lines and do not restate the problem. End with a separate line in the "
    "exact form: Final answer: <answer>"
)


def progress(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass(frozen=True)
class Example:
    example_id: str
    benchmark: str
    category: str
    question: str
    gold_answer: str


def extract_last_boxed(text: str) -> str | None:
    """Return the content of the last balanced \\boxed{...} expression."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
    return None


def normalize_math(text: str | None) -> str | None:
    if text is None:
        return None
    value = text.strip().strip("$")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\,", "").replace(" ", "")
    while value.endswith((".", ",")):
        value = value[:-1]
    return value


def extract_prediction(text: str) -> str | None:
    final_matches = re.findall(
        r"Final\s+answer\s*:\s*([^\n]+)", text, flags=re.IGNORECASE
    )
    if final_matches:
        candidate = final_matches[-1].strip()
        boxed = extract_last_boxed(candidate)
        return boxed if boxed is not None else candidate
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed
    numeric_matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?", text)
    return numeric_matches[-1] if numeric_matches else None


def normalize_number(text: str | None) -> Decimal | None:
    if text is None:
        return None
    value = text.strip().replace(",", "").replace("$", "")
    value = value.rstrip(". ")
    if value.endswith("%"):
        value = value[:-1]
    fraction = re.fullmatch(r"([-+]?\d+)\s*/\s*(\d+)", value)
    try:
        if fraction:
            return Decimal(fraction.group(1)) / Decimal(fraction.group(2))
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        return Decimal(match.group(0)) if match else None
    except (InvalidOperation, ZeroDivisionError):
        return None


def is_correct(benchmark: str, prediction: str | None, gold: str) -> bool:
    if benchmark == "gsm8k":
        return normalize_number(prediction) == normalize_number(gold)
    return normalize_math(prediction) == normalize_math(gold)


def load_gsm8k(root: Path) -> list[Example]:
    path = root / "datasets" / "gsm8k" / "main" / "test-00000-of-00001.parquet"
    rows = pq.read_table(path).to_pylist()
    return [
        Example(
            example_id=f"gsm8k:test:{index}",
            benchmark="gsm8k",
            category="main",
            question=row["question"],
            gold_answer=row["answer"].rsplit("####", 1)[-1].strip(),
        )
        for index, row in enumerate(rows)
    ]


def load_math(root: Path) -> list[Example]:
    dataset_root = root / "datasets" / "hendrycks_math"
    examples: list[Example] = []
    for path in sorted(dataset_root.glob("*/test-*.parquet")):
        category = path.parent.name
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            gold = extract_last_boxed(row["solution"])
            if gold is None:
                raise ValueError(f"No boxed answer in {category} test row {index}")
            examples.append(
                Example(
                    example_id=f"math:{category}:test:{index}",
                    benchmark="math",
                    category=category,
                    question=row["problem"],
                    gold_answer=gold,
                )
            )
    return examples


def load_jsonl(path: Path, benchmark: str) -> list[Example]:
    examples = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                Example(
                    example_id=row["example_id"],
                    benchmark=benchmark,
                    category=row.get("source", "jsonl"),
                    question=row["question"],
                    gold_answer=row["answer"],
                )
            )
    return examples


def choose_examples(examples: list[Example], limit: int, seed: int) -> list[Example]:
    if limit <= 0 or limit >= len(examples):
        return examples
    generator = random.Random(seed)
    indices = sorted(generator.sample(range(len(examples)), limit))
    return [examples[index] for index in indices]


def batches(items: list[Example], size: int) -> Iterable[list[Example]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_dtype(requested: str, device: str) -> torch.dtype:
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if requested != "auto":
        return choices[requested]
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        # PyTorch 2.2 MPS cannot execute bfloat16, and converting this checkpoint
        # to float16 caused repeated-token numerical collapse in diagnostics.
        return torch.float32
    return torch.float32


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    model_path = args.model.resolve()
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)

    loader = load_gsm8k if args.benchmark == "gsm8k" else load_math
    all_examples = load_jsonl(args.data, args.benchmark) if args.data else loader(root)
    if args.example_ids:
        by_id = {example.example_id: example for example in all_examples}
        requested_ids = [value.strip() for value in args.example_ids.split(",")]
        missing_ids = [value for value in requested_ids if value not in by_id]
        if missing_ids:
            raise ValueError(f"Unknown example IDs: {missing_ids}")
        examples = [by_id[value] for value in requested_ids]
    else:
        examples = choose_examples(all_examples, args.limit, args.seed)
    progress(
        f"starting generation evaluation: model={model_path.name} "
        f"examples={len(examples)} benchmark={args.benchmark} device={device} dtype={dtype}"
    )
    progress(f"loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to(device)
    model.eval()

    records: list[dict] = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(batches(examples, args.batch_size), start=1):
        prompts = [build_prompt(tokenizer, item.question) for item in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(device)
        synchronize(device)
        batch_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        synchronize(device)
        batch_seconds = time.perf_counter() - batch_started
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for item, completion, token_ids in zip(batch, completions, new_tokens):
            prediction = extract_prediction(completion)
            records.append(
                {
                    **asdict(item),
                    "prompt": build_prompt(tokenizer, item.question),
                    "completion": completion,
                    "prediction": prediction,
                    "correct": is_correct(item.benchmark, prediction, item.gold_answer),
                    "generated_tokens": int(
                        (token_ids != tokenizer.pad_token_id).sum().item()
                    ),
                    "batch_seconds": batch_seconds,
                }
            )
        correct_so_far = sum(record["correct"] for record in records)
        progress(
            f"batch {batch_index}: {len(records)}/{len(examples)}, "
            f"accuracy={correct_so_far / len(records):.3f}, "
            f"batch_seconds={batch_seconds:.1f}"
        )

    elapsed = time.perf_counter() - started
    correct = sum(record["correct"] for record in records)
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "model_path": str(model_path),
        "model_name": model_path.name,
        "data": str(args.data.resolve()) if args.data else None,
        "device": device,
        "dtype": str(dtype),
        "seed": args.seed,
        "limit": args.limit,
        "requested_example_ids": args.example_ids,
        "num_examples": len(records),
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "prompt": SYSTEM_PROMPT,
        "correct": correct,
        "accuracy": correct / len(records),
        "elapsed_seconds": elapsed,
        "generated_tokens": sum(record["generated_tokens"] for record in records),
    }
    result = {"metadata": metadata, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    progress(f"complete: saved {args.output}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, help="Optional materialized JSONL subset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--example-ids",
        help="Comma-separated exact example IDs; overrides --limit and --seed selection",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
