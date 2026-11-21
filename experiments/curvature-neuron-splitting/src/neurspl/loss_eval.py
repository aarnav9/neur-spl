#!/usr/bin/env python3
"""Fast teacher-forced solution-loss evaluation for development experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurspl.evaluate import ROOT, SYSTEM_PROMPT, select_device, select_dtype, synchronize


DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "dev_v1" / "validation_128.jsonl"


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def formatted_solution(row: dict) -> str:
    reasoning = row["solution"].rsplit("####", 1)[0].strip()
    return f"{reasoning}\nFinal answer: {row['answer']}"


def tokenize_example(tokenizer, row: dict, max_length: int) -> tuple[list[int], list[int]]:
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question"]},
    ]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": formatted_solution(row)},
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    common = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        common += 1
    full_ids = full_ids[:max_length]
    labels = [-100] * min(common, len(full_ids)) + full_ids[common:]
    if not any(value != -100 for value in labels):
        raise ValueError(f"No target tokens remained for {row['example_id']}")
    return full_ids, labels


def collate(tokenizer, encoded: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
    maximum = max(len(input_ids) for input_ids, _ in encoded)
    input_rows = []
    label_rows = []
    masks = []
    for input_ids, labels in encoded:
        padding = maximum - len(input_ids)
        input_rows.append(input_ids + [tokenizer.pad_token_id] * padding)
        label_rows.append(labels + [-100] * padding)
        masks.append([1] * len(input_ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    rows = load_rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=False
    ).to(device)
    model.eval()
    encoded = [tokenize_example(tokenizer, row, args.max_length) for row in rows]

    total_nll = 0.0
    total_tokens = 0
    per_example = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch = collate(tokenizer, encoded[start : start + args.batch_size])
        batch = {key: value.to(device) for key, value in batch.items()}
        synchronize(device)
        with torch.inference_mode():
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
        synchronize(device)
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch["labels"][:, 1:]
        losses = functional.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels, reduction="none", ignore_index=-100
        )
        mask = shift_labels.ne(-100)
        for index, row in enumerate(batch_rows):
            count = int(mask[index].sum().item())
            nll = float(losses[index][mask[index]].sum().item())
            total_tokens += count
            total_nll += nll
            per_example.append(
                {
                    "example_id": row["example_id"],
                    "target_tokens": count,
                    "nll": nll,
                    "mean_nll": nll / count,
                }
            )
        print(f"batch {start // args.batch_size + 1}: {len(per_example)}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - started
    mean_nll = total_nll / total_tokens
    result = {
        "metadata": {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "device": device,
            "dtype": str(dtype),
            "examples": len(rows),
            "target_tokens": total_tokens,
            "mean_nll": mean_nll,
            "perplexity": math.exp(min(mean_nll, 50)),
            "elapsed_seconds": elapsed,
        },
        "records": per_example,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()

