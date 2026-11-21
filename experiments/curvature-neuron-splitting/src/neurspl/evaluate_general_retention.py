#!/usr/bin/env python3
"""Paired multiple-choice retention evaluation for original and split models."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurspl.curvature_proxy import log
from neurspl.evaluate import ROOT, select_device, select_dtype, synchronize
from neurspl.evaluate_split_generation import exact_mcnemar_two_sided
from neurspl.split import split_swiglu_layer
from neurspl.split_line_search import set_separation


DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "general_retention_v1" / "eval_192.jsonl"
DEFAULT_REFINEMENT = ROOT / "runs" / "curvature_refine_qwen_layer12_refine.json"
SYSTEM_PROMPT = (
    "Answer the multiple-choice question. Respond with only the letter of the "
    "best answer and no explanation."
)


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def format_question(row: dict) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(
        f"{letters[index]}. {choice}" for index, choice in enumerate(row["choices"])
    )
    parts = []
    if row.get("context"):
        parts.append(f"Passage:\n{row['context']}")
    parts.append(f"Question:\n{row['question']}")
    parts.append(f"Choices:\n{options}")
    return "\n\n".join(parts)


def encode_candidates(tokenizer, rows: list[dict], max_length: int) -> list[dict]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    candidates = []
    for example_index, row in enumerate(rows):
        if not 2 <= len(row["choices"]) <= len(letters):
            raise ValueError(f"Unsupported choice count in {row['example_id']}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_question(row)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for choice_index in range(len(row["choices"])):
            answer = letters[choice_index]
            full_ids = tokenizer(
                prompt + answer, add_special_tokens=False
            )["input_ids"]
            common = 0
            for left, right in zip(prompt_ids, full_ids):
                if left != right:
                    break
                common += 1
            answer_ids = full_ids[common:]
            if not answer_ids:
                raise ValueError(f"No answer tokens for {row['example_id']} choice {answer}")
            if len(full_ids) > max_length:
                removed = len(full_ids) - max_length
                full_ids = full_ids[removed:]
                common = max(0, common - removed)
            labels = [-100] * common + full_ids[common:]
            if not any(value != -100 for value in labels):
                raise ValueError(f"Answer truncated for {row['example_id']}")
            candidates.append(
                {
                    "example_index": example_index,
                    "choice_index": choice_index,
                    "input_ids": full_ids,
                    "labels": labels,
                }
            )
    return candidates


def collate_candidates(tokenizer, candidates: list[dict]) -> dict[str, torch.Tensor]:
    maximum = max(len(item["input_ids"]) for item in candidates)
    input_rows = []
    label_rows = []
    masks = []
    for item in candidates:
        padding = maximum - len(item["input_ids"])
        input_rows.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
        label_rows.append(item["labels"] + [-100] * padding)
        masks.append([1] * len(item["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def evaluate_model(
    model,
    tokenizer,
    rows: list[dict],
    encoded: list[dict],
    batch_size: int,
    device: str,
    label: str,
) -> tuple[list[dict], float]:
    score_by_example: dict[int, list[tuple[int, float]]] = defaultdict(list)
    batch_count = math.ceil(len(encoded) / batch_size)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch_number = start // batch_size + 1
            batch_started = time.perf_counter()
            items = encoded[start : start + batch_size]
            batch = collate_candidates(tokenizer, items)
            batch = {key: value.to(device) for key, value in batch.items()}
            synchronize(device)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = batch["labels"][:, 1:]
            losses = functional.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction="none",
                ignore_index=-100,
            )
            mask = shift_labels.ne(-100)
            for item_index, item in enumerate(items):
                token_count = int(mask[item_index].sum().item())
                score = -float(losses[item_index][mask[item_index]].sum().item())
                score_by_example[item["example_index"]].append(
                    (item["choice_index"], score / token_count)
                )
            synchronize(device)
            log(
                f"{label} batch {batch_number}/{batch_count} complete; "
                f"candidates={min(start + batch_size, len(encoded))}/{len(encoded)} "
                f"batch_seconds={time.perf_counter() - batch_started:.1f}"
            )
            del batch, logits, shift_logits, shift_labels, losses, mask
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    records = []
    for example_index, row in enumerate(rows):
        ordered = sorted(score_by_example[example_index])
        scores = [score for _, score in ordered]
        prediction = max(range(len(scores)), key=scores.__getitem__)
        records.append(
            {
                "example_id": row["example_id"],
                "benchmark": row["benchmark"],
                "answer_index": int(row["answer_index"]),
                "prediction_index": prediction,
                "correct": prediction == int(row["answer_index"]),
                "choice_scores": scores,
            }
        )
    return records, time.perf_counter() - started


def paired_counts(original: list[dict], split: list[dict]) -> dict[str, Any]:
    both = original_only = split_only = neither = changed = 0
    for before, after in zip(original, split):
        if before["example_id"] != after["example_id"]:
            raise ValueError("Original and split records are not aligned")
        if before["correct"] and after["correct"]:
            both += 1
        elif before["correct"]:
            original_only += 1
        elif after["correct"]:
            split_only += 1
        else:
            neither += 1
        changed += before["prediction_index"] != after["prediction_index"]
    return {
        "both_correct": both,
        "original_only_correct": original_only,
        "split_only_correct": split_only,
        "both_wrong": neither,
        "changed_prediction": changed,
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(
            original_only, split_only
        ),
    }


def summarize(original: list[dict], split: list[dict]) -> dict[str, dict]:
    tasks = sorted({record["benchmark"] for record in original})
    summaries = {}
    for task in [*tasks, "overall"]:
        if task == "overall":
            before = original
            after = split
        else:
            before = [record for record in original if record["benchmark"] == task]
            after = [record for record in split if record["benchmark"] == task]
        original_correct = sum(record["correct"] for record in before)
        split_correct = sum(record["correct"] for record in after)
        summaries[task] = {
            "examples": len(before),
            "original_correct": original_correct,
            "original_accuracy": original_correct / len(before),
            "split_correct": split_correct,
            "split_accuracy": split_correct / len(before),
            "accuracy_delta": (split_correct - original_correct) / len(before),
            **paired_counts(before, after),
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--refinement", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    rows = load_rows(args.data)
    refinement = json.loads(args.refinement.read_text())
    candidates = refinement["refined_candidates"][: args.top_k]
    if len(candidates) != args.top_k:
        raise ValueError("The refinement file does not contain enough candidates")
    layer_index = int(refinement["metadata"]["layer"])
    direction_by_neuron = {
        int(item["neuron_index"]): (item["direction_gate"], item["direction_up"])
        for item in candidates
    }
    indices = [int(item["neuron_index"]) for item in candidates]
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    log(
        f"starting paired general retention: examples={len(rows)} tasks="
        f"{sorted({row['benchmark'] for row in rows})} epsilon={args.epsilon:g} "
        f"device={device} dtype={dtype}"
    )
    log(f"loading tokenizer and model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = encode_candidates(tokenizer, rows, args.max_length)
    log(f"prepared {len(encoded)} answer candidates for {len(rows)} examples")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    parameters_before = sum(parameter.numel() for parameter in model.parameters())

    original_records, original_seconds = evaluate_model(
        model, tokenizer, rows, encoded, args.batch_size, device, "original"
    )
    original_correct = sum(record["correct"] for record in original_records)
    log(
        f"original complete: correct={original_correct}/{len(rows)} "
        f"accuracy={original_correct / len(rows):.4f}"
    )

    split_record = split_swiglu_layer(model, layer_index, indices)
    mlp = model.model.layers[layer_index].mlp
    originals = torch.tensor(split_record.original_indices, device=device)
    gate_centers = mlp.gate_proj.weight[originals].detach().clone()
    up_centers = mlp.up_proj.weight[originals].detach().clone()
    gate_directions = torch.tensor(
        [direction_by_neuron[index][0] for index in split_record.original_indices],
        device=device,
        dtype=mlp.gate_proj.weight.dtype,
    )
    up_directions = torch.tensor(
        [direction_by_neuron[index][1] for index in split_record.original_indices],
        device=device,
        dtype=mlp.up_proj.weight.dtype,
    )
    set_separation(
        mlp,
        split_record.original_indices,
        split_record.duplicate_indices,
        gate_centers,
        up_centers,
        gate_directions,
        up_directions,
        args.epsilon,
    )
    parameters_after = sum(parameter.numel() for parameter in model.parameters())
    split_records, split_seconds = evaluate_model(
        model, tokenizer, rows, encoded, args.batch_size, device, "curvature_split"
    )
    summaries = summarize(original_records, split_records)
    result = {
        "metadata": {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "paired_letter_likelihood_general_retention",
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "refinement": str(args.refinement.resolve()),
            "layer": layer_index,
            "device": device,
            "dtype": str(dtype),
            "examples": len(rows),
            "answer_candidates": len(encoded),
            "top_k": args.top_k,
            "epsilon": args.epsilon,
            "selected_neurons": list(split_record.original_indices),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "scoring": "mean_log_likelihood_of_answer_letter",
            "prompt": SYSTEM_PROMPT,
            "parameters_before": parameters_before,
            "parameters_after": parameters_after,
            "added_parameters": parameters_after - parameters_before,
            "original_elapsed_seconds": original_seconds,
            "split_elapsed_seconds": split_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summaries": summaries,
        "original_records": original_records,
        "split_records": split_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    log(f"complete: {json.dumps(summaries, sort_keys=True)}")
    log(f"saved {args.output}")


if __name__ == "__main__":
    main()
