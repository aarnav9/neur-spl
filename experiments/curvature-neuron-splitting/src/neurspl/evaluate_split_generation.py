#!/usr/bin/env python3
"""Paired greedy-generation evaluation of an original and curvature-split model."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurspl.curvature_proxy import log
from neurspl.evaluate import (
    ROOT,
    SYSTEM_PROMPT,
    batches,
    build_prompt,
    extract_prediction,
    is_correct,
    load_jsonl,
    select_device,
    select_dtype,
    synchronize,
)
from neurspl.split import split_swiglu_layer
from neurspl.split_line_search import set_separation
from neurspl.trainable_split import load_adapter_checkpoint


DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "micro_v1" / "test_16.jsonl"
DEFAULT_REFINEMENT = ROOT / "runs" / "curvature_refine_first_refine.json"


def exact_mcnemar_two_sided(original_only: int, split_only: int) -> float:
    """Two-sided exact sign-test p-value for discordant paired outcomes."""
    discordant = original_only + split_only
    if discordant == 0:
        return 1.0
    smaller = min(original_only, split_only)
    lower_tail = sum(math.comb(discordant, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def paired_summary(original: list[dict], split: list[dict]) -> dict[str, Any]:
    if [item["example_id"] for item in original] != [
        item["example_id"] for item in split
    ]:
        raise ValueError("Original and split records are not aligned")
    both_correct = 0
    original_only = 0
    split_only = 0
    both_wrong = 0
    changed_completion = 0
    changed_prediction = 0
    for before, after in zip(original, split):
        if before["correct"] and after["correct"]:
            both_correct += 1
        elif before["correct"]:
            original_only += 1
        elif after["correct"]:
            split_only += 1
        else:
            both_wrong += 1
        changed_completion += before["completion"] != after["completion"]
        changed_prediction += before["prediction"] != after["prediction"]
    return {
        "both_correct": both_correct,
        "original_only_correct": original_only,
        "split_only_correct": split_only,
        "both_wrong": both_wrong,
        "changed_completion": changed_completion,
        "changed_prediction": changed_prediction,
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(
            original_only, split_only
        ),
    }


def load_reusable_baseline(
    path: Path,
    *,
    model_path: Path,
    data_path: Path,
    benchmark: str,
    examples,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Load a baseline only if it exactly matches this generation experiment."""
    result = json.loads(path.read_text())
    metadata = result.get("metadata", {})
    records = result.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Baseline result has no records list: {path}")

    expected = {
        "benchmark": benchmark,
        "batch_size": batch_size,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "prompt": SYSTEM_PROMPT,
    }
    mismatches = [
        f"{key}: saved={metadata.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    saved_model = metadata.get("model_path")
    saved_data = metadata.get("data")
    if saved_model is None or Path(saved_model).resolve() != model_path.resolve():
        mismatches.append(
            f"model_path: saved={saved_model!r}, requested={str(model_path.resolve())!r}"
        )
    if saved_data is None or Path(saved_data).resolve() != data_path.resolve():
        mismatches.append(
            f"data: saved={saved_data!r}, requested={str(data_path.resolve())!r}"
        )

    expected_ids = [item.example_id for item in examples]
    saved_ids = [item.get("example_id") for item in records]
    if saved_ids != expected_ids:
        mismatches.append("record example IDs/order do not match the requested data")
    if len(records) != len(examples):
        mismatches.append(
            f"record count: saved={len(records)}, requested={len(examples)}"
        )
    else:
        for index, (record, example) in enumerate(zip(records, examples)):
            if (
                record.get("question") != example.question
                or record.get("gold_answer") != example.gold_answer
            ):
                mismatches.append(
                    f"record {index} question or gold answer does not match requested data"
                )
                break
    if mismatches:
        raise ValueError(
            "Cannot reuse an incompatible generation baseline:\n- "
            + "\n- ".join(mismatches)
        )
    return records, metadata


def generate_records(
    model,
    tokenizer,
    examples,
    device: str,
    label: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[list[dict], float]:
    records: list[dict] = []
    started = time.perf_counter()
    batch_count = (len(examples) + batch_size - 1) // batch_size
    for batch_index, batch in enumerate(batches(examples, batch_size), start=1):
        prompts = [build_prompt(tokenizer, item.question) for item in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)
        synchronize(device)
        batch_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
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
        for item, prompt, completion, token_ids in zip(
            batch, prompts, completions, new_tokens
        ):
            prediction = extract_prediction(completion)
            records.append(
                {
                    **asdict(item),
                    "prompt": prompt,
                    "completion": completion,
                    "prediction": prediction,
                    "correct": is_correct(item.benchmark, prediction, item.gold_answer),
                    "generated_tokens": int(
                        (token_ids != tokenizer.pad_token_id).sum().item()
                    ),
                    "batch_seconds": batch_seconds,
                }
            )
        correct = sum(record["correct"] for record in records)
        log(
            f"{label} batch {batch_index}/{batch_count} complete; "
            f"examples={len(records)}/{len(examples)} "
            f"accuracy={correct / len(records):.4f} "
            f"batch_seconds={batch_seconds:.1f}"
        )
        del encoded, generated, new_tokens
    return records, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--refinement", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Evaluate a saved localized-training adapter instead of a static split",
    )
    parser.add_argument(
        "--baseline-result",
        type=Path,
        help="Reuse a compatible saved baseline and generate only the split model",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("gsm8k", "math"), default="gsm8k")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()
    if args.epsilon < 0:
        raise ValueError("epsilon must be nonnegative")

    total_started = time.perf_counter()
    candidates = None
    direction_by_neuron = None
    indices = None
    layer_index = None
    if args.adapter is None:
        refinement = json.loads(args.refinement.read_text())
        candidates = refinement["refined_candidates"][: args.top_k]
        if len(candidates) != args.top_k:
            raise ValueError("The refinement file does not contain enough candidates")
        layer_index = int(refinement["metadata"]["layer"])
        direction_by_neuron = {
            int(item["neuron_index"]): (
                item["direction_gate"],
                item["direction_up"],
            )
            for item in candidates
        }
        indices = [int(item["neuron_index"]) for item in candidates]
    examples = load_jsonl(args.data, args.benchmark)
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    if args.adapter is None:
        log(
            f"starting paired generation: benchmark={args.benchmark} examples={len(examples)} "
            f"top_k={args.top_k} epsilon={args.epsilon:g} device={device} dtype={dtype}"
        )
        log(f"refinement input: {args.refinement}")
    else:
        log(
            f"starting paired adapter generation: benchmark={args.benchmark} "
            f"examples={len(examples)} adapter={args.adapter} device={device} dtype={dtype}"
        )
    baseline_metadata = None
    original_records = None
    if args.baseline_result is not None:
        original_records, baseline_metadata = load_reusable_baseline(
            args.baseline_result,
            model_path=args.model,
            data_path=args.data,
            benchmark=args.benchmark,
            examples=examples,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        log(
            f"reusing compatible baseline: {args.baseline_result} "
            f"({sum(record['correct'] for record in original_records)}/{len(examples)} correct)"
        )
    log(f"loading tokenizer and model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=False
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())

    if original_records is None:
        original_records, original_seconds = generate_records(
            model,
            tokenizer,
            examples,
            device,
            "original",
            args.batch_size,
            args.max_input_tokens,
            args.max_new_tokens,
        )
        original_elapsed_this_run = original_seconds
    else:
        original_seconds = float(baseline_metadata["elapsed_seconds"])
        original_elapsed_this_run = 0.0
    original_correct = sum(record["correct"] for record in original_records)
    log(
        f"original generation complete: correct={original_correct}/{len(examples)} "
        f"accuracy={original_correct / len(examples):.4f}"
    )

    adapter_metadata = None
    if args.adapter is None:
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
        split_label = "curvature_split"
    else:
        _adapter, split_record, adapter_metadata = load_adapter_checkpoint(
            model, args.adapter
        )
        saved_model = adapter_metadata.get("model")
        if saved_model is not None and Path(saved_model).resolve() != args.model.resolve():
            raise ValueError(
                f"Adapter model mismatch: saved={saved_model}, requested={args.model.resolve()}"
            )
        layer_index = split_record.layer_index
        split_label = "trained_split_adapter"
        log(
            f"loaded adapter: layer={layer_index} "
            f"neurons={list(split_record.original_indices)} "
            f"training_status={adapter_metadata.get('status')}"
        )
    model.requires_grad_(False)
    parameter_count_after = sum(parameter.numel() for parameter in model.parameters())
    split_records, split_seconds = generate_records(
        model,
        tokenizer,
        examples,
        device,
        split_label,
        args.batch_size,
        args.max_input_tokens,
        args.max_new_tokens,
    )
    split_correct = sum(record["correct"] for record in split_records)
    paired = paired_summary(original_records, split_records)
    paired_records = []
    for before, after in zip(original_records, split_records):
        paired_records.append(
            {
                "example_id": before["example_id"],
                "question": before["question"],
                "gold_answer": before["gold_answer"],
                "original_prediction": before["prediction"],
                "split_prediction": after["prediction"],
                "original_correct": before["correct"],
                "split_correct": after["correct"],
                "original_completion": before["completion"],
                "split_completion": after["completion"],
            }
        )
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "paired_greedy_generation_original_vs_trained_split_adapter"
            if args.adapter is not None
            else "paired_greedy_generation_original_vs_curvature_split"
        ),
        "benchmark": args.benchmark,
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "refinement": (
            str(args.refinement.resolve()) if args.adapter is None else None
        ),
        "adapter": str(args.adapter.resolve()) if args.adapter is not None else None,
        "adapter_training_metadata": adapter_metadata,
        "baseline_result": (
            str(args.baseline_result.resolve()) if args.baseline_result else None
        ),
        "original_reused": args.baseline_result is not None,
        "layer": layer_index,
        "device": device,
        "dtype": str(dtype),
        "examples": len(examples),
        "top_k": len(split_record.original_indices),
        "epsilon": (
            args.epsilon
            if args.adapter is None
            else adapter_metadata.get("initial_epsilon")
        ),
        "selected_neurons": list(split_record.original_indices),
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "prompt": SYSTEM_PROMPT,
        "parameters_before": parameter_count_before,
        "parameters_after": parameter_count_after,
        "added_parameters": parameter_count_after - parameter_count_before,
        "original_correct": original_correct,
        "original_accuracy": original_correct / len(examples),
        "split_correct": split_correct,
        "split_accuracy": split_correct / len(examples),
        "accuracy_delta": (split_correct - original_correct) / len(examples),
        "original_elapsed_seconds": original_seconds,
        "original_elapsed_this_run_seconds": original_elapsed_this_run,
        "split_elapsed_seconds": split_seconds,
        "elapsed_seconds": time.perf_counter() - total_started,
        **paired,
    }
    result = {
        "metadata": metadata,
        "paired_records": paired_records,
        "original_records": original_records,
        "split_records": split_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    log(
        f"complete: original={original_correct}/{len(examples)} "
        f"split={split_correct}/{len(examples)} split_only={paired['split_only_correct']} "
        f"original_only={paired['original_only_correct']}"
    )
    log(f"saved {args.output}")


if __name__ == "__main__":
    main()
