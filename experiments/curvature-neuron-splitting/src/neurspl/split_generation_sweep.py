#!/usr/bin/env python3
"""Sweep split epsilon on validation using paired greedy-generation behavior."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurspl.curvature_proxy import log
from neurspl.evaluate import ROOT, SYSTEM_PROMPT, load_jsonl, select_device, select_dtype
from neurspl.evaluate_split_generation import (
    generate_records,
    load_reusable_baseline,
    paired_summary,
)
from neurspl.split import split_swiglu_layer
from neurspl.split_line_search import set_separation


DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "micro_v1" / "validation_32.jsonl"
DEFAULT_REFINEMENT = ROOT / "runs" / "curvature_refine_first_refine.json"


def parse_epsilons(text: str) -> list[float]:
    values = sorted(set(float(value.strip()) for value in text.split(",")))
    if not values or values[0] < 0:
        raise ValueError("epsilons must contain nonnegative values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--refinement", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument(
        "--baseline-result",
        type=Path,
        help="Reuse a compatible full-data baseline instead of generating it again",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("gsm8k", "math"), default="gsm8k")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epsilons", default="0,0.05,0.1,0.2,0.4,0.65")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--offset", type=int, default=0, help="First example in the data file to use"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Number of examples to use; 0 means all"
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 0:
        raise ValueError("offset and limit must be nonnegative")

    started = time.perf_counter()
    epsilons = parse_epsilons(args.epsilons)
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
    all_examples = load_jsonl(args.data, args.benchmark)
    stop = None if args.limit == 0 else args.offset + args.limit
    examples = all_examples[args.offset : stop]
    if not examples:
        raise ValueError("offset/limit selected no examples")
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    log(
        f"starting validation generation sweep: examples={len(examples)} "
        f"epsilons={epsilons} top_k={args.top_k} device={device} dtype={dtype}"
    )
    log(f"refinement input: {args.refinement}")
    baseline_metadata = None
    original_records = None
    if args.baseline_result is not None:
        all_original_records, baseline_metadata = load_reusable_baseline(
            args.baseline_result,
            model_path=args.model,
            data_path=args.data,
            benchmark=args.benchmark,
            examples=all_examples,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        original_records = all_original_records[args.offset : stop]
        log(
            f"reusing compatible baseline slice: {args.baseline_result} "
            f"offset={args.offset} examples={len(original_records)}"
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
    mlp = model.model.layers[layer_index].mlp
    base_gate = mlp.gate_proj
    base_up = mlp.up_proj
    base_down = mlp.down_proj
    base_intermediate_size = getattr(mlp, "intermediate_size", None)
    parameters_before = sum(parameter.numel() for parameter in model.parameters())

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
        f"original complete: correct={original_correct}/{len(examples)} "
        f"accuracy={original_correct / len(examples):.4f}"
    )

    def restore_base_layer() -> None:
        mlp.gate_proj = base_gate
        mlp.up_proj = base_up
        mlp.down_proj = base_down
        if base_intermediate_size is not None:
            mlp.intermediate_size = base_intermediate_size

    variants = []
    for epsilon in epsilons:
        restore_base_layer()
        split_record = split_swiglu_layer(model, layer_index, indices)
        expanded = model.model.layers[layer_index].mlp
        originals = torch.tensor(split_record.original_indices, device=device)
        gate_centers = expanded.gate_proj.weight[originals].detach().clone()
        up_centers = expanded.up_proj.weight[originals].detach().clone()
        gate_directions = torch.tensor(
            [direction_by_neuron[index][0] for index in split_record.original_indices],
            device=device,
            dtype=expanded.gate_proj.weight.dtype,
        )
        up_directions = torch.tensor(
            [direction_by_neuron[index][1] for index in split_record.original_indices],
            device=device,
            dtype=expanded.up_proj.weight.dtype,
        )
        set_separation(
            expanded,
            split_record.original_indices,
            split_record.duplicate_indices,
            gate_centers,
            up_centers,
            gate_directions,
            up_directions,
            epsilon,
        )
        model.requires_grad_(False)
        records, elapsed = generate_records(
            model,
            tokenizer,
            examples,
            device,
            f"epsilon={epsilon:g}",
            args.batch_size,
            args.max_input_tokens,
            args.max_new_tokens,
        )
        correct = sum(record["correct"] for record in records)
        paired = paired_summary(original_records, records)
        variants.append(
            {
                "epsilon": epsilon,
                "correct": correct,
                "accuracy": correct / len(examples),
                "accuracy_delta": (correct - original_correct) / len(examples),
                "elapsed_seconds": elapsed,
                **paired,
                "records": records,
            }
        )
        log(
            f"epsilon={epsilon:g} complete: correct={correct}/{len(examples)} "
            f"changed_predictions={paired['changed_prediction']}/{len(examples)} "
            f"split_only={paired['split_only_correct']} "
            f"original_only={paired['original_only_correct']}"
        )
        restore_base_layer()
        del expanded, originals, gate_centers, up_centers
        del gate_directions, up_directions
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    restore_base_layer()
    result = {
        "metadata": {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "validation_curvature_split_generation_epsilon_sweep",
            "benchmark": args.benchmark,
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "refinement": str(args.refinement.resolve()),
            "baseline_result": (
                str(args.baseline_result.resolve()) if args.baseline_result else None
            ),
            "original_reused": args.baseline_result is not None,
            "layer": layer_index,
            "device": device,
            "dtype": str(dtype),
            "examples": len(examples),
            "data_examples": len(all_examples),
            "offset": args.offset,
            "limit": args.limit,
            "top_k": args.top_k,
            "epsilons": epsilons,
            "selected_neurons": list(sorted(indices)),
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
            "prompt": SYSTEM_PROMPT,
            "parameters_before": parameters_before,
            "original_correct": original_correct,
            "original_accuracy": original_correct / len(examples),
            "original_elapsed_seconds": original_seconds,
            "original_elapsed_this_run_seconds": original_elapsed_this_run,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "original_records": original_records,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    log(f"complete: saved {args.output}")


if __name__ == "__main__":
    main()
