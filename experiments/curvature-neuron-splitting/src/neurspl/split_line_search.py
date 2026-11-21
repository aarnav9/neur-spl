#!/usr/bin/env python3
"""Test whether refined split directions lower loss for small epsilon values."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurspl.curvature_proxy import log
from neurspl.evaluate import ROOT, select_device, select_dtype, synchronize
from neurspl.loss_eval import collate, load_rows, tokenize_example
from neurspl.split import split_swiglu_layer


DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "micro_v1" / "curvature_16.jsonl"
DEFAULT_REFINEMENT = ROOT / "runs" / "curvature_refine_first_refine.json"


def evaluate_loss(
    model,
    tokenizer,
    encoded: list[tuple[list[int], list[int]]],
    batch_size: int,
    device: str,
    label: str,
) -> tuple[float, int, float]:
    total_nll = 0.0
    total_tokens = 0
    batches = (len(encoded) + batch_size - 1) // batch_size
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch_number = start // batch_size + 1
            batch_started = time.perf_counter()
            batch = collate(tokenizer, encoded[start : start + batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            synchronize(device)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
            shift_labels = batch["labels"][:, 1:]
            loss_sum = functional.cross_entropy(
                logits[:, :-1, :].float().transpose(1, 2),
                shift_labels,
                reduction="sum",
                ignore_index=-100,
            )
            tokens = int(shift_labels.ne(-100).sum().item())
            total_nll += float(loss_sum.item())
            total_tokens += tokens
            synchronize(device)
            log(
                f"{label} batch {batch_number}/{batches} complete; "
                f"examples={min(start + batch_size, len(encoded))}/{len(encoded)} "
                f"running_nll={total_nll / total_tokens:.8f} "
                f"batch_seconds={time.perf_counter() - batch_started:.1f}"
            )
            del batch, logits, shift_labels, loss_sum
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
    return total_nll / total_tokens, total_tokens, time.perf_counter() - started


def set_separation(
    mlp,
    original_indices: tuple[int, ...],
    duplicate_indices: tuple[int, ...],
    gate_centers: torch.Tensor,
    up_centers: torch.Tensor,
    gate_directions: torch.Tensor,
    up_directions: torch.Tensor,
    epsilon: float,
) -> None:
    original = torch.tensor(original_indices, device=mlp.gate_proj.weight.device)
    duplicate = torch.tensor(duplicate_indices, device=mlp.gate_proj.weight.device)
    with torch.no_grad():
        mlp.gate_proj.weight[original] = gate_centers + epsilon * gate_directions
        mlp.gate_proj.weight[duplicate] = gate_centers - epsilon * gate_directions
        mlp.up_proj.weight[original] = up_centers + epsilon * up_directions
        mlp.up_proj.weight[duplicate] = up_centers - epsilon * up_directions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--refinement", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epsilons", default="0,0.025,0.05,0.1,0.2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    refinement = json.loads(args.refinement.read_text())
    candidates = refinement["refined_candidates"][: args.top_k]
    if len(candidates) != args.top_k:
        raise ValueError("The refinement file does not contain enough candidates")
    layer_index = int(refinement["metadata"]["layer"])
    epsilons = [float(value.strip()) for value in args.epsilons.split(",")]
    if 0.0 not in epsilons:
        epsilons.insert(0, 0.0)
    epsilons = sorted(set(epsilons))
    if epsilons[0] < 0:
        raise ValueError("Epsilon values must be nonnegative")

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    rows = load_rows(args.data)
    log(
        f"starting split line search: top_k={args.top_k} layer={layer_index} "
        f"epsilons={epsilons} examples={len(rows)} device={device} dtype={dtype}"
    )
    log(f"refinement input: {args.refinement}")
    log(f"selected neurons: {[candidate['neuron_index'] for candidate in candidates]}")
    log(f"loading tokenizer and model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=False
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    encoded = [tokenize_example(tokenizer, row, args.max_length) for row in rows]

    original_nll, target_tokens, original_seconds = evaluate_loss(
        model, tokenizer, encoded, args.batch_size, device, "original"
    )
    log(f"original model complete: mean_nll={original_nll:.8f}")

    direction_by_neuron = {
        int(candidate["neuron_index"]): (
            candidate["direction_gate"],
            candidate["direction_up"],
        )
        for candidate in candidates
    }
    indices = [int(candidate["neuron_index"]) for candidate in candidates]
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())
    split_record = split_swiglu_layer(model, layer_index, indices)
    parameter_count_after = sum(parameter.numel() for parameter in model.parameters())
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

    curvature_sum = sum(float(candidate["minimum_ritz_value"]) for candidate in candidates)
    measurements = []
    exact_split_nll = None
    for epsilon in epsilons:
        set_separation(
            mlp,
            split_record.original_indices,
            split_record.duplicate_indices,
            gate_centers,
            up_centers,
            gate_directions,
            up_directions,
            epsilon,
        )
        label = f"epsilon={epsilon:g}"
        mean_nll, measured_tokens, elapsed = evaluate_loss(
            model, tokenizer, encoded, args.batch_size, device, label
        )
        if measured_tokens != target_tokens:
            raise RuntimeError("Target-token count changed during line search")
        if epsilon == 0.0:
            exact_split_nll = mean_nll
        predicted_delta = 0.5 * epsilon * epsilon * curvature_sum
        measurement = {
            "epsilon": epsilon,
            "mean_nll": mean_nll,
            "delta_from_original": mean_nll - original_nll,
            "delta_from_exact_split": (
                mean_nll - exact_split_nll if exact_split_nll is not None else None
            ),
            "quadratic_predicted_delta": predicted_delta,
            "elapsed_seconds": elapsed,
        }
        measurements.append(measurement)
        log(
            f"{label} complete: mean_nll={mean_nll:.8f} "
            f"delta_original={measurement['delta_from_original']:+.8e} "
            f"predicted={predicted_delta:+.8e}"
        )

    if exact_split_nll is None:
        raise RuntimeError("The exact split at epsilon zero was not evaluated")
    result = {
        "metadata": {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "refined_curvature_split_direction_line_search",
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "refinement": str(args.refinement.resolve()),
            "layer": layer_index,
            "device": device,
            "dtype": str(dtype),
            "examples": len(rows),
            "target_tokens": target_tokens,
            "top_k": args.top_k,
            "selected_neurons": indices,
            "curvature_sum": curvature_sum,
            "parameters_before": parameter_count_before,
            "parameters_after": parameter_count_after,
            "added_parameters": parameter_count_after - parameter_count_before,
            "original_mean_nll": original_nll,
            "exact_split_mean_nll": exact_split_nll,
            "exact_split_numerical_delta": exact_split_nll - original_nll,
            "original_elapsed_seconds": original_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "selected_candidates": [
            {
                key: candidate[key]
                for key in (
                    "refined_rank",
                    "neuron_index",
                    "minimum_ritz_value",
                    "residual_estimate",
                )
            }
            for candidate in candidates
        ],
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    best = min(measurements, key=lambda item: item["mean_nll"])
    log(
        f"complete: best_epsilon={best['epsilon']:g} best_nll={best['mean_nll']:.8f} "
        f"best_delta={best['delta_from_original']:+.8e}"
    )
    log(f"saved {args.output}")


if __name__ == "__main__":
    main()
