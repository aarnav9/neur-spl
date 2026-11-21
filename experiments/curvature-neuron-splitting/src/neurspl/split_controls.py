#!/usr/bin/env python3
"""Compare a locked curvature split against matched random split controls."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurspl.curvature_proxy import log
from neurspl.evaluate import ROOT, select_device, select_dtype
from neurspl.loss_eval import load_rows, tokenize_example
from neurspl.split import split_swiglu_layer
from neurspl.split_line_search import evaluate_loss, set_separation


DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "micro_v1" / "validation_32.jsonl"
DEFAULT_REFINEMENT = ROOT / "runs" / "curvature_refine_first_refine.json"


def random_unit_directions(count: int, dimension: int, seed: int) -> torch.Tensor:
    """Return reproducible CPU random directions with unit row norm."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    directions = torch.randn(count, dimension, generator=generator)
    return directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)


def choose_random_indices(
    width: int, count: int, excluded: set[int], seed: int
) -> tuple[int, ...]:
    """Choose sorted unique neuron indices, excluding curvature-selected neurons."""
    eligible = [index for index in range(width) if index not in excluded]
    if count > len(eligible):
        raise ValueError("Not enough eligible neurons for the random control")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(eligible), generator=generator)[:count].tolist()
    return tuple(sorted(eligible[position] for position in order))


def summarize_family(
    arms: list[dict], curvature_mean_nll: float
) -> dict[str, float | int]:
    deltas = [float(arm["delta_from_original"]) for arm in arms]
    nlls = [float(arm["mean_nll"]) for arm in arms]
    beats = sum(value <= curvature_mean_nll for value in nlls)
    return {
        "count": len(arms),
        "mean_nll": statistics.mean(nlls),
        "mean_delta_from_original": statistics.mean(deltas),
        "std_delta_from_original": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "best_delta_from_original": min(deltas),
        "worst_delta_from_original": max(deltas),
        "improved_over_original": sum(delta < 0.0 for delta in deltas),
        "beat_or_tied_curvature": beats,
        "plus_one_tail_fraction": (beats + 1) / (len(arms) + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--refinement", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.65)
    parser.add_argument("--control-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()
    if args.epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if args.control_seeds < 1:
        raise ValueError("control-seeds must be positive")

    started = time.perf_counter()
    refinement = json.loads(args.refinement.read_text())
    candidates = refinement["refined_candidates"][: args.top_k]
    if len(candidates) != args.top_k:
        raise ValueError("The refinement file does not contain enough candidates")
    layer_index = int(refinement["metadata"]["layer"])
    selected_indices = tuple(sorted(int(item["neuron_index"]) for item in candidates))
    if len(set(selected_indices)) != args.top_k:
        raise ValueError("The selected curvature candidates contain duplicate neurons")

    direction_by_neuron = {
        int(item["neuron_index"]): item["direction_gate"] + item["direction_up"]
        for item in candidates
    }
    curvature_directions_cpu = torch.tensor(
        [direction_by_neuron[index] for index in selected_indices], dtype=torch.float32
    )
    curvature_directions_cpu /= curvature_directions_cpu.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12)

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    rows = load_rows(args.data)
    arm_count = 2 + 3 * args.control_seeds
    log(
        f"starting locked split controls: epsilon={args.epsilon:g} top_k={args.top_k} "
        f"control_seeds={args.control_seeds} arms={arm_count} examples={len(rows)} "
        f"device={device} dtype={dtype}"
    )
    log(f"refinement input: {args.refinement}")
    log(f"selected neurons: {list(selected_indices)}")
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
    mlp = model.model.layers[layer_index].mlp
    base_gate = mlp.gate_proj
    base_up = mlp.up_proj
    base_down = mlp.down_proj
    base_intermediate_size = getattr(mlp, "intermediate_size", None)
    width = base_gate.out_features
    hidden_size = base_gate.in_features
    direction_dimension = hidden_size * 2
    if curvature_directions_cpu.shape != (args.top_k, direction_dimension):
        raise ValueError("Refined direction dimensions do not match the selected model")
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())

    original_nll, target_tokens, original_seconds = evaluate_loss(
        model, tokenizer, encoded, args.batch_size, device, "original"
    )
    log(f"original model complete: mean_nll={original_nll:.8f}")

    arms: list[dict] = []

    def restore_base_layer() -> None:
        mlp.gate_proj = base_gate
        mlp.up_proj = base_up
        mlp.down_proj = base_down
        if base_intermediate_size is not None:
            mlp.intermediate_size = base_intermediate_size

    def run_arm(
        name: str,
        family: str,
        indices: tuple[int, ...],
        directions_cpu: torch.Tensor,
        epsilon: float,
        control_index: int | None = None,
        random_index_seed: int | None = None,
        random_direction_seed: int | None = None,
    ) -> dict:
        restore_base_layer()
        record = split_swiglu_layer(model, layer_index, list(indices))
        expanded_mlp = model.model.layers[layer_index].mlp
        originals = torch.tensor(record.original_indices, device=device)
        gate_centers = expanded_mlp.gate_proj.weight[originals].detach().clone()
        up_centers = expanded_mlp.up_proj.weight[originals].detach().clone()
        directions = directions_cpu.to(
            device=device, dtype=expanded_mlp.gate_proj.weight.dtype
        )
        set_separation(
            expanded_mlp,
            record.original_indices,
            record.duplicate_indices,
            gate_centers,
            up_centers,
            directions[:, :hidden_size],
            directions[:, hidden_size:],
            epsilon,
        )
        model.requires_grad_(False)
        mean_nll, measured_tokens, elapsed = evaluate_loss(
            model, tokenizer, encoded, args.batch_size, device, name
        )
        if measured_tokens != target_tokens:
            raise RuntimeError("Target-token count changed between control arms")
        delta = mean_nll - original_nll
        arm = {
            "name": name,
            "family": family,
            "control_index": control_index,
            "random_index_seed": random_index_seed,
            "random_direction_seed": random_direction_seed,
            "neuron_indices": list(record.original_indices),
            "epsilon": epsilon,
            "mean_nll": mean_nll,
            "delta_from_original": delta,
            "perplexity_ratio": math.exp(delta),
            "added_parameters": record.added_parameters,
            "elapsed_seconds": elapsed,
        }
        arms.append(arm)
        log(
            f"arm complete: name={name} mean_nll={mean_nll:.8f} "
            f"delta_original={delta:+.8e} perplexity_ratio={math.exp(delta):.8f}"
        )
        restore_base_layer()
        del expanded_mlp, originals, gate_centers, up_centers, directions
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        return arm

    exact_arm = run_arm(
        "exact_split", "exact_split", selected_indices, curvature_directions_cpu, 0.0
    )
    curvature_arm = run_arm(
        "curvature", "curvature", selected_indices, curvature_directions_cpu, args.epsilon
    )

    excluded = set(selected_indices)
    for control_index in range(args.control_seeds):
        index_seed = args.seed + control_index
        same_direction_seed = args.seed + 100_000 + control_index
        random_direction_seed = args.seed + 200_000 + control_index
        random_indices = choose_random_indices(
            width, args.top_k, excluded=excluded, seed=index_seed
        )
        same_neuron_random = random_unit_directions(
            args.top_k, direction_dimension, same_direction_seed
        )
        random_neuron_random = random_unit_directions(
            args.top_k, direction_dimension, random_direction_seed
        )
        run_arm(
            f"same_neurons_random_direction_{control_index:02d}",
            "same_neurons_random_direction",
            selected_indices,
            same_neuron_random,
            args.epsilon,
            control_index,
            None,
            same_direction_seed,
        )
        run_arm(
            f"random_neurons_curvature_directions_{control_index:02d}",
            "random_neurons_curvature_directions",
            random_indices,
            curvature_directions_cpu,
            args.epsilon,
            control_index,
            index_seed,
            None,
        )
        run_arm(
            f"random_neurons_random_directions_{control_index:02d}",
            "random_neurons_random_directions",
            random_indices,
            random_neuron_random,
            args.epsilon,
            control_index,
            index_seed,
            random_direction_seed,
        )

    control_families = sorted(
        family for family in {arm["family"] for arm in arms} if family not in {"exact_split", "curvature"}
    )
    summaries = {
        family: summarize_family(
            [arm for arm in arms if arm["family"] == family],
            curvature_arm["mean_nll"],
        )
        for family in control_families
    }
    restore_base_layer()
    parameter_count_after_restore = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "metadata": {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "locked_curvature_split_matched_random_controls",
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "refinement": str(args.refinement.resolve()),
            "layer": layer_index,
            "device": device,
            "dtype": str(dtype),
            "examples": len(rows),
            "target_tokens": target_tokens,
            "top_k": args.top_k,
            "epsilon": args.epsilon,
            "control_seeds": args.control_seeds,
            "master_seed": args.seed,
            "selected_neurons": list(selected_indices),
            "parameters_before": parameter_count_before,
            "parameters_after_restore": parameter_count_after_restore,
            "original_mean_nll": original_nll,
            "original_elapsed_seconds": original_seconds,
            "exact_split_numerical_delta": exact_arm["delta_from_original"],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "curvature_arm": curvature_arm,
        "control_summaries": summaries,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for family, summary in summaries.items():
        log(
            f"control summary: family={family} mean_delta="
            f"{summary['mean_delta_from_original']:+.8e} "
            f"best_delta={summary['best_delta_from_original']:+.8e} "
            f"beat_curvature={summary['beat_or_tied_curvature']}/{summary['count']}"
        )
    log(
        f"complete: curvature_delta={curvature_arm['delta_from_original']:+.8e} "
        f"exact_split_delta={exact_arm['delta_from_original']:+.3e}"
    )
    log(f"saved {args.output}")
    del model, tokenizer, encoded
    gc.collect()


if __name__ == "__main__":
    main()
