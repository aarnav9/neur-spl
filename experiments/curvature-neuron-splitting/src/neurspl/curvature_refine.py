#!/usr/bin/env python3
"""Refine screened SwiGLU neurons with batched Lanczos eigensolvers."""

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

from neurspl.curvature_proxy import log, silu_derivatives
from neurspl.evaluate import ROOT, select_device, select_dtype
from neurspl.loss_eval import collate, load_rows, tokenize_example


DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "micro_v1" / "curvature_16.jsonl"
DEFAULT_SCREENING = ROOT / "runs" / "curvature_proxy_first_scan.json"


def random_unit_vectors(count: int, dimension: int, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vectors = torch.randn(count, dimension, generator=generator)
    vectors /= vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return vectors.to(device)


def build_tridiagonal(
    diagonal: torch.Tensor, off_diagonal: torch.Tensor, candidate: int
) -> torch.Tensor:
    size = diagonal.shape[0]
    matrix = torch.diag(diagonal[:, candidate])
    if size > 1:
        values = off_diagonal[: size - 1, candidate]
        positions = torch.arange(size - 1)
        matrix[positions, positions + 1] = values
        matrix[positions + 1, positions] = values
    return matrix


def current_minimum_ritz(
    diagonals: list[torch.Tensor], off_diagonals: list[torch.Tensor]
) -> float:
    diagonal = torch.stack([value.detach().cpu() for value in diagonals])
    if len(diagonals) == 1:
        return float(diagonal.min().item())
    off = torch.stack([value.detach().cpu() for value in off_diagonals])
    minima = [
        torch.linalg.eigvalsh(build_tridiagonal(diagonal, off, candidate))[0]
        for candidate in range(diagonal.shape[1])
    ]
    return float(torch.stack(minima).min().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--screening", type=Path, default=DEFAULT_SCREENING)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    screening = json.loads(args.screening.read_text())
    screened = screening["top_candidates"][: args.candidate_count]
    indices = [int(candidate["neuron_index"]) for candidate in screened]
    if len(indices) < 2:
        raise ValueError("The screening result must contain at least two candidates")
    if len(set(indices)) != len(indices):
        raise ValueError("The screening candidates contain duplicate neuron indices")

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    rows = load_rows(args.data)
    log(
        f"starting Lanczos refinement: candidates={len(indices)} examples={len(rows)} "
        f"iterations={args.iterations} device={device} dtype={dtype}"
    )
    log(f"screening input: {args.screening}")
    log(f"candidate neuron indices: {indices}")
    log(f"loading tokenizer and model from {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=False
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    layer_index = int(screening["metadata"]["layer"])
    mlp = model.model.layers[layer_index].mlp
    hidden_size = mlp.gate_proj.in_features
    dimension = hidden_size * 2
    selected = torch.tensor(indices, dtype=torch.long, device=device)
    encoded = [tokenize_example(tokenizer, row, args.max_length) for row in rows]

    captured: dict[str, torch.Tensor] = {}

    def capture_mlp_input(_module, inputs):
        captured["mlp_input"] = inputs[0].detach()

    def capture_down_input(_module, inputs):
        activation = inputs[0].detach().requires_grad_(True)
        captured["activation"] = activation
        return (activation,)

    input_hook = mlp.register_forward_pre_hook(capture_mlp_input)
    activation_hook = mlp.down_proj.register_forward_pre_hook(capture_down_input)
    inputs_parts: list[torch.Tensor] = []
    gate_parts: list[torch.Tensor] = []
    cross_parts: list[torch.Tensor] = []
    total_target_tokens = 0
    batches = (len(rows) + args.batch_size - 1) // args.batch_size

    try:
        for start in range(0, len(rows), args.batch_size):
            batch_number = start // args.batch_size + 1
            batch_started = time.perf_counter()
            batch = collate(tokenizer, encoded[start : start + args.batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            captured.clear()
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
            shift_labels = batch["labels"][:, 1:]
            target_tokens = int(shift_labels.ne(-100).sum().item())
            loss_sum = functional.cross_entropy(
                logits[:, :-1, :].float().transpose(1, 2),
                shift_labels,
                reduction="sum",
                ignore_index=-100,
            )
            mlp_input = captured["mlp_input"].float()
            activation_gradient = torch.autograd.grad(
                loss_sum, captured["activation"], retain_graph=False
            )[0]
            valid = batch["attention_mask"].bool()
            with torch.no_grad():
                gate_weight = mlp.gate_proj.weight.index_select(0, selected).float()
                up_weight = mlp.up_proj.weight.index_select(0, selected).float()
                gate_bias = (
                    mlp.gate_proj.bias.index_select(0, selected).float()
                    if mlp.gate_proj.bias is not None
                    else None
                )
                up_bias = (
                    mlp.up_proj.bias.index_select(0, selected).float()
                    if mlp.up_proj.bias is not None
                    else None
                )
                gate_values = functional.linear(mlp_input, gate_weight, gate_bias)
                up_values = functional.linear(mlp_input, up_weight, up_bias)
                silu_first, silu_second = silu_derivatives(gate_values)
                selected_gradient = activation_gradient.index_select(2, selected).float()
                inputs_parts.append(mlp_input[valid].detach())
                gate_parts.append(
                    (selected_gradient * silu_second * up_values)[valid].detach()
                )
                cross_parts.append((selected_gradient * silu_first)[valid].detach())
            total_target_tokens += target_tokens
            log(
                f"collection batch {batch_number}/{batches} complete; examples="
                f"{min(start + args.batch_size, len(rows))}/{len(rows)} "
                f"valid_positions={sum(part.shape[0] for part in inputs_parts)} "
                f"target_tokens={total_target_tokens} "
                f"batch_seconds={time.perf_counter() - batch_started:.1f}"
            )
            del (
                logits,
                loss_sum,
                shift_labels,
                batch,
                mlp_input,
                activation_gradient,
                gate_values,
                up_values,
                silu_first,
                silu_second,
                selected_gradient,
                valid,
            )
            captured.clear()
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
    finally:
        input_hook.remove()
        activation_hook.remove()

    input_matrix = torch.cat(inputs_parts, dim=0)
    gate_coefficient = torch.cat(gate_parts, dim=0)
    cross_coefficient = torch.cat(cross_parts, dim=0)
    del (
        inputs_parts,
        gate_parts,
        cross_parts,
        encoded,
        model,
        tokenizer,
        mlp,
        captured,
        gate_weight,
        up_weight,
        gate_bias,
        up_bias,
    )
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    log(
        f"statistics ready: positions={input_matrix.shape[0]} hidden={hidden_size} "
        f"candidates={len(indices)}; starting batched Lanczos"
    )

    def hessian_vector_product(vectors: torch.Tensor) -> torch.Tensor:
        gate_direction = vectors[:, :hidden_size]
        up_direction = vectors[:, hidden_size:]
        input_gate = input_matrix @ gate_direction.transpose(0, 1)
        input_up = input_matrix @ up_direction.transpose(0, 1)
        output_gate = input_matrix.transpose(0, 1) @ (
            gate_coefficient * input_gate + cross_coefficient * input_up
        )
        output_up = input_matrix.transpose(0, 1) @ (cross_coefficient * input_gate)
        return torch.cat((output_gate.transpose(0, 1), output_up.transpose(0, 1)), dim=1) / total_target_tokens

    symmetry_left = random_unit_vectors(len(indices), dimension, args.seed + 1, device)
    symmetry_right = random_unit_vectors(len(indices), dimension, args.seed + 2, device)
    with torch.no_grad():
        left_product = (symmetry_left * hessian_vector_product(symmetry_right)).sum(dim=1)
        right_product = (symmetry_right * hessian_vector_product(symmetry_left)).sum(dim=1)
        symmetry_error = (left_product - right_product).abs()
    log(
        f"HVP symmetry check: max_abs_error={float(symmetry_error.max().item()):.3e} "
        f"mean_abs_error={float(symmetry_error.mean().item()):.3e}"
    )

    basis: list[torch.Tensor] = []
    diagonals: list[torch.Tensor] = []
    off_diagonals: list[torch.Tensor] = []
    previous = torch.zeros(len(indices), dimension, device=device)
    previous_beta = torch.zeros(len(indices), device=device)
    current = random_unit_vectors(len(indices), dimension, args.seed, device)
    final_beta = None

    with torch.no_grad():
        for iteration in range(args.iterations):
            iteration_started = time.perf_counter()
            product = hessian_vector_product(current)
            if iteration:
                product -= previous_beta[:, None] * previous
            diagonal = (current * product).sum(dim=1)
            residual = product - diagonal[:, None] * current
            for old_basis in basis:
                residual -= (old_basis * residual).sum(dim=1)[:, None] * old_basis
            beta = residual.norm(dim=1)
            basis.append(current)
            diagonals.append(diagonal)
            final_beta = beta
            if iteration < args.iterations - 1:
                off_diagonals.append(beta)
                previous = current
                previous_beta = beta
                current = residual / beta.clamp_min(1e-12)[:, None]
            estimate = current_minimum_ritz(diagonals, off_diagonals)
            log(
                f"Lanczos iteration {iteration + 1}/{args.iterations} complete; "
                f"current_global_min={estimate:.6e} "
                f"iteration_seconds={time.perf_counter() - iteration_started:.2f}"
            )

    diagonal_cpu = torch.stack([value.cpu() for value in diagonals])
    off_cpu = torch.stack([value.cpu() for value in off_diagonals])
    basis_cpu = torch.stack([value.cpu() for value in basis])
    final_beta_cpu = final_beta.cpu() if final_beta is not None else torch.zeros(len(indices))
    refined = []
    for candidate, neuron_index in enumerate(indices):
        tridiagonal = build_tridiagonal(diagonal_cpu, off_cpu, candidate)
        eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
        coefficients = eigenvectors[:, 0]
        direction = coefficients @ basis_cpu[:, candidate, :]
        direction /= direction.norm().clamp_min(1e-12)
        largest = int(direction.abs().argmax().item())
        if direction[largest] < 0:
            direction = -direction
        residual_estimate = float(
            (final_beta_cpu[candidate] * coefficients[-1].abs()).item()
        )
        refined.append(
            {
                "neuron_index": neuron_index,
                "screening_rank": int(screened[candidate]["rank"]),
                "screening_curvature": float(screened[candidate]["curvature"]),
                "minimum_ritz_value": float(eigenvalues[0].item()),
                "residual_estimate": residual_estimate,
                "direction_gate": direction[:hidden_size].tolist(),
                "direction_up": direction[hidden_size:].tolist(),
            }
        )
    refined.sort(key=lambda item: item["minimum_ritz_value"])
    for rank, candidate in enumerate(refined, start=1):
        candidate["refined_rank"] = rank

    result = {
        "metadata": {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "batched_lanczos_swiglu_splitting_matrix",
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "screening": str(args.screening.resolve()),
            "layer": layer_index,
            "device": device,
            "dtype": str(dtype),
            "examples": len(rows),
            "target_tokens": total_target_tokens,
            "positions": int(input_matrix.shape[0]),
            "candidate_count": len(indices),
            "iterations": args.iterations,
            "lanczos_seed": args.seed,
            "hvp_symmetry_max_abs_error": float(symmetry_error.max().item()),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "refined_candidates": refined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    best = refined[0]
    log(
        f"complete: best_neuron={best['neuron_index']} "
        f"minimum_ritz={best['minimum_ritz_value']:.6e} "
        f"residual={best['residual_estimate']:.3e}"
    )
    log(f"saved {args.output}")


if __name__ == "__main__":
    main()
