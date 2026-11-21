#!/usr/bin/env python3
"""Evaluate math-derived split directions on general multiple-choice loss."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurspl.curvature_proxy import log, silu_derivatives
from neurspl.evaluate import ROOT, select_device, select_dtype
from neurspl.evaluate_general_retention import SYSTEM_PROMPT, format_question, load_rows
from neurspl.loss_eval import collate


DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "general_retention_v1" / "eval_192.jsonl"
DEFAULT_REFINEMENT = ROOT / "runs" / "curvature_refine_qwen_layer12_refine.json"


def tokenize_general_example(
    tokenizer, row: dict, max_length: int
) -> tuple[list[int], list[int]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_question(row)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[int(row["answer_index"])]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    common = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        common += 1
    if len(full_ids) > max_length:
        removed = len(full_ids) - max_length
        full_ids = full_ids[removed:]
        common = max(0, common - removed)
    labels = [-100] * common + full_ids[common:]
    if not any(value != -100 for value in labels):
        raise ValueError(f"No answer tokens remained for {row['example_id']}")
    return full_ids, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--refinement", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")

    started = time.perf_counter()
    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    refinement = json.loads(args.refinement.read_text())
    candidates = refinement["refined_candidates"][: args.top_k]
    if len(candidates) != args.top_k:
        raise ValueError("The refinement file does not contain enough candidates")
    layer_index = int(refinement["metadata"]["layer"])
    indices = [int(item["neuron_index"]) for item in candidates]
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    tasks = sorted({row["benchmark"] for row in rows})
    log(
        f"starting cross-domain curvature: examples={len(rows)} tasks={tasks} "
        f"directions={len(indices)} layer={layer_index} device={device} dtype={dtype}"
    )
    log(f"math refinement input: {args.refinement}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = [tokenize_general_example(tokenizer, row, args.max_length) for row in rows]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    mlp = model.model.layers[layer_index].mlp
    selected = torch.tensor(indices, dtype=torch.long, device=device)
    gate_directions = torch.tensor(
        [item["direction_gate"] for item in candidates],
        device=device,
        dtype=torch.float32,
    )
    up_directions = torch.tensor(
        [item["direction_up"] for item in candidates],
        device=device,
        dtype=torch.float32,
    )

    captured: dict[str, torch.Tensor] = {}

    def capture_mlp_input(_module, inputs):
        captured["mlp_input"] = inputs[0].detach()

    def capture_down_input(_module, inputs):
        activation = inputs[0].detach().requires_grad_(True)
        captured["activation"] = activation
        return (activation,)

    input_hook = mlp.register_forward_pre_hook(capture_mlp_input)
    activation_hook = mlp.down_proj.register_forward_pre_hook(capture_down_input)
    curvature_by_task = {
        task: torch.zeros(args.top_k, device=device, dtype=torch.float32)
        for task in tasks
    }
    tokens_by_task: dict[str, int] = defaultdict(int)
    positions_by_task: dict[str, int] = defaultdict(int)
    batch_count = (len(rows) + args.batch_size - 1) // args.batch_size

    try:
        for start in range(0, len(rows), args.batch_size):
            batch_number = start // args.batch_size + 1
            batch_started = time.perf_counter()
            batch_rows = rows[start : start + args.batch_size]
            batch_tasks = {row["benchmark"] for row in batch_rows}
            if len(batch_tasks) != 1:
                raise ValueError(
                    "Each curvature batch must contain one task; use batch size 1"
                )
            task = next(iter(batch_tasks))
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
            activation_gradient = torch.autograd.grad(
                loss_sum, captured["activation"], retain_graph=False
            )[0]
            valid = batch["attention_mask"].bool()
            with torch.no_grad():
                mlp_input = captured["mlp_input"].float()
                gate_weight = mlp.gate_proj.weight.index_select(0, selected).float()
                up_weight = mlp.up_proj.weight.index_select(0, selected).float()
                gate_values = functional.linear(mlp_input, gate_weight)
                up_values = functional.linear(mlp_input, up_weight)
                silu_first, silu_second = silu_derivatives(gate_values)
                selected_gradient = activation_gradient.index_select(2, selected).float()
                gate_coefficient = selected_gradient * silu_second * up_values
                cross_coefficient = selected_gradient * silu_first
                input_gate = torch.einsum(
                    "bth,kh->btk", mlp_input, gate_directions
                )
                input_up = torch.einsum("bth,kh->btk", mlp_input, up_directions)
                directional = (
                    gate_coefficient * input_gate.square()
                    + 2.0 * cross_coefficient * input_gate * input_up
                )
                curvature_by_task[task] += directional[valid].sum(dim=0)
            tokens_by_task[task] += target_tokens
            positions_by_task[task] += int(valid.sum().item())
            running = curvature_by_task[task] / tokens_by_task[task]
            log(
                f"batch {batch_number}/{batch_count} complete; "
                f"examples={min(start + args.batch_size, len(rows))}/{len(rows)} "
                f"task={task} target_tokens={tokens_by_task[task]} "
                f"task_curvature_sum={float(running.sum().item()):+.6e} "
                f"batch_seconds={time.perf_counter() - batch_started:.1f}"
            )
            del (
                batch,
                logits,
                shift_labels,
                loss_sum,
                activation_gradient,
                valid,
                mlp_input,
                gate_weight,
                up_weight,
                gate_values,
                up_values,
                silu_first,
                silu_second,
                selected_gradient,
                gate_coefficient,
                cross_coefficient,
                input_gate,
                input_up,
                directional,
                running,
            )
            captured.clear()
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
    finally:
        input_hook.remove()
        activation_hook.remove()

    normalized = {
        task: (curvature_by_task[task] / tokens_by_task[task]).cpu()
        for task in tasks
    }
    total_tokens = sum(tokens_by_task.values())
    overall = sum(curvature_by_task.values()) / total_tokens
    per_neuron = []
    for candidate_index, candidate in enumerate(candidates):
        per_neuron.append(
            {
                "neuron_index": int(candidate["neuron_index"]),
                "math_curvature": float(candidate["minimum_ritz_value"]),
                "general_curvature": float(overall[candidate_index].item()),
                "curvature_by_task": {
                    task: float(normalized[task][candidate_index].item())
                    for task in tasks
                },
            }
        )
    general_sum = float(overall.sum().item())
    math_sum = sum(float(item["minimum_ritz_value"]) for item in candidates)
    result = {
        "metadata": {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "cross_domain_directional_splitting_curvature",
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "math_refinement": str(args.refinement.resolve()),
            "layer": layer_index,
            "device": device,
            "dtype": str(dtype),
            "examples": len(rows),
            "tasks": tasks,
            "target_tokens": total_tokens,
            "tokens_by_task": dict(tokens_by_task),
            "positions_by_task": dict(positions_by_task),
            "top_k": args.top_k,
            "epsilon": args.epsilon,
            "math_curvature_sum": math_sum,
            "general_curvature_sum": general_sum,
            "general_curvature_sum_by_task": {
                task: float(normalized[task].sum().item()) for task in tasks
            },
            "quadratic_predicted_general_delta": 0.5
            * args.epsilon
            * args.epsilon
            * general_sum,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "per_neuron": per_neuron,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    log(
        f"complete: math_curvature_sum={math_sum:+.6e} "
        f"general_curvature_sum={general_sum:+.6e} "
        f"predicted_delta_at_epsilon={result['metadata']['quadratic_predicted_general_delta']:+.6e}"
    )
    log(f"saved {args.output}")


if __name__ == "__main__":
    main()
