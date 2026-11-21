#!/usr/bin/env python3
"""Train only the two members of curvature-initialized split neuron pairs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurspl.curvature_proxy import log
from neurspl.evaluate import ROOT, SYSTEM_PROMPT, select_device, select_dtype, synchronize
from neurspl.loss_eval import collate, formatted_solution, load_rows
from neurspl.trainable_split import install_trainable_split, save_adapter_checkpoint


DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "dev_v1" / "train_512.jsonl"
DEFAULT_REFINEMENT = ROOT / "runs" / "curvature_refine_qwen_layer12_refine.json"


def answer_only(row: dict) -> str:
    return f"Final answer: {row['answer']}"


def tokenize_target(
    tokenizer, row: dict, target: str, max_length: int
) -> tuple[list[int], list[int]]:
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question"]},
    ]
    full_messages = [*prompt_messages, {"role": "assistant", "content": target}]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_length]
    common = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        common += 1
    labels = [-100] * min(common, len(full_ids)) + full_ids[common:]
    if not any(value != -100 for value in labels):
        raise ValueError(f"No target tokens remained for {row['example_id']}")
    return full_ids, labels


def choose_epoch_targets(
    row_count: int,
    objective: str,
    answer_fraction: float,
    generator: random.Random,
) -> list[str]:
    if objective == "answer":
        return ["answer"] * row_count
    if objective == "solution":
        return ["solution"] * row_count
    return [
        "answer" if generator.random() < answer_fraction else "solution"
        for _ in range(row_count)
    ]


def sibling_statistics(adapter, pair_count: int) -> dict[str, float]:
    with torch.no_grad():
        gate_distance = (
            adapter.gate_rows[:pair_count] - adapter.gate_rows[pair_count:]
        ).norm(dim=1)
        up_distance = (
            adapter.up_rows[:pair_count] - adapter.up_rows[pair_count:]
        ).norm(dim=1)
        down_distance = (
            adapter.down_columns[:, :pair_count]
            - adapter.down_columns[:, pair_count:]
        ).norm(dim=0)
    return {
        "mean_gate_sibling_distance": float(gate_distance.mean().item()),
        "mean_up_sibling_distance": float(up_distance.mean().item()),
        "mean_down_sibling_distance": float(down_distance.mean().item()),
        "max_gate_sibling_distance": float(gate_distance.max().item()),
        "max_up_sibling_distance": float(up_distance.max().item()),
        "max_down_sibling_distance": float(down_distance.max().item()),
    }


def make_checkpoint_metadata(
    *,
    args,
    layer_index: int,
    indices: list[int],
    parameter_count_before: int,
    parameter_count_after: int,
    trainable_count: int,
    completed_steps: int,
    total_steps: int,
    status: str,
) -> dict[str, Any]:
    return {
        "method": "localized_split_sibling_training",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "refinement": str(args.refinement.resolve()),
        "layer": layer_index,
        "selected_neurons": indices,
        "top_k": args.top_k,
        "initialization": "curvature",
        "initial_epsilon": args.epsilon,
        "objective": args.objective,
        "answer_fraction": args.answer_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "anchor_strength": args.anchor_strength,
        "max_grad_norm": args.max_grad_norm,
        "max_length": args.max_length,
        "seed": args.seed,
        "parameters_before": parameter_count_before,
        "parameters_after": parameter_count_after,
        "architectural_added_parameters": parameter_count_after - parameter_count_before,
        "trainable_parameters": trainable_count,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--refinement", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument(
        "--objective", choices=("answer", "solution", "mixed"), default="mixed"
    )
    parser.add_argument("--answer-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--anchor-strength", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--checkpoint-every-steps", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()

    if args.top_k <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("top-k, epochs, and batch-size must be positive")
    if args.gradient_accumulation <= 0:
        raise ValueError("gradient-accumulation must be positive")
    if args.epsilon < 0 or args.learning_rate <= 0:
        raise ValueError("epsilon must be nonnegative and learning-rate positive")
    if not 0.0 <= args.answer_fraction <= 1.0:
        raise ValueError("answer-fraction must be in [0, 1]")
    if args.objective != "mixed" and args.answer_fraction != 0.5:
        raise ValueError("answer-fraction is only meaningful for the mixed objective")

    total_started = time.perf_counter()
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    refinement = json.loads(args.refinement.read_text())
    candidates = refinement["refined_candidates"][: args.top_k]
    if len(candidates) != args.top_k:
        raise ValueError("The refinement file does not contain enough candidates")
    layer_index = int(refinement["metadata"]["layer"])
    direction_by_neuron = {
        int(candidate["neuron_index"]): (
            candidate["direction_gate"],
            candidate["direction_up"],
        )
        for candidate in candidates
    }
    indices = sorted(direction_by_neuron)
    rows = load_rows(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Training data is empty")

    log(
        f"starting localized split training: examples={len(rows)} layer={layer_index} "
        f"top_k={args.top_k} epsilon={args.epsilon:g} objective={args.objective} "
        f"epochs={args.epochs} batch={args.batch_size} "
        f"accumulation={args.gradient_accumulation} device={device} dtype={dtype}"
    )
    log(f"selected neurons: {indices}")
    log(f"adapter checkpoint: {args.adapter_output}")
    log(f"result: {args.output}")
    log(f"loading tokenizer and model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to(device)
    model.requires_grad_(False)
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())

    gate_directions = torch.tensor(
        [direction_by_neuron[index][0] for index in indices], dtype=dtype
    )
    up_directions = torch.tensor(
        [direction_by_neuron[index][1] for index in indices], dtype=dtype
    )
    adapter, split_record = install_trainable_split(
        model,
        layer_index,
        indices,
        gate_directions=gate_directions,
        up_directions=up_directions,
        epsilon=args.epsilon,
    )
    parameter_count_after = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = adapter.trainable_parameters()
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    if parameter_count_after - parameter_count_before != split_record.added_parameters:
        raise RuntimeError("Trainable view changed the expanded architecture's parameter count")
    if any(parameter.requires_grad for parameter in adapter.base_mlp.parameters()):
        raise RuntimeError("A base-MLP parameter unexpectedly remains trainable")
    log(
        f"model ready: architectural_added={split_record.added_parameters} "
        f"trainable_coordinates={trainable_count}"
    )

    log("tokenizing answer and solution targets")
    answer_encoded = [
        tokenize_target(tokenizer, row, answer_only(row), args.max_length) for row in rows
    ]
    solution_encoded = [
        tokenize_target(tokenizer, row, formatted_solution(row), args.max_length)
        for row in rows
    ]
    answer_tokens = [sum(value != -100 for value in labels) for _, labels in answer_encoded]
    solution_tokens = [
        sum(value != -100 for value in labels) for _, labels in solution_encoded
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    anchors = [parameter.detach().clone() for parameter in trainable_parameters]
    initial_siblings = sibling_statistics(adapter, args.top_k)
    examples_per_step = args.batch_size * args.gradient_accumulation
    steps_per_epoch = math.ceil(len(rows) / examples_per_step)
    total_steps = steps_per_epoch * args.epochs
    generator = random.Random(args.seed)
    global_step = 0
    total_training_nll = 0.0
    total_training_tokens = 0
    history = []
    target_counts = {"answer": 0, "solution": 0}
    model.train()

    for epoch in range(args.epochs):
        order = list(range(len(rows)))
        generator.shuffle(order)
        modes = choose_epoch_targets(
            len(rows), args.objective, args.answer_fraction, generator
        )
        for start in range(0, len(order), examples_per_step):
            step_started = time.perf_counter()
            group_indices = order[start : start + examples_per_step]
            selected = []
            group_modes = {"answer": 0, "solution": 0}
            group_tokens = 0
            for index in group_indices:
                mode = modes[index]
                encoded = answer_encoded[index] if mode == "answer" else solution_encoded[index]
                tokens = answer_tokens[index] if mode == "answer" else solution_tokens[index]
                selected.append((encoded, mode))
                group_modes[mode] += 1
                group_tokens += tokens

            optimizer.zero_grad(set_to_none=True)
            group_nll = 0.0
            for micro_start in range(0, len(selected), args.batch_size):
                micro = selected[micro_start : micro_start + args.batch_size]
                batch = collate(tokenizer, [item[0] for item in micro])
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                loss_sum = functional.cross_entropy(
                    logits[:, :-1, :].float().transpose(1, 2),
                    batch["labels"][:, 1:],
                    reduction="sum",
                    ignore_index=-100,
                )
                (loss_sum / group_tokens).backward()
                group_nll += float(loss_sum.detach().item())
                del batch, logits, loss_sum

            anchor_penalty = torch.zeros((), device=device)
            if args.anchor_strength > 0:
                anchor_penalty = sum(
                    (parameter - anchor).square().sum()
                    for parameter, anchor in zip(trainable_parameters, anchors)
                ) / trainable_count
                (args.anchor_strength * anchor_penalty).backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.max_grad_norm
            )
            optimizer.step()
            synchronize(device)

            global_step += 1
            total_training_nll += group_nll
            total_training_tokens += group_tokens
            for mode, count in group_modes.items():
                target_counts[mode] += count
            measurement = {
                "step": global_step,
                "epoch": epoch + 1,
                "examples": len(group_indices),
                "answer_examples": group_modes["answer"],
                "solution_examples": group_modes["solution"],
                "target_tokens": group_tokens,
                "mean_nll": group_nll / group_tokens,
                "anchor_penalty": float(anchor_penalty.detach().item()),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - step_started,
            }
            history.append(measurement)
            log(
                f"epoch {epoch + 1}/{args.epochs} step {global_step}/{total_steps} "
                f"complete; examples={min(start + len(group_indices), len(rows))}/{len(rows)} "
                f"answer={group_modes['answer']} solution={group_modes['solution']} "
                f"tokens={group_tokens} nll={measurement['mean_nll']:.6f} "
                f"grad_norm={measurement['gradient_norm']:.4f} "
                f"step_seconds={measurement['elapsed_seconds']:.1f}"
            )

            if (
                args.checkpoint_every_steps > 0
                and global_step % args.checkpoint_every_steps == 0
            ):
                checkpoint_metadata = make_checkpoint_metadata(
                    args=args,
                    layer_index=layer_index,
                    indices=indices,
                    parameter_count_before=parameter_count_before,
                    parameter_count_after=parameter_count_after,
                    trainable_count=trainable_count,
                    completed_steps=global_step,
                    total_steps=total_steps,
                    status="partial",
                )
                save_adapter_checkpoint(args.adapter_output, adapter, checkpoint_metadata)
                log(f"checkpointed adapter at step {global_step}: {args.adapter_output}")
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    final_checkpoint_metadata = make_checkpoint_metadata(
        args=args,
        layer_index=layer_index,
        indices=indices,
        parameter_count_before=parameter_count_before,
        parameter_count_after=parameter_count_after,
        trainable_count=trainable_count,
        completed_steps=global_step,
        total_steps=total_steps,
        status="complete",
    )
    save_adapter_checkpoint(args.adapter_output, adapter, final_checkpoint_metadata)
    parameter_deltas = {
        name: float((parameter.detach() - anchor).norm().item())
        for name, parameter, anchor in zip(
            ("gate_rows", "up_rows", "down_columns"),
            trainable_parameters,
            anchors,
        )
    }
    result = {
        "metadata": {
            "schema_version": 1,
            **final_checkpoint_metadata,
            "device": device,
            "dtype": str(dtype),
            "examples": len(rows),
            "elapsed_seconds": time.perf_counter() - total_started,
            "adapter_output": str(args.adapter_output.resolve()),
        },
        "split_record": split_record.to_dict(),
        "target_counts": target_counts,
        "training_mean_nll": total_training_nll / total_training_tokens,
        "training_target_tokens": total_training_tokens,
        "initial_sibling_statistics": initial_siblings,
        "final_sibling_statistics": sibling_statistics(adapter, args.top_k),
        "parameter_delta_norms": parameter_deltas,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    log(
        f"complete: steps={global_step} mean_training_nll="
        f"{result['training_mean_nll']:.6f} adapter={args.adapter_output}"
    )
    log(f"saved result {args.output}")


if __name__ == "__main__":
    main()
