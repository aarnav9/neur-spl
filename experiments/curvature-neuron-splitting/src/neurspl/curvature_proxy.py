#!/usr/bin/env python3
"""Random-probe splitting-curvature proxy for one SwiGLU FFN layer.

For neuron activation a = silu(g @ x) * (u @ x), this computes directional
curvature along symmetric split directions without constructing a Hessian.
It is an inexpensive screening stage, not yet a minimum-eigenvalue estimate.
"""

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

from neurspl.evaluate import ROOT, select_device, select_dtype, synchronize
from neurspl.loss_eval import collate, load_rows, tokenize_example


DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"
DEFAULT_DATA = ROOT / "subsets" / "micro_v1" / "curvature_16.jsonl"


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def silu_derivatives(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sigmoid = torch.sigmoid(values)
    first = sigmoid + values * sigmoid * (1.0 - sigmoid)
    second = sigmoid * (1.0 - sigmoid) * (
        2.0 + values * (1.0 - 2.0 * sigmoid)
    )
    return first, second


def make_probe_directions(
    probes: int, hidden_size: int, seed: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    directions = torch.randn(probes, hidden_size * 2, generator=generator)
    directions /= directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    directions = directions.to(device)
    return directions[:, :hidden_size], directions[:, hidden_size:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    rows = load_rows(args.data)
    log(
        f"starting curvature proxy: examples={len(rows)} layer={args.layer} "
        f"probes={args.probes} device={device} dtype={dtype}"
    )
    log(f"loading tokenizer and model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=False
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    mlp = model.model.layers[args.layer].mlp
    hidden_size = mlp.gate_proj.in_features
    width = mlp.gate_proj.out_features
    direction_gate, direction_up = make_probe_directions(
        args.probes, hidden_size, args.seed, device
    )
    encoded = [tokenize_example(tokenizer, row, args.max_length) for row in rows]
    log(
        f"model ready: hidden={hidden_size} ffn_width={width}; "
        f"prepared {len(encoded)} examples"
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
    curvature_sum = torch.zeros(width, args.probes, device=device, dtype=torch.float32)
    total_target_tokens = 0
    start_example = 0
    checkpoint_signature = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "layer": args.layer,
        "examples": len(rows),
        "hidden_size": hidden_size,
        "ffn_width": width,
        "probes": args.probes,
        "probe_seed": args.seed,
    }
    if args.resume and args.checkpoint is not None and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if checkpoint.get("signature") != checkpoint_signature:
            raise ValueError("Checkpoint does not match the current curvature scan")
        start_example = int(checkpoint["next_example"])
        total_target_tokens = int(checkpoint["total_target_tokens"])
        curvature_sum.copy_(checkpoint["curvature_sum"].to(device))
        if not 0 <= start_example <= len(rows):
            raise ValueError("Checkpoint next_example is outside the dataset")
        log(
            f"resumed checkpoint {args.checkpoint}: examples={start_example}/{len(rows)} "
            f"target_tokens={total_target_tokens}"
        )
    batches = (len(rows) + args.batch_size - 1) // args.batch_size

    try:
        for start in range(start_example, len(rows), args.batch_size):
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
            mlp_input = captured["mlp_input"]
            activation_gradient = torch.autograd.grad(
                loss_sum, captured["activation"], retain_graph=False
            )[0]
            with torch.no_grad():
                gate_values = mlp.gate_proj(mlp_input)
                up_values = mlp.up_proj(mlp_input)
                silu_first, silu_second = silu_derivatives(gate_values)
                input_gate_projection = torch.einsum(
                    "bth,rh->btr", mlp_input.float(), direction_gate.float()
                )
                input_up_projection = torch.einsum(
                    "bth,rh->btr", mlp_input.float(), direction_up.float()
                )
                gradient = activation_gradient.float()
                gate_coefficient = gradient * silu_second.float() * up_values.float()
                cross_coefficient = 2.0 * gradient * silu_first.float()
                curvature_sum += torch.einsum(
                    "btw,btr->wr", gate_coefficient, input_gate_projection.square()
                )
                curvature_sum += torch.einsum(
                    "btw,btr->wr",
                    cross_coefficient,
                    input_gate_projection * input_up_projection,
                )

            total_target_tokens += target_tokens
            current = curvature_sum / total_target_tokens
            current_min = float(current.min().item())
            elapsed = time.perf_counter() - batch_started
            log(
                f"batch {batch_number}/{batches} complete; examples="
                f"{min(start + args.batch_size, len(rows))}/{len(rows)} "
                f"target_tokens={total_target_tokens} current_min={current_min:.6e} "
                f"batch_seconds={elapsed:.1f}"
            )
            del (
                logits,
                loss_sum,
                shift_labels,
                batch,
                mlp_input,
                gate_values,
                up_values,
                silu_first,
                silu_second,
                input_gate_projection,
                input_up_projection,
                activation_gradient,
                gradient,
                gate_coefficient,
                cross_coefficient,
                current,
            )
            captured.clear()
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            if args.checkpoint is not None:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_temp = args.checkpoint.with_suffix(
                    args.checkpoint.suffix + ".tmp"
                )
                torch.save(
                    {
                        "signature": checkpoint_signature,
                        "next_example": min(start + args.batch_size, len(rows)),
                        "total_target_tokens": total_target_tokens,
                        "curvature_sum": curvature_sum.cpu(),
                    },
                    checkpoint_temp,
                )
                checkpoint_temp.replace(args.checkpoint)
    finally:
        input_hook.remove()
        activation_hook.remove()

    scores = (curvature_sum / total_target_tokens).cpu()
    minimum_scores, minimum_probes = scores.min(dim=1)
    ordering = torch.argsort(minimum_scores)
    top_k = min(args.top_k, width)
    candidates = [
        {
            "rank": rank + 1,
            "neuron_index": int(index),
            "curvature": float(minimum_scores[index]),
            "probe_index": int(minimum_probes[index]),
        }
        for rank, index in enumerate(ordering[:top_k].tolist())
    ]
    all_neurons = [
        {
            "neuron_index": index,
            "minimum_probe_curvature": float(minimum_scores[index]),
            "minimum_probe_index": int(minimum_probes[index]),
        }
        for index in range(width)
    ]
    result = {
        "metadata": {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "analytic_swiglu_random_probe_splitting_curvature",
            "method_scope": "screening proxy; not a minimum-eigenvalue estimate",
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "layer": args.layer,
            "device": device,
            "dtype": str(dtype),
            "examples": len(rows),
            "target_tokens": total_target_tokens,
            "hidden_size": hidden_size,
            "ffn_width": width,
            "probes": args.probes,
            "probe_seed": args.seed,
            "resumed_from_example": start_example,
            "negative_neurons": int(minimum_scores.lt(0).sum().item()),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "top_candidates": candidates,
        "neurons": all_neurons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    log(
        f"complete: negative_neurons={result['metadata']['negative_neurons']}/{width} "
        f"best_neuron={candidates[0]['neuron_index']} "
        f"best_curvature={candidates[0]['curvature']:.6e}"
    )
    log(f"saved {args.output}")


if __name__ == "__main__":
    main()
