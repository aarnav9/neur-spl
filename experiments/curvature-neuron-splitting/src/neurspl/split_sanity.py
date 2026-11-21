#!/usr/bin/env python3
"""Verify that an in-memory neuron split preserves model logits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurspl.evaluate import ROOT, select_device, select_dtype, synchronize
from neurspl.split import split_swiglu_layer


DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    args = parser.parse_args()

    refinement_path = None
    if args.refinement is not None:
        refinement = json.loads(args.refinement.read_text())
        candidates = refinement["refined_candidates"][: args.top_k]
        if len(candidates) != args.top_k:
            raise ValueError("The refinement file does not contain enough candidates")
        indices = [int(candidate["neuron_index"]) for candidate in candidates]
        layer_index = int(refinement["metadata"]["layer"])
        refinement_path = str(args.refinement.resolve())
    else:
        indices = [int(value.strip()) for value in args.indices.split(",")]
        layer_index = args.layer

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=False
    ).to(device)
    model.eval()
    encoded = tokenizer(
        "Function-preserving neuron splitting should leave these logits unchanged.",
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        before = model(**encoded).logits.float().cpu()
    parameters_before = sum(parameter.numel() for parameter in model.parameters())
    record = split_swiglu_layer(
        model, layer_index, indices
    )
    parameters_after = sum(parameter.numel() for parameter in model.parameters())
    synchronize(device)
    with torch.inference_mode():
        after = model(**encoded).logits.float().cpu()
    synchronize(device)

    absolute = (after - before).abs()
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "refinement": refinement_path,
        "selected_neurons": list(record.original_indices),
        "device": device,
        "dtype": str(dtype),
        "split": record.to_dict(),
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
        "observed_added_parameters": parameters_after - parameters_before,
        "max_absolute_logit_difference": float(absolute.max().item()),
        "mean_absolute_logit_difference": float(absolute.mean().item()),
        "allclose_at_1e-5": bool(torch.allclose(before, after, atol=1e-5, rtol=1e-5)),
        "allclose_at_1e-4": bool(torch.allclose(before, after, atol=1e-4, rtol=1e-4)),
    }
    if result["observed_added_parameters"] != record.added_parameters:
        raise RuntimeError("Added parameter count does not match the split record")
    if not result["allclose_at_1e-4"]:
        raise RuntimeError(f"Split failed to preserve logits: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
