#!/usr/bin/env python3
"""Evaluate exact-budget joint Fisher/spectral masks with BN recalibration."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from cut_to_grow.diagnostics import aggregate_channel_scores, channel_magnitude_scores
from cut_to_grow.pilot import (
    build_cifar100_loaders,
    build_synthetic_loaders,
    evaluate,
    load_checkpoint,
    model_from_checkpoint,
    resolve_device,
    seed_everything,
)
from cut_to_grow.spectral_fisher import (
    collect_gate_gradient_matrices,
    ridge_leverage_scores,
)
from cut_to_grow.spectral_pruning import (
    JointChannelMask,
    group_channel_scales,
    operator_trace_ratio,
    recalibrate_batch_norm,
    ridge_relative_operator_error,
    sampled_rescaled_mask,
    topk_mask,
)


PROTOCOL = "joint-spectral-mask-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("pull/raw"))
    parser.add_argument("--data-source", choices=("torchvision", "huggingface"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--score-batches", type=int, default=8)
    parser.add_argument("--bn-batches", type=int, default=8)
    parser.add_argument("--evaluation-batches", type=int)
    parser.add_argument(
        "--sparsity",
        type=float,
        action="append",
        dest="sparsities",
        help="fraction of structured groups removed; repeat for multiple values",
    )
    parser.add_argument("--spectral-replicates", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-examples", type=int, default=200)
    return parser.parse_args()


def _append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()


def _load_existing(path: Path, expected: dict[str, object]) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        for name, value in expected.items():
            if record.get(name) != value:
                raise SystemExit(
                    f"existing output protocol mismatch for {name}: "
                    f"{record.get(name)!r} != {value!r}"
                )
    return records


def _record_key(record: dict[str, object]) -> tuple[str, float, int]:
    return (
        str(record["method"]),
        float(record["target_sparsity"]),
        int(record["replicate"]),
    )


def _validate_metrics(name: str, metrics: dict[str, float]) -> None:
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError(f"non-finite {name} metrics: {metrics}")


def main() -> None:
    args = parse_args()
    sparsities = args.sparsities or [0.25, 0.50, 0.75]
    if any(not 0.0 < sparsity < 1.0 for sparsity in sparsities):
        raise SystemExit("every --sparsity must be in (0, 1)")
    if len(set(sparsities)) != len(sparsities):
        raise SystemExit("duplicate sparsities are not allowed")
    if args.score_batches < 1 or args.bn_batches < 1:
        raise SystemExit("score and BN batch counts must be positive")
    if args.group_size < 1 or args.spectral_replicates < 1:
        raise SystemExit("group size and replicate count must be positive")

    payload = load_checkpoint(args.checkpoint)
    config = payload["config"]
    checkpoint_seed = int(payload["seed"])
    seed_everything(checkpoint_seed)
    evaluation_device = resolve_device(args.device)
    batch_size = int(args.batch_size or config["training"]["batch_size"])

    if args.smoke:
        fit_loader, selection_loader = build_synthetic_loaders(
            examples=args.smoke_examples,
            num_classes=int(payload["num_classes"]),
            batch_size=batch_size,
            seed=checkpoint_seed,
        )
    else:
        fit_loader, selection_loader = build_cifar100_loaders(
            args.data_root,
            split_seed=int(config["data"]["split_seed"]),
            selection_examples=int(config["data"]["selection_examples"]),
            batch_size=batch_size,
            num_workers=args.num_workers,
            download=False,
            source=args.data_source or config["data"]["source"],
        )

    # Materialize once so every scorer and BN recalibration sees exactly the
    # same augmented examples in the same order.
    needed_batches = max(args.score_batches, args.bn_batches)
    fit_batches = []
    for batch_index, batch in enumerate(fit_loader):
        if batch_index >= needed_batches:
            break
        fit_batches.append(batch)
    if len(fit_batches) < needed_batches:
        raise SystemExit(
            f"fit loader yielded {len(fit_batches)} batches; needed {needed_batches}"
        )

    score_model = model_from_checkpoint(payload).to("cpu")
    groups = list(score_model.channel_groups())[1:]
    score_start = perf_counter()
    matrices = collect_gate_gradient_matrices(
        score_model,
        fit_batches[: args.score_batches],
        groups,
        torch.device("cpu"),
        group_size=args.group_size,
        max_batches=args.score_batches,
    )

    keys: list[tuple[str, int]] = []
    columns: list[torch.Tensor] = []
    magnitude_values: list[float] = []
    for group in groups:
        matrix = matrices[group.name]
        magnitude = aggregate_channel_scores(
            channel_magnitude_scores(score_model, group), args.group_size
        )
        if matrix.shape[1] != magnitude.numel():
            raise RuntimeError(f"score columns do not align for {group.name}")
        for group_index in range(matrix.shape[1]):
            keys.append((group.name, group_index))
            columns.append(matrix[:, group_index])
            magnitude_values.append(float(magnitude[group_index]))
    gradient_matrix = torch.stack(columns, dim=1)
    fisher, leverage, ridge = ridge_leverage_scores(gradient_matrix)
    magnitude = torch.tensor(magnitude_values, dtype=torch.float64)
    scoring_seconds = perf_counter() - score_start
    del score_model

    output = args.output or args.checkpoint.with_name(
        args.checkpoint.stem + "_joint_spectral_masks.jsonl"
    )
    fixed_fields: dict[str, object] = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "checkpoint_seed": checkpoint_seed,
        "group_size": args.group_size,
        "score_batches": args.score_batches,
        "bn_batches": args.bn_batches,
        "evaluation_batches": args.evaluation_batches,
        "spectral_replicates": args.spectral_replicates,
        "smoke": args.smoke,
    }
    records = _load_existing(output, fixed_fields)
    completed = {_record_key(record) for record in records}
    criterion = nn.CrossEntropyLoss()

    def evaluate_scales(
        *,
        method: str,
        target_sparsity: float,
        replicate: int,
        scales: torch.Tensor,
        inclusion_probabilities: torch.Tensor | None = None,
    ) -> None:
        key = (method, target_sparsity, replicate)
        if key in completed:
            print(f"reuse method={method} sparsity={target_sparsity:.2f} replicate={replicate}")
            return
        retained = scales != 0
        values = {
            group_key: float(scales[index]) for index, group_key in enumerate(keys)
        }
        gate_scales = group_channel_scales(
            groups,
            group_size=args.group_size,
            values=values,
        )
        model = model_from_checkpoint(payload).to(evaluation_device)
        start = perf_counter()
        with JointChannelMask(model, gate_scales):
            immediate = evaluate(
                model,
                selection_loader,
                criterion,
                evaluation_device,
                max_batches=args.evaluation_batches,
            )
            bn_examples = recalibrate_batch_norm(
                model,
                fit_batches[: args.bn_batches],
                evaluation_device,
            )
            recalibrated = evaluate(
                model,
                selection_loader,
                criterion,
                evaluation_device,
                max_batches=args.evaluation_batches,
            )
        evaluation_seconds = perf_counter() - start
        _validate_metrics("immediate", immediate)
        _validate_metrics("recalibrated", recalibrated)

        record: dict[str, object] = dict(fixed_fields)
        record.update(
            {
                "method": method,
                "target_sparsity": target_sparsity,
                "replicate": replicate,
                "groups_total": len(keys),
                "groups_retained": int(retained.sum()),
                "actual_sparsity": 1.0 - int(retained.sum()) / len(keys),
                "maximum_activation_scale": float(scales.max()),
                "mean_retained_activation_scale": float(scales[retained].mean()),
                "operator_relative_error": ridge_relative_operator_error(
                    gradient_matrix,
                    scales,
                    ridge=ridge,
                ),
                "operator_trace_ratio": operator_trace_ratio(gradient_matrix, scales),
                "spectral_fisher_ridge": ridge,
                "score_examples": gradient_matrix.shape[0],
                "scoring_seconds": scoring_seconds,
                "bn_calibration_examples": bn_examples,
                "evaluation_device": str(evaluation_device),
                "evaluation_seconds": evaluation_seconds,
                "immediate_loss": immediate["loss"],
                "immediate_accuracy": immediate["accuracy"],
                "recalibrated_loss": recalibrated["loss"],
                "recalibrated_accuracy": recalibrated["accuracy"],
                "evaluation_examples": immediate["examples"],
            }
        )
        if inclusion_probabilities is not None:
            selected_probabilities = inclusion_probabilities[retained]
            record["minimum_retained_inclusion_probability"] = float(
                selected_probabilities.min()
            )
            record["maximum_inclusion_probability"] = float(
                inclusion_probabilities.max()
            )
        _append_record(output, record)
        records.append(record)
        completed.add(key)
        print(
            json.dumps(
                {
                    "method": method,
                    "sparsity": target_sparsity,
                    "replicate": replicate,
                    "retained": int(retained.sum()),
                    "immediate_accuracy": immediate["accuracy"],
                    "recalibrated_accuracy": recalibrated["accuracy"],
                    "operator_error": record["operator_relative_error"],
                    "seconds": evaluation_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del model
        if evaluation_device.type == "mps":
            torch.mps.empty_cache()

    evaluate_scales(
        method="dense",
        target_sparsity=0.0,
        replicate=0,
        scales=torch.ones(len(keys), dtype=torch.float64),
    )

    for sparsity in sparsities:
        keep_count = round((1.0 - sparsity) * len(keys))
        for method, scores in (
            ("magnitude_topk", magnitude),
            ("fisher_topk", fisher),
            ("spectral_topk_unweighted", leverage),
        ):
            retained = topk_mask(scores, keep_count)
            evaluate_scales(
                method=method,
                target_sparsity=sparsity,
                replicate=0,
                scales=retained.to(dtype=torch.float64),
            )
        for replicate in range(args.spectral_replicates):
            retained, scales, probabilities = sampled_rescaled_mask(
                leverage,
                keep_count=keep_count,
                seed=checkpoint_seed + 10_000 * round(sparsity * 100) + replicate,
            )
            if int(retained.sum()) != keep_count:
                raise RuntimeError("sampled spectral mask violated the fixed budget")
            evaluate_scales(
                method="spectral_sampled_rescaled",
                target_sparsity=sparsity,
                replicate=replicate,
                scales=scales,
                inclusion_probabilities=probabilities,
            )

    by_method_sparsity: dict[tuple[str, float], list[float]] = {}
    for record in records:
        key = (str(record["method"]), float(record["target_sparsity"]))
        by_method_sparsity.setdefault(key, []).append(float(record["recalibrated_accuracy"]))
    summary: dict[str, dict[str, float]] = {}
    for method in ("spectral_topk_unweighted", "spectral_sampled_rescaled"):
        differences: list[float] = []
        for sparsity in sparsities:
            fisher_accuracy = statistics.mean(by_method_sparsity[("fisher_topk", sparsity)])
            spectral_accuracy = statistics.mean(by_method_sparsity[(method, sparsity)])
            differences.append(spectral_accuracy - fisher_accuracy)
        passes = sum(difference >= 0.005 for difference in differences) >= 2 and all(
            difference >= -0.005 for difference in differences
        )
        summary[method] = {
            "minimum_difference": min(differences),
            "maximum_difference": max(differences),
            "passing_sparsities": float(sum(value >= 0.005 for value in differences)),
            "passes_frozen_gate": float(passes),
        }
    print(json.dumps({"output": str(output), "frozen_gate": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
