"""Counterfactual output-channel deletion diagnostics."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from torch import Tensor, nn

from .block_flow import (
    ChannelFlowScores,
    ClassConditionalGateCapacities,
    ResidualGateCapacities,
)
from .model import ChannelGroup, CifarResNet
from .pilot import evaluate
from .global_flow import GlobalClassGateCapacities
from .torch_capacity import (
    GateSensitivityMonitor,
    GateSensitivityScores,
    TransportCapacityMonitor,
)


class ChannelAblation(AbstractContextManager["ChannelAblation"]):
    """Temporarily zero one output channel at a group's BatchNorm gate."""

    def __init__(self, model: nn.Module, group: ChannelGroup, channel: int) -> None:
        if not 0 <= channel < group.channels:
            raise ValueError(f"channel {channel} is outside [0, {group.channels})")
        modules = dict(model.named_modules())
        if group.gate not in modules:
            raise KeyError(f"unknown gate module {group.gate!r}")
        self.module = modules[group.gate]
        self.channel = channel
        self.handle: object | None = None

    def __enter__(self) -> "ChannelAblation":
        if self.handle is not None:
            raise RuntimeError("ablation is already active")

        def zero_channel(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
            if output.ndim < 2 or self.channel >= output.shape[1]:
                raise RuntimeError("gate output does not have the declared channel dimension")
            modified = output.clone()
            modified[:, self.channel] = 0
            return modified

        self.handle = self.module.register_forward_hook(zero_channel)
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            self.handle.remove()  # type: ignore[union-attr]
            self.handle = None


class ChannelRangeAblation(AbstractContextManager["ChannelRangeAblation"]):
    """Temporarily zero a contiguous output-channel range at a BatchNorm gate."""

    def __init__(
        self, model: nn.Module, group: ChannelGroup, start: int, end: int
    ) -> None:
        if not 0 <= start < end <= group.channels:
            raise ValueError(
                f"channel range [{start}, {end}) is outside [0, {group.channels})"
            )
        modules = dict(model.named_modules())
        if group.gate not in modules:
            raise KeyError(f"unknown gate module {group.gate!r}")
        self.module = modules[group.gate]
        self.start = start
        self.end = end
        self.handle: object | None = None

    def __enter__(self) -> "ChannelRangeAblation":
        if self.handle is not None:
            raise RuntimeError("ablation is already active")

        def zero_channels(
            _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
        ) -> Tensor:
            if output.ndim < 2 or self.end > output.shape[1]:
                raise RuntimeError("gate output does not have the declared channel range")
            modified = output.clone()
            modified[:, self.start : self.end] = 0
            return modified

        self.handle = self.module.register_forward_hook(zero_channels)
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            self.handle.remove()  # type: ignore[union-attr]
            self.handle = None


def channel_magnitude_scores(model: nn.Module, group: ChannelGroup) -> Tensor:
    modules = dict(model.named_modules())
    producer = modules.get(group.producer)
    if not isinstance(producer, nn.Conv2d):
        raise TypeError(f"producer {group.producer!r} is not a Conv2d")
    return producer.weight.detach().abs().reshape(group.channels, -1).mean(dim=1).cpu()


def collect_transport_weight_scores(
    model: CifarResNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    groups: Sequence[ChannelGroup],
    device: torch.device,
    *,
    max_batches: int = 8,
    decay: float = 0.95,
) -> dict[str, Tensor]:
    """Collect task-conditioned scores aligned to convolution weights."""

    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    was_training = model.training
    model.eval()
    selected = [group.producer for group in groups]
    criterion = nn.CrossEntropyLoss()
    monitor = TransportCapacityMonitor(
        model,
        decay=decay,
        module_names=selected,
    ).attach()
    try:
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
    finally:
        monitor.close()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    missing = set(selected) - set(monitor.scores)
    if missing:
        raise RuntimeError(f"no transport observations for modules: {sorted(missing)}")
    return {name: monitor.scores[name].clone() for name in selected}


def output_channel_transport_scores(
    weight_scores: Mapping[str, Tensor], groups: Sequence[ChannelGroup]
) -> dict[str, Tensor]:
    output: dict[str, Tensor] = {}
    for group in groups:
        score = weight_scores[group.producer]
        output[group.name] = score.reshape(score.shape[0], -1).sum(dim=1)
    return output


def collect_transport_scores(
    model: CifarResNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    groups: Sequence[ChannelGroup],
    device: torch.device,
    *,
    max_batches: int = 8,
    decay: float = 0.95,
) -> dict[str, Tensor]:
    """Compatibility wrapper returning output-channel transport scores."""

    weight_scores = collect_transport_weight_scores(
        model,
        loader,
        groups,
        device,
        max_batches=max_batches,
        decay=decay,
    )
    return output_channel_transport_scores(weight_scores, groups)


def collect_skip_aware_capacities(
    model: CifarResNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    groups: Sequence[ChannelGroup],
    device: torch.device,
    *,
    max_batches: int = 8,
    decay: float = 0.95,
) -> tuple[dict[str, Tensor], dict[str, ResidualGateCapacities]]:
    """Collect convolution and residual-gate capacities on identical batches."""

    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    selected_producers = [group.producer for group in groups]
    prefixes = sorted({group.name.rsplit(".", 1)[0] for group in groups})
    expected_names = {
        f"{prefix}.{convolution}"
        for prefix in prefixes
        for convolution in ("conv1", "conv2")
    }
    if set(selected_producers) != expected_names:
        raise ValueError("skip-aware collection requires both convolutions of each block")
    gate_names = [
        f"{prefix}.{gate}"
        for prefix in prefixes
        for gate in (
            "branch_transport_gate",
            "skip_transport_gate",
            "merge_transport_gate",
        )
    ]

    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss()
    transport_monitor = TransportCapacityMonitor(
        model,
        decay=decay,
        module_names=selected_producers,
    ).attach()
    gate_monitor = GateSensitivityMonitor(
        model,
        module_names=gate_names,
        group_size=1,
        capacity_decay=decay,
    )
    try:
        gate_monitor.attach()
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
    finally:
        gate_monitor.close()
        transport_monitor.close()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    missing = set(selected_producers) - set(transport_monitor.scores)
    if missing:
        raise RuntimeError(f"no transport observations for modules: {sorted(missing)}")
    weight_scores = {
        name: transport_monitor.scores[name].clone() for name in selected_producers
    }
    gate_capacities: dict[str, ResidualGateCapacities] = {}
    for prefix in prefixes:
        branch = gate_monitor.scores(f"{prefix}.branch_transport_gate")
        skip = gate_monitor.scores(f"{prefix}.skip_transport_gate")
        merged = gate_monitor.scores(f"{prefix}.merge_transport_gate")
        if len({branch.examples, skip.examples, merged.examples}) != 1:
            raise RuntimeError(f"residual gates saw inconsistent examples for {prefix}")
        if not (
            branch.mean_abs_transport.shape
            == skip.mean_abs_transport.shape
            == merged.mean_abs_transport.shape
        ):
            raise RuntimeError(f"residual gate capacities do not align for {prefix}")
        gate_capacities[prefix] = ResidualGateCapacities(
            branch=branch.mean_abs_transport,
            skip=skip.mean_abs_transport,
            merged=merged.mean_abs_transport,
            examples=branch.examples,
        )
    return weight_scores, gate_capacities


def collect_class_conditioned_capacities(
    model: CifarResNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    groups: Sequence[ChannelGroup],
    device: torch.device,
    *,
    class_to_commodity: Sequence[int],
    max_batches: int = 8,
    decay: float = 0.95,
) -> tuple[dict[str, Tensor], dict[str, ClassConditionalGateCapacities]]:
    """Collect base edge transport and per-commodity residual gate demand."""

    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    if not class_to_commodity:
        raise ValueError("class_to_commodity must be nonempty")
    selected_producers = [group.producer for group in groups]
    prefixes = sorted({group.name.rsplit(".", 1)[0] for group in groups})
    expected_names = {
        f"{prefix}.{convolution}"
        for prefix in prefixes
        for convolution in ("conv1", "conv2")
    }
    if set(selected_producers) != expected_names:
        raise ValueError("class-conditioned collection requires both block convolutions")
    gate_names = [
        f"{prefix}.{gate}"
        for prefix in prefixes
        for gate in ("bn1", "branch_transport_gate")
    ]

    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss()
    transport_monitor = TransportCapacityMonitor(
        model,
        decay=decay,
        module_names=selected_producers,
    ).attach()
    gate_monitor = GateSensitivityMonitor(
        model,
        module_names=gate_names,
        group_size=1,
        capacity_decay=decay,
    )
    try:
        gate_monitor.attach()
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            target_values = [int(target) for target in targets.tolist()]
            if any(not 0 <= target < len(class_to_commodity) for target in target_values):
                raise ValueError("target is outside class_to_commodity")
            commodities = torch.tensor(
                [class_to_commodity[target] for target in target_values],
                dtype=torch.int64,
            )
            gate_monitor.set_sample_groups(commodities)
            model.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
    finally:
        gate_monitor.close()
        transport_monitor.close()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    missing = set(selected_producers) - set(transport_monitor.scores)
    if missing:
        raise RuntimeError(f"no transport observations for modules: {sorted(missing)}")
    weight_scores = {
        name: transport_monitor.scores[name].clone() for name in selected_producers
    }
    conditional: dict[str, ClassConditionalGateCapacities] = {}
    for prefix in prefixes:
        hidden_name = f"{prefix}.bn1"
        output_name = f"{prefix}.branch_transport_gate"
        hidden = gate_monitor.conditional_mean_abs_transport(hidden_name)
        branch_output = gate_monitor.conditional_mean_abs_transport(output_name)
        hidden_examples = gate_monitor.conditional_examples(hidden_name)
        output_examples = gate_monitor.conditional_examples(output_name)
        if hidden_examples != output_examples or set(hidden) != set(branch_output):
            raise RuntimeError(f"commodity observations do not align for {prefix}")
        conditional[prefix] = ClassConditionalGateCapacities(
            hidden=hidden,
            output=branch_output,
            examples=hidden_examples,
        )
    return weight_scores, conditional


def collect_global_class_capacities(
    model: CifarResNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    class_to_commodity: Sequence[int],
    max_batches: int = 8,
    decay: float = 0.95,
) -> tuple[dict[str, Tensor], GlobalClassGateCapacities]:
    """Collect all edge and gate capacities for the global commodity graph."""

    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    if not class_to_commodity:
        raise ValueError("class_to_commodity must be nonempty")
    weight_names = ["conv1", "fc"]
    gate_names = ["bn1", "pool"]
    for layer_index, layer in enumerate(
        (model.layer1, model.layer2, model.layer3, model.layer4), start=1
    ):
        for block_index, block in enumerate(layer):
            prefix = f"layer{layer_index}.{block_index}"
            weight_names.extend((f"{prefix}.conv1", f"{prefix}.conv2"))
            if block.downsample is not None:
                weight_names.append(f"{prefix}.downsample.0")
            gate_names.extend(
                (
                    f"{prefix}.bn1",
                    f"{prefix}.branch_transport_gate",
                    f"{prefix}.skip_transport_gate",
                    f"{prefix}.merge_transport_gate",
                )
            )

    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss()
    transport_monitor = TransportCapacityMonitor(
        model,
        decay=decay,
        module_names=weight_names,
    ).attach()
    gate_monitor = GateSensitivityMonitor(
        model,
        module_names=gate_names,
        group_size=1,
        capacity_decay=decay,
    )
    try:
        gate_monitor.attach()
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            target_values = [int(target) for target in targets.tolist()]
            if any(not 0 <= target < len(class_to_commodity) for target in target_values):
                raise ValueError("target is outside class_to_commodity")
            commodities = torch.tensor(
                [class_to_commodity[target] for target in target_values],
                dtype=torch.int64,
            )
            gate_monitor.set_sample_groups(commodities)
            model.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
    finally:
        gate_monitor.close()
        transport_monitor.close()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    missing = set(weight_names) - set(transport_monitor.scores)
    if missing:
        raise RuntimeError(f"no global transport observations for: {sorted(missing)}")
    weight_scores = {
        name: transport_monitor.scores[name].clone() for name in weight_names
    }
    by_gate: dict[str, dict[int, Tensor]] = {}
    examples: dict[int, int] | None = None
    for name in gate_names:
        by_gate[name] = gate_monitor.conditional_mean_abs_transport(name)
        gate_examples = gate_monitor.conditional_examples(name)
        if examples is None:
            examples = gate_examples
        elif examples != gate_examples:
            raise RuntimeError(f"global gate commodity counts do not align at {name}")
    if examples is None:  # pragma: no cover - gate_names is statically nonempty
        raise RuntimeError("no global gate capacities were collected")
    return weight_scores, GlobalClassGateCapacities(
        by_gate=by_gate,
        examples=examples,
        fine_to_coarse=tuple(int(value) for value in class_to_commodity),
    )


def collect_gate_sensitivity_scores(
    model: CifarResNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    groups: Sequence[ChannelGroup],
    device: torch.device,
    *,
    group_size: int,
    max_batches: int = 8,
) -> dict[str, GateSensitivityScores]:
    """Collect first-order Taylor and Group Fisher scores at deletion gates.

    Each gate uses the same contiguous mask as the structured deletion probe.
    The Taylor score is the absolute calibration-set mean of the signed mask
    gradient. Group Fisher is the mean squared per-example mask gradient.
    """

    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if not groups:
        raise ValueError("groups must be nonempty")
    selected = [group.gate for group in groups]
    if len(selected) != len(set(selected)):
        raise ValueError("group gates must be unique")

    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss()
    monitor = GateSensitivityMonitor(
        model,
        module_names=selected,
        group_size=group_size,
    ).attach()
    try:
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
    finally:
        monitor.close()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    output: dict[str, GateSensitivityScores] = {}
    for group in groups:
        scores = monitor.scores(group.gate)
        expected_groups = math.ceil(group.channels / group_size)
        if scores.taylor.numel() != expected_groups:
            raise RuntimeError(
                f"gate score length does not match {group.name}: "
                f"{scores.taylor.numel()} != {expected_groups}"
            )
        output[group.name] = scores
    return output


def rank_stratified_channels(scores: Tensor, count: int) -> tuple[int, ...]:
    """Choose channels evenly across a score ranking, including both tails."""

    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty vector")
    if count < 1:
        raise ValueError("count must be positive")
    order = torch.argsort(scores).tolist()
    if count >= len(order):
        return tuple(order)
    positions = torch.linspace(0, len(order) - 1, steps=count).round().to(torch.int64).tolist()
    return tuple(order[position] for position in positions)


def aggregate_channel_scores(scores: Tensor, group_size: int) -> Tensor:
    """Sum a channel score vector into contiguous structured groups."""

    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty vector")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    return torch.stack(
        [
            scores[start : start + group_size].sum()
            for start in range(0, scores.numel(), group_size)
        ]
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Tie-aware Spearman correlation without a SciPy dependency."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_rank, right_rank, strict=True)
    )
    left_energy = sum((value - left_mean) ** 2 for value in left_rank)
    right_energy = sum((value - right_mean) ** 2 for value in right_rank)
    denominator = math.sqrt(left_energy * right_energy)
    return float("nan") if denominator == 0.0 else numerator / denominator


def rank_percentile_scores(values: Sequence[float]) -> list[float]:
    """Return tie-aware percentile ranks using no outcome information."""

    if not values:
        raise ValueError("rank percentile inputs must be nonempty")
    ranks = _average_ranks(values)
    return [(rank - 0.5) / len(values) for rank in ranks]


def summarize_probe_correlations(
    records: Sequence[Mapping[str, object]],
    score_names: Sequence[str] | None = None,
    *,
    accuracy_damage_threshold: float = 0.02,
) -> dict[str, dict[str, float | int]]:
    """Return pooled and equal-layer-weighted score/damage correlations."""

    if not records:
        raise ValueError("probe records are empty")
    by_group: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        by_group[str(record["group"])].append(record)

    if score_names is None:
        candidates = (
            "magnitude",
            "transport",
            "taylor",
            "group_fisher",
            "spectral_fisher_leverage",
            "class_flow_worst",
            "class_flow_top3_mean",
            "class_flow_mean",
            "global_class_flow_worst",
            "global_class_flow_top3_mean",
            "global_class_flow_mean",
            "concurrent_flow_dual",
            "skip_flow_marginal",
            "flow_marginal",
            "flow_group_marginal",
            "cut_incident",
            "cut_group_incident",
        )
        score_names = tuple(name for name in candidates if name in records[0])

    summary: dict[str, dict[str, float | int]] = {}
    for score_name in score_names:
        pooled = spearman_correlation(
            [float(record[score_name]) for record in records],
            [float(record["loss_damage"]) for record in records],
        )
        group_values: list[float] = []
        for group_records in by_group.values():
            if len(group_records) < 2:
                continue
            correlation = spearman_correlation(
                [float(record[score_name]) for record in group_records],
                [float(record["loss_damage"]) for record in group_records],
            )
            if not math.isnan(correlation):
                group_values.append(correlation)
        scores = [float(record[score_name]) for record in records]
        top_labels = _top_damage_labels(records, fraction=0.05)
        threshold_labels = [
            float(record["accuracy_damage"]) >= accuracy_damage_threshold
            for record in records
        ]
        macro_ci = _bootstrap_mean_ci(group_values, seed=20260824)
        top_auroc = binary_auroc(scores, top_labels)
        top_auroc_ci = _bootstrap_binary_ci(
            scores, top_labels, binary_auroc, seed=20260825
        )
        threshold_auroc = binary_auroc(scores, threshold_labels)
        threshold_auroc_ci = _bootstrap_binary_ci(
            scores, threshold_labels, binary_auroc, seed=20260826
        )
        within_layer_scores = _within_group_rank_scores(records, score_name)
        shuffle_mean, shuffle_low, shuffle_high, shuffle_pvalue = (
            _within_group_shuffle_auroc(
                scores,
                threshold_labels,
                [str(record["group"]) for record in records],
                observed=threshold_auroc,
                seed=20260827,
            )
        )
        summary[score_name] = {
            "pooled_spearman": pooled,
            "layer_macro_spearman": (
                sum(group_values) / len(group_values) if group_values else float("nan")
            ),
            "layer_macro_spearman_ci95_low": macro_ci[0],
            "layer_macro_spearman_ci95_high": macro_ci[1],
            "layers": len(group_values),
            "probes": len(records),
            "top_5_percent_damage_positives": sum(top_labels),
            "top_5_percent_damage_auroc": top_auroc,
            "top_5_percent_damage_auroc_ci95_low": top_auroc_ci[0],
            "top_5_percent_damage_auroc_ci95_high": top_auroc_ci[1],
            "top_5_percent_damage_average_precision": binary_average_precision(
                scores, top_labels
            ),
            "accuracy_drop_2pt_positives": sum(threshold_labels),
            "accuracy_drop_2pt_auroc": threshold_auroc,
            "accuracy_drop_2pt_auroc_ci95_low": threshold_auroc_ci[0],
            "accuracy_drop_2pt_auroc_ci95_high": threshold_auroc_ci[1],
            "accuracy_drop_2pt_average_precision": binary_average_precision(
                scores, threshold_labels
            ),
            "within_layer_rank_accuracy_drop_2pt_auroc": binary_auroc(
                within_layer_scores, threshold_labels
            ),
            "within_layer_rank_accuracy_drop_2pt_average_precision": (
                binary_average_precision(within_layer_scores, threshold_labels)
            ),
            "within_layer_shuffle_accuracy_drop_2pt_auroc_mean": shuffle_mean,
            "within_layer_shuffle_accuracy_drop_2pt_auroc_ci95_low": shuffle_low,
            "within_layer_shuffle_accuracy_drop_2pt_auroc_ci95_high": shuffle_high,
            "within_layer_shuffle_accuracy_drop_2pt_auroc_pvalue": shuffle_pvalue,
        }
    return summary


def top_damage_auroc(
    records: Sequence[Mapping[str, object]], score_name: str, *, fraction: float
) -> float:
    """Tie-aware pooled AUROC for identifying the most damaging deletions."""

    if not records:
        raise ValueError("probe records are empty")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    labels = _top_damage_labels(records, fraction=fraction)
    scores = [float(record[score_name]) for record in records]
    return binary_auroc(scores, labels)


def _top_damage_labels(
    records: Sequence[Mapping[str, object]], *, fraction: float
) -> list[bool]:
    if not records:
        raise ValueError("probe records are empty")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    positive_count = max(1, math.ceil(len(records) * fraction))
    damage_order = sorted(
        range(len(records)),
        key=lambda index: float(records[index]["loss_damage"]),
        reverse=True,
    )
    positives = set(damage_order[:positive_count])
    return [index in positives for index in range(len(records))]


def binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Tie-aware AUROC for binary labels."""

    if len(scores) != len(labels) or not scores:
        raise ValueError("binary metric inputs must be nonempty and aligned")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def binary_average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Threshold-grouped average precision for binary labels."""

    if len(scores) != len(labels) or not scores:
        raise ValueError("binary metric inputs must be nonempty and aligned")
    positive_count = sum(labels)
    if positive_count == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    average_precision = 0.0
    true_positives = 0
    processed = 0
    previous_recall = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        true_positives += sum(labels[index] for index in order[start:end])
        processed = end
        recall = true_positives / positive_count
        precision = true_positives / processed
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return average_precision


def paired_binary_metric_difference_ci(
    left_scores: Sequence[float],
    right_scores: Sequence[float],
    labels: Sequence[bool],
    metric: Callable[[Sequence[float], Sequence[bool]], float],
    *,
    seed: int,
    replicates: int = 5000,
) -> tuple[float, float, float]:
    """Return a paired stratified-bootstrap difference and its 95% interval."""

    if len(left_scores) != len(right_scores) or len(left_scores) != len(labels):
        raise ValueError("paired metric inputs must be aligned")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    if not positives or not negatives:
        return float("nan"), float("nan"), float("nan")
    observed = metric(left_scores, labels) - metric(right_scores, labels)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        indices = [rng.choice(positives) for _ in positives]
        indices.extend(rng.choice(negatives) for _ in negatives)
        sampled_labels = [labels[index] for index in indices]
        left = [left_scores[index] for index in indices]
        right = [right_scores[index] for index in indices]
        estimates.append(metric(left, sampled_labels) - metric(right, sampled_labels))
    return observed, _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _within_group_rank_scores(
    records: Sequence[Mapping[str, object]], score_name: str
) -> list[float]:
    """Replace raw scores with tie-aware rank percentiles inside each layer."""

    by_group: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_group[str(record["group"])].append(index)
    output = [0.0] * len(records)
    for indices in by_group.values():
        ranks = _average_ranks([float(records[index][score_name]) for index in indices])
        for index, rank in zip(indices, ranks, strict=True):
            output[index] = (rank - 0.5) / len(indices)
    return output


def _within_group_shuffle_auroc(
    scores: Sequence[float],
    labels: Sequence[bool],
    groups: Sequence[str],
    *,
    observed: float,
    seed: int,
    replicates: int = 5000,
) -> tuple[float, float, float, float]:
    """AUROC null from shuffling scores inside layers, preserving layer allocation."""

    if len(scores) != len(labels) or len(scores) != len(groups) or not scores:
        raise ValueError("within-group shuffle inputs must be nonempty and aligned")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        shuffled = list(scores)
        for indices in by_group.values():
            group_scores = [shuffled[index] for index in indices]
            rng.shuffle(group_scores)
            for index, value in zip(indices, group_scores, strict=True):
                shuffled[index] = value
        estimates.append(binary_auroc(shuffled, labels))
    exceedances = sum(estimate >= observed for estimate in estimates)
    return (
        sum(estimates) / len(estimates),
        _quantile(estimates, 0.025),
        _quantile(estimates, 0.975),
        (exceedances + 1) / (replicates + 1),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, replicates: int = 2000
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = [
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    ]
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _bootstrap_binary_ci(
    scores: Sequence[float],
    labels: Sequence[bool],
    metric: Callable[[Sequence[float], Sequence[bool]], float],
    *,
    seed: int,
    replicates: int = 2000,
) -> tuple[float, float]:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_positive = [rng.choice(positives) for _ in positives]
        sampled_negative = [rng.choice(negatives) for _ in negatives]
        sampled_scores = sampled_positive + sampled_negative
        sampled_labels = [True] * len(positives) + [False] * len(negatives)
        estimates.append(metric(sampled_scores, sampled_labels))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def probe_group_deletions(
    model: CifarResNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    groups: Sequence[ChannelGroup],
    transport_scores: Mapping[str, Tensor],
    flow_scores: Mapping[str, ChannelFlowScores],
    device: torch.device,
    *,
    group_size: int,
    groups_per_score: int = 4,
    max_batches: int | None = None,
    cached_records: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, float | int | str]]:
    """Measure damage from deleting the same structured groups used by flow."""

    if group_size < 1:
        raise ValueError("group_size must be positive")
    criterion = nn.CrossEntropyLoss()
    baseline = evaluate(model, loader, criterion, device, max_batches=max_batches)
    cached_by_key: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for cached in cached_records or ():
        key = (
            str(cached["group"]),
            int(cached["channel_group_index"]),
            int(cached["deletion_group_size"]),
        )
        if key in cached_by_key:
            raise ValueError(f"duplicate cached group deletion measurement for {key}")
        cached_by_key[key] = cached

    records: list[dict[str, float | int | str]] = []
    for group in groups:
        magnitude = aggregate_channel_scores(
            channel_magnitude_scores(model, group), group_size
        )
        transport = aggregate_channel_scores(transport_scores[group.name], group_size)
        flow = flow_scores[group.name]
        marginal_flow = aggregate_channel_scores(flow.marginal_flow, group_size)
        cut_incident = aggregate_channel_scores(flow.cut_incident, group_size)
        score_vectors = (magnitude, transport, marginal_flow, cut_incident)
        if len({vector.numel() for vector in score_vectors}) != 1:
            raise ValueError(f"group score lengths do not align for {group.name}")
        selected: set[int] = set()
        for scores in score_vectors:
            selected.update(rank_stratified_channels(scores, groups_per_score))

        for group_index in sorted(selected):
            start = group_index * group_size
            end = min(start + group_size, group.channels)
            cached = cached_by_key.get((group.name, group_index, group_size))
            if cached is None:
                with ChannelRangeAblation(model, group, start, end):
                    ablated = evaluate(
                        model, loader, criterion, device, max_batches=max_batches
                    )
                ablated_loss = float(ablated["loss"])
                ablated_accuracy = float(ablated["accuracy"])
                reused = 0
            else:
                if str(cached["producer"]) != group.producer or str(cached["gate"]) != group.gate:
                    raise ValueError(
                        f"cached group metadata does not match {(group.name, group_index)}"
                    )
                if int(cached["examples"]) != int(baseline["examples"]):
                    raise ValueError("cached group deletion used a different number of examples")
                if not math.isclose(
                    float(cached["baseline_loss"]),
                    float(baseline["loss"]),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ):
                    raise ValueError("cached group deletion baseline loss does not match")
                if not math.isclose(
                    float(cached["baseline_accuracy"]),
                    float(baseline["accuracy"]),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ):
                    raise ValueError("cached group deletion baseline accuracy does not match")
                ablated_loss = float(cached["ablated_loss"])
                ablated_accuracy = float(cached["ablated_accuracy"])
                reused = 1

            records.append(
                {
                    "group": group.name,
                    "producer": group.producer,
                    "gate": group.gate,
                    "channel_group_index": group_index,
                    "channel_start": start,
                    "channel_end": end,
                    "channels_deleted": end - start,
                    "deletion_group_size": group_size,
                    "magnitude": float(magnitude[group_index]),
                    "transport": float(transport[group_index]),
                    "flow_marginal": float(marginal_flow[group_index]),
                    "cut_incident": float(cut_incident[group_index]),
                    "block_base_flow": flow.base_flow,
                    "baseline_loss": baseline["loss"],
                    "ablated_loss": ablated_loss,
                    "loss_damage": ablated_loss - baseline["loss"],
                    "baseline_accuracy": baseline["accuracy"],
                    "ablated_accuracy": ablated_accuracy,
                    "accuracy_damage": baseline["accuracy"] - ablated_accuracy,
                    "examples": int(baseline["examples"]),
                    "measurement_reused": reused,
                }
            )
    return records


def probe_channel_deletions(
    model: CifarResNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    groups: Sequence[ChannelGroup],
    transport_scores: Mapping[str, Tensor],
    flow_scores: Mapping[str, ChannelFlowScores] | None,
    device: torch.device,
    *,
    channels_per_score: int = 4,
    max_batches: int | None = None,
    cached_records: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, float | int | str]]:
    """Measure immediate selection loss after deleting stratified channels."""

    criterion = nn.CrossEntropyLoss()
    baseline = evaluate(model, loader, criterion, device, max_batches=max_batches)
    cached_by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for cached in cached_records or ():
        key = (str(cached["group"]), int(cached["channel"]))
        if key in cached_by_key:
            raise ValueError(f"duplicate cached deletion measurement for {key}")
        cached_by_key[key] = cached
    records: list[dict[str, float | int | str]] = []
    for group in groups:
        magnitude = channel_magnitude_scores(model, group)
        transport = transport_scores[group.name]
        selected = sorted(
            set(rank_stratified_channels(magnitude, channels_per_score))
            | set(rank_stratified_channels(transport, channels_per_score))
        )
        flow = None if flow_scores is None else flow_scores[group.name]
        if flow is not None:
            selected = sorted(
                set(selected)
                | set(rank_stratified_channels(flow.marginal_flow, channels_per_score))
            )
        for channel in selected:
            cached = cached_by_key.get((group.name, channel))
            if cached is None:
                with ChannelAblation(model, group, channel):
                    ablated = evaluate(
                        model, loader, criterion, device, max_batches=max_batches
                    )
                ablated_loss = float(ablated["loss"])
                ablated_accuracy = float(ablated["accuracy"])
                reused = 0
            else:
                if str(cached["producer"]) != group.producer or str(cached["gate"]) != group.gate:
                    raise ValueError(
                        f"cached deletion metadata does not match {(group.name, channel)}"
                    )
                cached_examples = int(cached["examples"])
                cached_loss = float(cached["baseline_loss"])
                cached_accuracy = float(cached["baseline_accuracy"])
                if cached_examples != int(baseline["examples"]):
                    raise ValueError("cached deletion used a different number of examples")
                if not math.isclose(cached_loss, float(baseline["loss"]), rel_tol=1e-6, abs_tol=1e-7):
                    raise ValueError("cached deletion baseline loss does not match")
                if not math.isclose(
                    cached_accuracy,
                    float(baseline["accuracy"]),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ):
                    raise ValueError("cached deletion baseline accuracy does not match")
                ablated_loss = float(cached["ablated_loss"])
                ablated_accuracy = float(cached["ablated_accuracy"])
                reused = 1
            record: dict[str, float | int | str] = {
                    "group": group.name,
                    "producer": group.producer,
                    "gate": group.gate,
                    "channel": channel,
                    "magnitude": float(magnitude[channel]),
                    "transport": float(transport[channel]),
                    "baseline_loss": baseline["loss"],
                    "ablated_loss": ablated_loss,
                    "loss_damage": ablated_loss - baseline["loss"],
                    "baseline_accuracy": baseline["accuracy"],
                    "ablated_accuracy": ablated_accuracy,
                    "accuracy_damage": baseline["accuracy"] - ablated_accuracy,
                    "examples": int(baseline["examples"]),
                    "measurement_reused": reused,
                }
            if flow is not None:
                record.update(
                    {
                        "flow_marginal": float(flow.marginal_flow[channel]),
                        "flow_group_marginal": float(flow.group_marginal_flow[channel]),
                        "cut_incident": float(flow.cut_incident[channel]),
                        "cut_group_incident": float(flow.group_cut_incident[channel]),
                        "block_base_flow": flow.base_flow,
                        "flow_channel_group_size": flow.channel_group_size,
                    }
                )
            records.append(record)
    return records


def write_probe_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        raise ValueError("cannot write an empty probe table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)
