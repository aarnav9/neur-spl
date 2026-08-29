"""Joint structured masks derived from Fisher-operator sparsification."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import Tensor, nn

from .model import ChannelGroup


def topk_mask(scores: Tensor, keep_count: int) -> Tensor:
    """Return a deterministic exact-budget mask retaining the largest scores."""

    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty vector")
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("scores contain non-finite values")
    if not 0 <= keep_count <= scores.numel():
        raise ValueError("keep_count is outside the score vector")
    order = sorted(
        range(scores.numel()),
        key=lambda index: (float(scores[index]), -index),
        reverse=True,
    )
    mask = torch.zeros(scores.numel(), dtype=torch.bool)
    if keep_count:
        mask[order[:keep_count]] = True
    return mask


def leverage_inclusion_probabilities(scores: Tensor, keep_count: int) -> Tensor:
    """Choose ``q_g = min(1, c * score_g)`` with probabilities summing to budget."""

    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty vector")
    values = scores.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("leverage scores must be finite and nonnegative")
    count = values.numel()
    if not 0 <= keep_count <= count:
        raise ValueError("keep_count is outside the score vector")
    if keep_count == 0:
        return torch.zeros_like(values)
    if keep_count == count:
        return torch.ones_like(values)

    positive = values > 0
    positive_count = int(positive.sum())
    if positive_count == 0:
        return torch.full_like(values, keep_count / count)
    if keep_count > positive_count:
        probabilities = positive.to(dtype=torch.float64)
        probabilities[~positive] = (keep_count - positive_count) / (count - positive_count)
        return probabilities

    low = 0.0
    high = 1.0 / float(values[positive].min())
    for _ in range(100):
        middle = (low + high) / 2.0
        total = float(torch.clamp(values * middle, max=1.0).sum())
        if total < keep_count:
            low = middle
        else:
            high = middle
    probabilities = torch.clamp(values * high, max=1.0)
    # Remove insignificant bisection drift while keeping every probability in
    # range. The dependent rounder needs an integer total to machine precision.
    difference = keep_count - float(probabilities.sum())
    fractional = (probabilities > 0.0) & (probabilities < 1.0)
    if bool(fractional.any()):
        index = int(torch.nonzero(fractional, as_tuple=False)[0])
        probabilities[index] += difference
    if abs(float(probabilities.sum()) - keep_count) > 1e-9:
        raise RuntimeError("could not construct exact-budget inclusion probabilities")
    return probabilities


def dependent_round(probabilities: Tensor, *, seed: int) -> Tensor:
    """Exact-cardinality dependent rounding preserving marginal probabilities."""

    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a nonempty vector")
    values = probabilities.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool(
        (values > 1).any()
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    expected = round(float(values.sum()))
    if abs(float(values.sum()) - expected) > 1e-8:
        raise ValueError("probabilities must sum to an integer")

    rng = random.Random(seed)
    tolerance = 1e-12
    while True:
        fractional = [
            index for index, value in enumerate(values) if tolerance < value < 1.0 - tolerance
        ]
        if len(fractional) < 2:
            break
        left, right = fractional[:2]
        alpha = min(1.0 - float(values[left]), float(values[right]))
        beta = min(float(values[left]), 1.0 - float(values[right]))
        if rng.random() < beta / (alpha + beta):
            values[left] += alpha
            values[right] -= alpha
        else:
            values[left] -= beta
            values[right] += beta
        values[left] = values[left].clamp(0.0, 1.0)
        values[right] = values[right].clamp(0.0, 1.0)

    # At most one fractional value can remain, and the integer-sum invariant
    # determines its endpoint.
    rounded = values >= 0.5
    difference = expected - int(rounded.sum())
    if difference:
        candidates = sorted(
            range(values.numel()),
            key=lambda index: float(values[index]),
            reverse=difference > 0,
        )
        for index in candidates:
            if difference > 0 and not bool(rounded[index]):
                rounded[index] = True
                difference -= 1
            elif difference < 0 and bool(rounded[index]):
                rounded[index] = False
                difference += 1
            if difference == 0:
                break
    if int(rounded.sum()) != expected:
        raise RuntimeError("dependent rounding did not preserve the exact budget")
    return rounded


def sampled_rescaled_mask(
    leverage: Tensor, *, keep_count: int, seed: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample an exact-budget PSD mask and return activation rescaling factors."""

    probabilities = leverage_inclusion_probabilities(leverage, keep_count)
    retained = dependent_round(probabilities, seed=seed)
    scales = torch.zeros_like(probabilities)
    selected = retained & (probabilities > 0)
    scales[selected] = probabilities[selected].rsqrt()
    if not bool(torch.isfinite(scales).all()):
        raise FloatingPointError("spectral mask produced non-finite rescaling")
    return retained, scales, probabilities


def group_channel_scales(
    groups: Sequence[ChannelGroup],
    *,
    group_size: int,
    values: Mapping[tuple[str, int], float],
) -> dict[str, Tensor]:
    """Expand one value per structured group into per-channel gate scales."""

    if group_size < 1:
        raise ValueError("group_size must be positive")
    output: dict[str, Tensor] = {}
    used: set[tuple[str, int]] = set()
    for group in groups:
        channel_values = torch.empty(group.channels, dtype=torch.float32)
        for group_index, start in enumerate(range(0, group.channels, group_size)):
            key = (group.name, group_index)
            if key not in values:
                raise KeyError(f"missing structured mask value for {key}")
            channel_values[start : start + group_size] = float(values[key])
            used.add(key)
        output[group.gate] = channel_values
    unused = set(values) - used
    if unused:
        raise KeyError(f"mask contains unknown structured groups: {sorted(unused)}")
    return output


class JointChannelMask(AbstractContextManager["JointChannelMask"]):
    """Temporarily multiply several activation gates by fixed channel scales."""

    def __init__(self, model: nn.Module, gate_scales: Mapping[str, Tensor]) -> None:
        modules = dict(model.named_modules())
        missing = set(gate_scales) - set(modules)
        if missing:
            raise KeyError(f"unknown mask gates: {sorted(missing)}")
        self.modules = {name: modules[name] for name in gate_scales}
        self.gate_scales = {
            name: scale.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, scale in gate_scales.items()
        }
        self.handles: list[Any] = []

    def __enter__(self) -> "JointChannelMask":
        if self.handles:
            raise RuntimeError("joint mask is already active")
        for name, module in self.modules.items():
            scale = self.gate_scales[name]

            def apply_scale(
                _module: nn.Module,
                _inputs: tuple[Tensor, ...],
                output: Tensor,
                *,
                gate_name: str = name,
                channel_scale: Tensor = scale,
            ) -> Tensor:
                if not isinstance(output, Tensor) or output.ndim < 2:
                    raise RuntimeError(f"mask gate {gate_name!r} has no channel axis")
                if output.shape[1] != channel_scale.numel():
                    raise RuntimeError(f"mask scale does not align to gate {gate_name!r}")
                shape = [1, channel_scale.numel()] + [1] * (output.ndim - 2)
                return output * channel_scale.to(
                    device=output.device, dtype=output.dtype
                ).reshape(shape)

            self.handles.append(module.register_forward_hook(apply_scale))
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.no_grad()
def recalibrate_batch_norm(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> int:
    """Replace running BN statistics using the supplied fixed calibration batches."""

    normalizations = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    if not normalizations:
        raise ValueError("model has no BatchNorm2d modules")
    for module in normalizations:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    examples = 0
    for inputs, _targets in batches:
        model(inputs.to(device))
        examples += inputs.shape[0]
    if examples == 0:
        raise ValueError("BN calibration batches are empty")
    model.eval()
    return examples


def ridge_relative_operator_error(
    gradient_matrix: Tensor,
    activation_scales: Tensor,
    *,
    ridge: float,
) -> float:
    """Return ridge-whitened spectral error of a scaled structured subset."""

    if gradient_matrix.ndim != 2 or gradient_matrix.shape[0] < 1:
        raise ValueError("gradient_matrix must be a nonempty matrix")
    if activation_scales.ndim != 1 or activation_scales.numel() != gradient_matrix.shape[1]:
        raise ValueError("activation scales do not align to gradient columns")
    if ridge <= 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be positive and finite")
    matrix = gradient_matrix.to(device="cpu", dtype=torch.float64) / math.sqrt(
        gradient_matrix.shape[0]
    )
    scales = activation_scales.to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(scales).all()):
        raise FloatingPointError("operator-error inputs contain non-finite values")

    _left, singular, right_transpose = torch.linalg.svd(matrix, full_matrices=False)
    whitened = singular / torch.sqrt(singular.square() + ridge)
    coefficient_error = scales.square() - 1.0
    reduced = right_transpose.matmul(
        coefficient_error[:, None] * right_transpose.transpose(0, 1)
    )
    reduced = whitened[:, None] * reduced * whitened[None, :]
    eigenvalues = torch.linalg.eigvalsh(reduced)
    error = float(eigenvalues.abs().max()) if eigenvalues.numel() else 0.0
    if not math.isfinite(error):
        raise FloatingPointError("operator error is non-finite")
    return error


def operator_trace_ratio(gradient_matrix: Tensor, activation_scales: Tensor) -> float:
    """Return the retained/rescaled Fisher-operator trace ratio."""

    if gradient_matrix.ndim != 2 or gradient_matrix.shape[0] < 1:
        raise ValueError("gradient_matrix must be a nonempty matrix")
    if activation_scales.ndim != 1 or activation_scales.numel() != gradient_matrix.shape[1]:
        raise ValueError("activation scales do not align to gradient columns")
    energy = gradient_matrix.to(dtype=torch.float64).square().sum(dim=0)
    denominator = float(energy.sum())
    if denominator == 0.0:
        return 1.0
    numerator = float((energy * activation_scales.to(dtype=torch.float64).square()).sum())
    return numerator / denominator
