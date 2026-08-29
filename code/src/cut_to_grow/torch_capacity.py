"""PyTorch hooks for activation-gradient transport capacity.

The monitor is intentionally explicit and conservative. It supports standard
``Linear`` and non-grouped ``Conv2d`` modules and stores EMA scores on CPU.
Production-scale runs will need sampled layers/positions or a fused estimator.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True)
class GateSensitivityScores:
    """Structured mask-gradient baselines accumulated at one activation gate."""

    taylor: Tensor
    group_fisher: Tensor
    mean_abs_transport: Tensor
    examples: int


class GateSensitivityMonitor:
    """Collect Taylor and Group Fisher scores for contiguous channel masks."""

    def __init__(
        self,
        model: nn.Module,
        *,
        module_names: Iterable[str],
        group_size: int,
        calculation_device: str | torch.device = "cpu",
        capacity_decay: float = 0.95,
    ) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.model = model
        self.allowed_names = set(module_names)
        self.group_size = group_size
        self.calculation_device = torch.device(calculation_device)
        if not 0.0 <= capacity_decay < 1.0:
            raise ValueError("capacity_decay must be in [0, 1)")
        self.capacity_decay = capacity_decay
        self._taylor_sums: dict[str, Tensor] = {}
        self._fisher_sums: dict[str, Tensor] = {}
        self._capacity_scores: dict[str, Tensor] = {}
        self._examples: dict[str, int] = {}
        self._conditional_capacity_sums: dict[str, dict[int, Tensor]] = {}
        self._conditional_examples: dict[str, dict[int, int]] = {}
        self._sample_groups: Tensor | None = None
        self._handles: list[Any] = []

    def set_sample_groups(self, groups: Tensor | None) -> None:
        """Set optional per-example commodity IDs for the next forward/backward."""

        if groups is not None and (groups.ndim != 1 or groups.numel() == 0):
            raise ValueError("sample groups must be a nonempty vector")
        self._sample_groups = None if groups is None else groups.detach().to("cpu").clone()

    def attach(self) -> "GateSensitivityMonitor":
        if self._handles:
            raise RuntimeError("monitor is already attached")
        found: set[str] = set()
        for name, module in self.model.named_modules():
            if name not in self.allowed_names:
                continue
            found.add(name)
            self._handles.append(module.register_forward_hook(self._forward_hook(name)))
        missing = self.allowed_names - found
        if missing:
            self.close()
            raise KeyError(f"unknown sensitivity gate modules: {sorted(missing)}")
        if not self._handles:
            raise ValueError("no sensitivity gate modules were selected")
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._sample_groups = None

    def __enter__(self) -> "GateSensitivityMonitor":
        return self.attach()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _forward_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            if not isinstance(output, Tensor) or output.ndim < 2:
                raise RuntimeError(f"gate {name!r} did not produce a channel tensor")
            if not output.requires_grad:
                raise RuntimeError(f"gate {name!r} output does not require gradients")
            activation = output.detach().clone()
            sample_groups = (
                None if self._sample_groups is None else self._sample_groups.clone()
            )
            if sample_groups is not None and sample_groups.numel() != output.shape[0]:
                raise RuntimeError("sample-group IDs do not align to the gate batch")
            output.register_hook(self._gradient_hook(name, activation, sample_groups))

        return hook

    def _gradient_hook(
        self, name: str, activation: Tensor, sample_groups: Tensor | None
    ):
        def hook(gradient: Tensor) -> None:
            activation_cpu = activation.to(self.calculation_device)
            gradient_cpu = gradient.detach().to(self.calculation_device)
            batch_size = activation_cpu.shape[0]
            sample_mask_gradient = activation_cpu * gradient_cpu * batch_size
            channel_transport = sample_mask_gradient.abs()
            if channel_transport.ndim > 2:
                channel_transport = channel_transport.flatten(start_dim=2).mean(dim=2)
            grouped_transport = torch.stack(
                [
                    channel_transport[:, start : start + self.group_size].sum(dim=1)
                    for start in range(0, channel_transport.shape[1], self.group_size)
                ],
                dim=1,
            ).to(dtype=torch.float64)
            if sample_mask_gradient.ndim > 2:
                sample_mask_gradient = sample_mask_gradient.flatten(start_dim=2).sum(dim=2)
            grouped = torch.stack(
                [
                    sample_mask_gradient[:, start : start + self.group_size].sum(dim=1)
                    for start in range(0, sample_mask_gradient.shape[1], self.group_size)
                ],
                dim=1,
            ).to(dtype=torch.float64)
            if not bool(torch.isfinite(grouped).all()) or not bool(
                torch.isfinite(grouped_transport).all()
            ):
                raise FloatingPointError(
                    f"non-finite Taylor/Fisher observation for gate {name!r}"
                )
            taylor = grouped.sum(dim=0).cpu()
            fisher = grouped.square().sum(dim=0).cpu()
            capacity = grouped_transport.mean(dim=0).cpu()
            if sample_groups is not None:
                capacity_sums = self._conditional_capacity_sums.setdefault(name, {})
                example_counts = self._conditional_examples.setdefault(name, {})
                for group in torch.unique(sample_groups).tolist():
                    group_id = int(group)
                    mask = (sample_groups == group_id).to(grouped_transport.device)
                    group_sum = grouped_transport[mask].sum(dim=0).cpu()
                    group_examples = int(mask.sum().item())
                    if group_id not in capacity_sums:
                        capacity_sums[group_id] = group_sum
                        example_counts[group_id] = group_examples
                    else:
                        capacity_sums[group_id].add_(group_sum)
                        example_counts[group_id] += group_examples
            if name not in self._taylor_sums:
                self._taylor_sums[name] = taylor
                self._fisher_sums[name] = fisher
                self._capacity_scores[name] = capacity
                self._examples[name] = batch_size
            else:
                self._taylor_sums[name].add_(taylor)
                self._fisher_sums[name].add_(fisher)
                self._capacity_scores[name].mul_(self.capacity_decay).add_(
                    capacity, alpha=1.0 - self.capacity_decay
                )
                self._examples[name] += batch_size

        return hook

    def scores(self, name: str) -> GateSensitivityScores:
        if name not in self._examples:
            raise KeyError(f"no sensitivity observation has been recorded for {name!r}")
        examples = self._examples[name]
        return GateSensitivityScores(
            taylor=self._taylor_sums[name].abs() / examples,
            group_fisher=self._fisher_sums[name] / examples,
            mean_abs_transport=self._capacity_scores[name].clone(),
            examples=examples,
        )

    def conditional_mean_abs_transport(self, name: str) -> dict[int, Tensor]:
        """Return arithmetic mean gate capacity for every observed commodity."""

        if name not in self._conditional_capacity_sums:
            raise KeyError(f"no conditional capacity observation for {name!r}")
        return {
            group: total / self._conditional_examples[name][group]
            for group, total in self._conditional_capacity_sums[name].items()
        }

    def conditional_examples(self, name: str) -> dict[int, int]:
        if name not in self._conditional_examples:
            raise KeyError(f"no conditional capacity observation for {name!r}")
        return dict(self._conditional_examples[name])


class TransportCapacityMonitor:
    """Collect ``E[|activation * weight * downstream-gradient|]`` per weight."""

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.95,
        module_names: Iterable[str] | None = None,
        max_spatial_positions: int | None = 256,
        calculation_device: str | torch.device = "cpu",
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.model = model
        self.decay = decay
        self.max_spatial_positions = max_spatial_positions
        self.calculation_device = torch.device(calculation_device)
        self.allowed_names = None if module_names is None else set(module_names)
        self.scores: dict[str, Tensor] = {}
        self.updates: dict[str, int] = {}
        self._inputs: dict[str, Tensor] = {}
        self._handles: list[Any] = []

    def attach(self) -> "TransportCapacityMonitor":
        if self._handles:
            raise RuntimeError("monitor is already attached")
        for name, module in self.model.named_modules():
            if self.allowed_names is not None and name not in self.allowed_names:
                continue
            if isinstance(module, nn.Conv2d) and module.groups != 1:
                continue
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            self._handles.append(module.register_forward_hook(self._forward_hook(name)))
            self._handles.append(
                module.register_full_backward_hook(self._backward_hook(name, module))
            )
        if not self._handles:
            raise ValueError("no supported modules were selected")
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._inputs.clear()

    def __enter__(self) -> "TransportCapacityMonitor":
        return self.attach()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _forward_hook(self, name: str):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor) -> None:
            if not inputs:
                raise RuntimeError(f"module {name!r} had no tensor input")
            self._inputs[name] = inputs[0].detach()

        return hook

    def _backward_hook(self, name: str, module: nn.Module):
        def hook(
            _module: nn.Module,
            _grad_input: tuple[Tensor | None, ...],
            grad_output: tuple[Tensor | None, ...],
        ) -> None:
            if name not in self._inputs or not grad_output or grad_output[0] is None:
                return
            activation = self._inputs.pop(name).to(self.calculation_device)
            downstream = grad_output[0].detach().to(self.calculation_device)
            # Losses use mean reduction. Undo that batch-size scale so capacity
            # does not change merely because calibration batch size changes.
            downstream = downstream * activation.shape[0]
            if isinstance(module, nn.Linear):
                score = self._linear_score(module, activation, downstream)
            elif isinstance(module, nn.Conv2d):
                score = self._conv2d_score(module, activation, downstream)
            else:  # pragma: no cover - guarded by attach
                return
            self._update(name, score)

        return hook

    @staticmethod
    def _linear_score(module: nn.Linear, activation: Tensor, downstream: Tensor) -> Tensor:
        x = activation.reshape(-1, activation.shape[-1]).abs()
        delta = downstream.reshape(-1, downstream.shape[-1]).abs()
        if x.shape[0] != delta.shape[0]:
            raise RuntimeError("Linear activation and gradient observations do not align")
        correlation = delta.transpose(0, 1).matmul(x) / max(1, x.shape[0])
        weight = module.weight.detach().to(device=correlation.device, dtype=correlation.dtype)
        return correlation * weight.abs()

    def _conv2d_score(
        self, module: nn.Conv2d, activation: Tensor, downstream: Tensor
    ) -> Tensor:
        patches = functional.unfold(
            activation,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        ).abs()
        delta = downstream.flatten(start_dim=2).abs()
        if patches.shape[0] != delta.shape[0] or patches.shape[2] != delta.shape[2]:
            raise RuntimeError("Conv2d activation patches and gradients do not align")

        positions = patches.shape[2]
        if self.max_spatial_positions is not None and positions > self.max_spatial_positions:
            sample = torch.linspace(
                0,
                positions - 1,
                steps=self.max_spatial_positions,
                device=patches.device,
            ).round().long()
            patches = patches.index_select(2, sample)
            delta = delta.index_select(2, sample)
            positions = patches.shape[2]

        correlation = torch.einsum("nol,nil->oi", delta, patches)
        correlation = correlation / max(1, activation.shape[0] * positions)
        correlation = correlation.reshape(module.weight.shape)
        weight = module.weight.detach().to(device=correlation.device, dtype=correlation.dtype)
        return correlation * weight.abs()

    def _update(self, name: str, observation: Tensor) -> None:
        observation_cpu = observation.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(observation_cpu).all()):
            raise FloatingPointError(f"non-finite transport observation for module {name!r}")
        if name not in self.scores:
            self.scores[name] = observation_cpu.clone()
            self.updates[name] = 1
            return
        self.scores[name].mul_(self.decay).add_(observation_cpu, alpha=1.0 - self.decay)
        self.updates[name] += 1

    def output_channel_scores(self, name: str) -> Tensor:
        """Aggregate a stored per-weight score into one score per output unit."""

        if name not in self.scores:
            raise KeyError(f"no score has been recorded for {name!r}")
        score = self.scores[name]
        return score.reshape(score.shape[0], -1).sum(dim=1)
