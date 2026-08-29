"""Task-conditioned PSD capacities for structured pruning.

For a structured gate group ``g``, let ``v_g`` contain the signed per-example
loss derivatives with respect to that group's multiplicative mask.  The
rank-one matrix ``C_g = v_g v_g^T / n`` is a positive-semidefinite capacity in
the shared calibration-example space.  Its trace is the usual empirical Group
Fisher score.  Its ridge leverage against ``sum_g C_g`` additionally measures
whether the group's task response points in a direction not already supplied
by other groups.

The implementation never materializes the example-space PSD matrices.  It
works with their Gram matrix, whose dimension is the number of structured
groups rather than the number of examples.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .model import ChannelGroup, CifarResNet


@dataclass(frozen=True)
class SpectralFisherScores:
    """Scores for one structured group in the global Fisher operator."""

    fisher_trace: float
    ridge_leverage: float
    ridge: float
    examples: int


class GateGradientMatrixMonitor:
    """Retain signed per-example gradients of contiguous gate masks."""

    def __init__(
        self,
        model: nn.Module,
        *,
        module_names: Iterable[str],
        group_size: int,
        calculation_device: str | torch.device = "cpu",
    ) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.model = model
        self.allowed_names = set(module_names)
        self.group_size = group_size
        self.calculation_device = torch.device(calculation_device)
        self._rows: dict[str, list[Tensor]] = {}
        self._handles: list[Any] = []

    def attach(self) -> "GateGradientMatrixMonitor":
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
            raise KeyError(f"unknown gradient gate modules: {sorted(missing)}")
        if not self._handles:
            raise ValueError("no gradient gate modules were selected")
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GateGradientMatrixMonitor":
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
            output.register_hook(self._gradient_hook(name, activation))

        return hook

    def _gradient_hook(self, name: str, activation: Tensor):
        def hook(gradient: Tensor) -> None:
            activation_cpu = activation.to(self.calculation_device)
            gradient_cpu = gradient.detach().to(self.calculation_device)
            batch_size = activation_cpu.shape[0]

            # CrossEntropyLoss uses mean reduction in the collector.  Undo the
            # batch scale to recover one mask derivative per example.
            per_channel = activation_cpu * gradient_cpu * batch_size
            if per_channel.ndim > 2:
                per_channel = per_channel.flatten(start_dim=2).sum(dim=2)
            grouped = torch.stack(
                [
                    per_channel[:, start : start + self.group_size].sum(dim=1)
                    for start in range(0, per_channel.shape[1], self.group_size)
                ],
                dim=1,
            ).detach().to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(grouped).all()):
                raise FloatingPointError(
                    f"non-finite per-example mask gradient for gate {name!r}"
                )
            self._rows.setdefault(name, []).append(grouped)

        return hook

    def matrix(self, name: str) -> Tensor:
        """Return ``examples x structured-groups`` signed mask gradients."""

        if name not in self._rows:
            raise KeyError(f"no gradient observation has been recorded for {name!r}")
        return torch.cat(self._rows[name], dim=0)


def collect_gate_gradient_matrices(
    model: CifarResNet,
    loader: Iterable[tuple[Tensor, Tensor]],
    groups: Sequence[ChannelGroup],
    device: torch.device,
    *,
    group_size: int,
    max_batches: int = 8,
) -> dict[str, Tensor]:
    """Collect signed per-example structured-mask gradients in one backward per batch."""

    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    gate_names = [group.gate for group in groups]
    if len(set(gate_names)) != len(gate_names):
        raise ValueError("groups must have distinct gate names")

    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss()
    monitor = GateGradientMatrixMonitor(
        model,
        module_names=gate_names,
        group_size=group_size,
        calculation_device="cpu",
    )
    try:
        monitor.attach()
        observed_batches = 0
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite calibration loss at batch {batch_index}"
                )
            loss.backward()
            observed_batches += 1
        if observed_batches == 0:
            raise RuntimeError("calibration loader yielded no batches")
    finally:
        monitor.close()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    matrices = {group.name: monitor.matrix(group.gate) for group in groups}
    example_counts = {matrix.shape[0] for matrix in matrices.values()}
    if len(example_counts) != 1:
        raise RuntimeError(f"gates saw inconsistent example counts: {sorted(example_counts)}")
    return matrices


def ridge_leverage_scores(
    gradient_matrix: Tensor,
    *,
    ridge: float | None = None,
) -> tuple[Tensor, Tensor, float]:
    """Return Fisher traces and ridge leverage for columns of a gradient matrix.

    If ``ridge`` is omitted, it is fixed without labels to the mean diagonal of
    the empirical Gram matrix.  This makes the default invariant to a global
    rescaling of the loss while avoiding the all-ones leverage limit when the
    finite calibration matrix has full column rank.
    """

    if gradient_matrix.ndim != 2 or gradient_matrix.shape[0] < 1:
        raise ValueError("gradient_matrix must be a nonempty matrix")
    matrix = gradient_matrix.to(dtype=torch.float64)
    if not bool(torch.isfinite(matrix).all()):
        raise FloatingPointError("gradient_matrix contains non-finite values")
    gram = matrix.transpose(0, 1).matmul(matrix) / matrix.shape[0]
    fisher_trace = torch.diagonal(gram).clone()
    if ridge is None:
        ridge = float(fisher_trace.mean())
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    if not bool(torch.isfinite(torch.tensor(ridge))):
        raise ValueError("ridge must be finite")
    if float(fisher_trace.max()) == 0.0:
        return fisher_trace, torch.zeros_like(fisher_trace), float(ridge)
    if ridge == 0.0:
        raise ValueError("ridge must be positive for a nonzero matrix")

    regularized = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    leverage_matrix = torch.linalg.solve(regularized, gram)
    leverage = torch.diagonal(leverage_matrix).clamp(min=0.0, max=1.0)
    if not bool(torch.isfinite(leverage).all()):
        raise FloatingPointError("ridge leverage solve produced non-finite values")
    return fisher_trace, leverage, float(ridge)


def model_spectral_fisher_scores(
    gradient_matrices: Mapping[str, Tensor],
    groups: Sequence[ChannelGroup],
    *,
    group_size: int,
    ridge: float | None = None,
) -> dict[tuple[str, int], SpectralFisherScores]:
    """Score all model groups in one shared empirical Fisher operator."""

    if group_size < 1:
        raise ValueError("group_size must be positive")
    columns: list[Tensor] = []
    keys: list[tuple[str, int]] = []
    examples: int | None = None
    for group in groups:
        if group.name not in gradient_matrices:
            raise KeyError(f"missing gradient matrix for group {group.name!r}")
        matrix = gradient_matrices[group.name]
        if matrix.ndim != 2:
            raise ValueError(f"gradient matrix for {group.name!r} is not two-dimensional")
        expected_columns = (group.channels + group_size - 1) // group_size
        if matrix.shape[1] != expected_columns:
            raise ValueError(
                f"gradient matrix for {group.name!r} has {matrix.shape[1]} columns; "
                f"expected {expected_columns}"
            )
        if examples is None:
            examples = matrix.shape[0]
        elif matrix.shape[0] != examples:
            raise ValueError("gradient matrices do not share an example axis")
        for group_index in range(matrix.shape[1]):
            columns.append(matrix[:, group_index])
            keys.append((group.name, group_index))
    if not columns or examples is None:
        raise ValueError("no structured groups were supplied")

    global_matrix = torch.stack(columns, dim=1)
    fisher, leverage, used_ridge = ridge_leverage_scores(global_matrix, ridge=ridge)
    return {
        key: SpectralFisherScores(
            fisher_trace=float(fisher[index]),
            ridge_leverage=float(leverage[index]),
            ridge=used_ridge,
            examples=examples,
        )
        for index, key in enumerate(keys)
    }
