"""Capacity estimators independent of the neural-network framework.

These helpers intentionally operate on scalar iterables so their definitions
can be tested without PyTorch. Tensor instrumentation lives in
``torch_capacity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, sqrt
from typing import Iterable


def _triples(
    activations: Iterable[float],
    weights: Iterable[float],
    downstream_gradients: Iterable[float],
) -> list[tuple[float, float, float]]:
    a = list(activations)
    w = list(weights)
    g = list(downstream_gradients)
    if not a:
        raise ValueError("capacity requires at least one observation")
    if not (len(a) == len(w) == len(g)):
        raise ValueError("activation, weight, and gradient lengths must match")
    return list(zip(a, w, g, strict=True))


def mean_abs_transport(
    activations: Iterable[float],
    weights: Iterable[float],
    downstream_gradients: Iterable[float],
) -> float:
    """Return ``mean(abs(a * w * delta))`` as a nonnegative capacity proxy."""

    values = _triples(activations, weights, downstream_gradients)
    return fsum(abs(a * w * g) for a, w, g in values) / len(values)


def mean_squared_transport(
    activations: Iterable[float],
    weights: Iterable[float],
    downstream_gradients: Iterable[float],
) -> float:
    """Return RMS task transport, a Fisher-energy-like capacity proxy."""

    values = _triples(activations, weights, downstream_gradients)
    return sqrt(fsum((a * w * g) ** 2 for a, w, g in values) / len(values))


@dataclass
class ExponentialMovingAverage:
    """Numerically simple scalar EMA used by online capacity estimators."""

    decay: float = 0.95
    value: float | None = None
    updates: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1)")

    def update(self, observation: float) -> float:
        observation = float(observation)
        if observation < 0:
            raise ValueError("capacity observations must be nonnegative")
        if self.value is None:
            self.value = observation
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * observation
        self.updates += 1
        return self.value

    def bias_corrected(self) -> float:
        if self.value is None or self.updates == 0:
            raise ValueError("EMA has no observations")
        # The first observation initializes the running value directly, so the
        # stored estimate is already usable and does not use zero initialization.
        return self.value

