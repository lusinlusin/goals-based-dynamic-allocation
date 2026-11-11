"""Forward Monte Carlo: apply a policy along simulated return paths.

A policy is any callable ``policy(t, wealth) -> weight in asset A`` that works on
arrays. Fixed mixes, glide paths and DP policy tables all fit that shape, so they
are evaluated by the same loop on the same return paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dp import DPSolution

Array = np.ndarray
Policy = Callable[[int, Array], Array]


def fixed_mix(weight: float) -> Policy:
    """Constant weight in asset A, rebalanced every period."""

    def policy(t: int, wealth: Array) -> Array:
        return np.full(np.shape(wealth), float(weight))

    return policy


def glide_path(start: float, end: float, horizon: int) -> Policy:
    """Weight that moves linearly from ``start`` (first period) to ``end`` (last period)."""
    path = np.linspace(start, end, horizon)

    def policy(t: int, wealth: Array) -> Array:
        return np.full(np.shape(wealth), path[t])

    return policy


def dp_policy(solution: DPSolution) -> Policy:
    """Policy that looks up the DP table at the current time and wealth."""

    def policy(t: int, wealth: Array) -> Array:
        return solution.action(t, wealth)

    return policy


@dataclass(frozen=True)
class SimResult:
    """Simulated paths.

    Attributes:
        wealth: Wealth at times ``0..T``, shape ``(N, T + 1)``.
        weights: Weight in asset A chosen at times ``0..T-1``, shape ``(N, T)``.
    """

    wealth: Array
    weights: Array

    @property
    def terminal(self) -> Array:
        """Terminal wealth ``W_T`` for each path."""
        return self.wealth[:, -1]


def simulate(
    policy: Policy,
    r_a: Array,
    r_b: Array,
    w0: float,
    contribution: float = 0.0,
) -> SimResult:
    """Run a policy forward along given return paths.

    Args:
        policy: ``policy(t, wealth) -> weight in asset A``.
        r_a: Gross returns of asset A, shape ``(N, T)``.
        r_b: Gross returns of asset B, shape ``(N, T)``.
        w0: Starting wealth.
        contribution: Cash added at the end of each period (after returns).

    Returns:
        Wealth and weight paths.
    """
    n_paths, horizon = r_a.shape
    wealth = np.empty((n_paths, horizon + 1))
    weights = np.empty((n_paths, horizon))
    wealth[:, 0] = w0
    for t in range(horizon):
        a = policy(t, wealth[:, t])
        weights[:, t] = a
        wealth[:, t + 1] = wealth[:, t] * (a * r_a[:, t] + (1.0 - a) * r_b[:, t]) + contribution
    return SimResult(wealth, weights)
