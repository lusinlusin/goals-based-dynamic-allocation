"""Backward dynamic programming for a two-asset, multi-period allocation problem.

State: wealth ``W`` on a grid. Action: weight ``a`` in asset A (the rest goes to
asset B). Dynamics: ``W' = W * (a * R_A + (1 - a) * R_B) + c``.

Bellman equation::

    V_t(W) = max_a  E[ V_{t+1}( W * R_a + c ) ],      V_T(W) = objective.terminal(W)

The expectation is a weighted sum over a finite set of return scenarios
(:class:`ReturnScenarios`). Two exact outcomes with weight 1/2 each give the exact
binomial DP; ``M`` sampled outcomes with weight ``1/M`` give the Monte Carlo DP.
Both go through the same code path, so checks on the first carry over to the second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Objective:
    """Terminal reward plus the transform used when interpolating the value function.

    Attributes:
        name: Short label, e.g. ``"goal"``, ``"power(5)"``, ``"log"``.
        kind: ``"goal"`` (probability of reaching a target) or ``"utility"``.
        terminal: ``V_T(W)``.
        to_ce: Maps a value to the scale it is interpolated on. For utilities this
            is the certainty equivalent ``U^{-1}(V)``, which is close to linear in
            wealth (exactly linear when there are no contributions), so linear
            interpolation is accurate. For the goal objective it is the identity.
        from_ce: Inverse of ``to_ce``.
    """

    name: str
    kind: Literal["goal", "utility"]
    terminal: Callable[[Array], Array]
    to_ce: Callable[[Array], Array]
    from_ce: Callable[[Array], Array]


def goal_objective(goal: float) -> Objective:
    """Maximise the probability of ending at or above ``goal``: ``V_T = 1{W >= goal}``."""

    def identity(x: Array) -> Array:
        return x

    return Objective(
        name="goal",
        kind="goal",
        terminal=lambda w: (np.asarray(w) >= goal).astype(float),
        to_ce=identity,
        from_ce=identity,
    )


def power_objective(gamma: float, scale: float = 1.0) -> Objective:
    """CRRA utility ``U(W) = (W/scale)^(1-gamma) / (1-gamma)``, ``gamma > 0, gamma != 1``.

    ``scale`` only rescales wealth to keep the numbers well inside floating-point
    range (e.g. pass the goal amount); it does not change the optimal policy.
    """
    if gamma <= 0 or gamma == 1:
        raise ValueError("gamma must be positive and different from 1 (use log_objective for 1)")
    one_minus = 1.0 - gamma

    def utility(w: Array) -> Array:
        return (np.asarray(w, dtype=float) / scale) ** one_minus / one_minus

    def inverse(v: Array) -> Array:
        return scale * (one_minus * np.asarray(v, dtype=float)) ** (1.0 / one_minus)

    return Objective(f"power({gamma:g})", "utility", utility, inverse, utility)


def log_objective(scale: float = 1.0) -> Objective:
    """Log utility ``U(W) = ln(W/scale)`` (CRRA with ``gamma = 1``, growth-optimal)."""

    def utility(w: Array) -> Array:
        return np.log(np.asarray(w, dtype=float) / scale)

    def inverse(v: Array) -> Array:
        return scale * np.exp(np.asarray(v, dtype=float))

    return Objective("log", "utility", utility, inverse, utility)


# --------------------------------------------------------------------------- #
# Scenarios, grid, solution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReturnScenarios:
    """A finite set of one-period gross-return outcomes for the two assets.

    Attributes:
        r_a: Gross returns of asset A (the one the action weight applies to), shape ``(M,)``.
        r_b: Gross returns of asset B, shape ``(M,)``.
        probs: Scenario probabilities, shape ``(M,)``, summing to 1.
    """

    r_a: Array
    r_b: Array
    probs: Array

    @staticmethod
    def equally_weighted(r_a: Array, r_b: Array) -> "ReturnScenarios":
        """Scenarios from a Monte Carlo sample: each draw gets weight ``1/M``."""
        r_a = np.asarray(r_a, dtype=float).ravel()
        r_b = np.asarray(r_b, dtype=float).ravel()
        return ReturnScenarios(r_a, r_b, np.full(r_a.size, 1.0 / r_a.size))

    def portfolio(self, actions: Array) -> Array:
        """Gross portfolio return for every action and scenario, shape ``(A, M)``."""
        a = np.asarray(actions, dtype=float)[:, None]
        return a * self.r_a[None, :] + (1.0 - a) * self.r_b[None, :]


def log_grid(low: float, high: float, n: int, include: float | None = None) -> Array:
    """Log-spaced wealth grid (denser at low wealth), optionally forcing one point in."""
    grid = np.geomspace(low, high, n)
    if include is not None and low < include < high:
        grid = np.unique(np.append(grid, include))
    return grid


@dataclass(frozen=True)
class DPSolution:
    """Result of :func:`solve`.

    Attributes:
        grid: Wealth grid, shape ``(n,)``.
        actions: Allowed weights in asset A, shape ``(A,)``.
        policy: Optimal weight at each ``(t, grid point)``, shape ``(T, n)``.
        value: Value function at each ``(t, grid point)``, shape ``(T + 1, n)``.
        objective: The objective that was optimised.
        horizon: Number of periods ``T``.
        contribution: Amount added to wealth at the end of each period.
        lookup: How wealth between grid points is mapped to a grid point when the
            policy is applied (``"nearest"`` in log-wealth, or ``"floor"``).
    """

    grid: Array
    actions: Array
    policy: Array
    value: Array
    objective: Objective
    horizon: int
    contribution: float
    lookup: Literal["nearest", "floor"] = "nearest"

    def grid_index(self, wealth: Array) -> Array:
        """Index of the grid point used for each wealth level."""
        w = np.asarray(wealth, dtype=float)
        if self.lookup == "floor":
            idx = np.searchsorted(self.grid, w, side="right") - 1
        else:
            mids = np.sqrt(self.grid[:-1] * self.grid[1:])
            idx = np.searchsorted(mids, w, side="right")
        return np.clip(idx, 0, self.grid.size - 1)

    def action(self, t: int, wealth: Array) -> Array:
        """Optimal weight in asset A at time ``t`` for the given wealth level(s)."""
        return self.policy[t][self.grid_index(wealth)]

    def value_at(self, t: int, wealth: float) -> float:
        """Value function at time ``t``, linearly interpolated in wealth."""
        return float(np.interp(wealth, self.grid, self.value[t]))


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def _interp_linear_extrap(x: Array, xp: Array, fp: Array) -> Array:
    """Linear interpolation with linear extrapolation beyond both ends."""
    y = np.interp(x, xp, fp)
    lo_slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
    hi_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    below, above = x < xp[0], x > xp[-1]
    if below.any():
        y[below] = fp[0] + lo_slope * (x[below] - xp[0])
    if above.any():
        y[above] = fp[-1] + hi_slope * (x[above] - xp[-1])
    return y


def _choose(q: Array, risk: Array, kind: str, tie_tol: float, zero_tol: float) -> Array:
    """Pick the best action per state from ``q`` (shape ``(A, n)``).

    Utilities: plain argmax. Goal probability: many actions can give exactly the
    same probability (0 far below the goal, 1 far above it, and plateaus when
    outcomes are discrete). Ties are broken on purpose: take the least risky
    action that achieves the best probability. Where even the best probability
    is at most ``zero_tol`` the goal is out of reach and the riskiest action is
    taken, which is the limit of the optimal policy as wealth falls.
    """
    best = q.max(axis=0)
    if kind != "goal":
        return q.argmax(axis=0)
    tied = q >= best[None, :] - tie_tol
    safest = np.where(tied, risk[:, None], np.inf).argmin(axis=0)
    return np.where(best <= zero_tol, int(risk.argmax()), safest)


def solve(
    scenarios: ReturnScenarios,
    actions: Array,
    grid: Array,
    objective: Objective,
    horizon: int,
    contribution: float = 0.0,
    interp: Literal["linear", "floor", "ceil"] = "linear",
    tie_tol: float = 1e-12,
    zero_tol: float | None = None,
) -> DPSolution:
    """Solve the allocation problem by backward induction.

    Args:
        scenarios: One-period return outcomes and their probabilities. The same
            set is used for every action and every period (common random numbers),
            so actions are compared on identical outcomes.
        actions: Allowed weights in asset A.
        grid: Increasing wealth grid.
        objective: Terminal reward (goal probability, power or log utility).
        horizon: Number of periods ``T``.
        contribution: Cash added at the end of each period (after returns).
        interp: How ``V_{t+1}`` is read between grid points. ``"linear"`` suits
            smooth value functions (continuous return distributions). ``"floor"``
            reads the grid point at or below; for the goal objective this never
            overstates the probability, which matters when outcomes are discrete
            and the value function is a staircase. ``"ceil"`` reads the grid point
            at or above and never understates it; together they bracket the answer.
        tie_tol: Two goal probabilities closer than this count as tied.
        zero_tol: Goal probabilities at or below this count as "out of reach"
            (defaults to ``tie_tol``). With sampled scenarios a small positive
            value keeps sampling noise from deciding the action in states where
            success is practically impossible.

    Returns:
        A :class:`DPSolution` with the policy table and value function.
    """
    actions = np.asarray(actions, dtype=float)
    grid = np.asarray(grid, dtype=float)
    port = scenarios.portfolio(actions)  # (A, M)
    probs = scenarios.probs
    mean = np.einsum("am,m->a", port, probs)
    risk = np.sqrt(np.einsum("am,m->a", (port - mean[:, None]) ** 2, probs))

    n = grid.size
    value = np.empty((horizon + 1, n))
    policy_idx = np.empty((horizon, n), dtype=int)
    value[horizon] = objective.terminal(grid)

    for t in range(horizon - 1, -1, -1):
        last_step = t == horizon - 1
        if not last_step:
            ce_next = objective.to_ce(value[t + 1])
        q = np.empty((actions.size, n))
        for k in range(actions.size):
            w_next = grid[:, None] * port[k][None, :] + contribution  # (n, M)
            if last_step:
                # The terminal reward is known in closed form: no interpolation.
                v = objective.terminal(w_next)
            elif objective.kind == "goal":
                flat = w_next.ravel()
                if interp == "floor":
                    idx = np.clip(np.searchsorted(grid, flat, side="right") - 1, 0, n - 1)
                    v = np.where(flat < grid[0], 0.0, ce_next[idx]).reshape(w_next.shape)
                elif interp == "ceil":
                    idx = np.clip(np.searchsorted(grid, flat, side="left"), 0, n - 1)
                    v = np.where(flat > grid[-1], 1.0, ce_next[idx]).reshape(w_next.shape)
                else:
                    v = np.interp(w_next.ravel(), grid, ce_next).reshape(w_next.shape)
            else:
                ce = _interp_linear_extrap(w_next.ravel(), grid, ce_next)
                v = objective.from_ce(np.maximum(ce, 1e-12)).reshape(w_next.shape)
            # einsum, not ``v @ probs``: the BLAS call spins up threads that fight
            # with process-level parallelism and is slower for this shape.
            q[k] = np.einsum("nm,m->n", v, probs)
        if objective.kind == "goal":
            np.clip(q, 0.0, 1.0, out=q)  # a probability; removes round-off such as 1 + 1e-16
        choice = _choose(q, risk, objective.kind, tie_tol, tie_tol if zero_tol is None else zero_tol)
        policy_idx[t] = choice
        value[t] = q[choice, np.arange(n)]

    return DPSolution(
        grid=grid,
        actions=actions,
        policy=actions[policy_idx],
        value=value,
        objective=objective,
        horizon=horizon,
        contribution=contribution,
        lookup="floor" if (objective.kind == "goal" and interp == "floor") else "nearest",
    )
