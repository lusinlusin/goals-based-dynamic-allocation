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
from functools import partial
from typing import Callable, Literal, Sequence

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


# Module-level functions bound with functools.partial (not closures), so that
# objectives and solved tables can be pickled and sent between processes.
def _identity(x: Array) -> Array:
    return x


def _reached(w: Array, goal: float) -> Array:
    return (np.asarray(w) >= goal).astype(float)


def _power_utility(w: Array, one_minus: float, scale: float) -> Array:
    return (np.asarray(w, dtype=float) / scale) ** one_minus / one_minus


def _power_inverse(v: Array, one_minus: float, scale: float) -> Array:
    return scale * (one_minus * np.asarray(v, dtype=float)) ** (1.0 / one_minus)


def _log_utility(w: Array, scale: float) -> Array:
    return np.log(np.asarray(w, dtype=float) / scale)


def _log_inverse(v: Array, scale: float) -> Array:
    return scale * np.exp(np.asarray(v, dtype=float))


def goal_objective(goal: float) -> Objective:
    """Maximise the probability of ending at or above ``goal``: ``V_T = 1{W >= goal}``."""
    return Objective(
        name="goal",
        kind="goal",
        terminal=partial(_reached, goal=goal),
        to_ce=_identity,
        from_ce=_identity,
    )


def power_objective(gamma: float, scale: float = 1.0) -> Objective:
    """CRRA utility ``U(W) = (W/scale)^(1-gamma) / (1-gamma)``, ``gamma > 0, gamma != 1``.

    ``scale`` only rescales wealth to keep the numbers well inside floating-point
    range (e.g. pass the goal amount); it does not change the optimal policy.
    """
    if gamma <= 0 or gamma == 1:
        raise ValueError("gamma must be positive and different from 1 (use log_objective for 1)")
    utility = partial(_power_utility, one_minus=1.0 - gamma, scale=scale)
    inverse = partial(_power_inverse, one_minus=1.0 - gamma, scale=scale)
    return Objective(f"power({gamma:g})", "utility", utility, inverse, utility)


def log_objective(scale: float = 1.0) -> Objective:
    """Log utility ``U(W) = ln(W/scale)`` (CRRA with ``gamma = 1``, growth-optimal)."""
    utility = partial(_log_utility, scale=scale)
    return Objective("log", "utility", utility, partial(_log_inverse, scale=scale), utility)


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


def _action_risk(port: Array, probs: Array) -> Array:
    """Standard deviation of the portfolio return for each action (used to break ties)."""
    mean = np.einsum("am,m->a", port, probs)
    return np.sqrt(np.einsum("am,m->a", (port - mean[:, None]) ** 2, probs))


def solve(
    scenarios: ReturnScenarios | Sequence[ReturnScenarios],
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
            set is used for every action (common random numbers), so actions are
            compared on identical outcomes. Either one set used in every period,
            or a sequence of ``horizon`` sets where entry ``t`` drives the step
            from ``t`` to ``t + 1`` (return assumptions that change over time,
            known in advance).
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
    if isinstance(scenarios, ReturnScenarios):
        per_period = None
        port = scenarios.portfolio(actions)  # (A, M)
        probs = scenarios.probs
        risk = _action_risk(port, probs)
    else:
        per_period = list(scenarios)
        if len(per_period) != horizon:
            raise ValueError(f"need one scenario set per period: got {len(per_period)}, horizon {horizon}")

    n = grid.size
    value = np.empty((horizon + 1, n))
    policy_idx = np.empty((horizon, n), dtype=int)
    value[horizon] = objective.terminal(grid)

    for t in range(horizon - 1, -1, -1):
        if per_period is not None:
            port = per_period[t].portfolio(actions)
            probs = per_period[t].probs
            risk = _action_risk(port, probs)
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


# --------------------------------------------------------------------------- #
# A second state variable: the market state or a belief about the market
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateStep:
    """One period's return scenarios in one exogenous state, and where the state goes next.

    The exogenous state ``s`` lives on a finite set ``0..K-1`` (a market regime,
    or a point on a grid of beliefs about the expected return). After scenario
    ``j`` the next state is ``lo[j]`` with weight ``1 - w_hi[j]`` and ``hi[j]``
    with weight ``w_hi[j]``. That covers both uses in this project:

    * a two-state Markov chain whose next state does not depend on this period's
      return: ``lo = 0``, ``hi = 1``, ``w_hi = P(s, 1)`` for every scenario;
    * a belief that is updated from this period's return and lands between two
      grid points: linear interpolation weights, which differ by scenario.

    Attributes:
        scenarios: Return outcomes in this state.
        lo, hi: Next-state indices, shape ``(M,)``.
        w_hi: Weight on ``hi``, shape ``(M,)``, in ``[0, 1]``.
    """

    scenarios: ReturnScenarios
    lo: Array
    hi: Array
    w_hi: Array


@dataclass(frozen=True)
class StateDPSolution:
    """Result of :func:`solve_with_state`.

    Attributes:
        grid: Wealth grid, shape ``(n,)``.
        actions: Allowed weights in asset A, shape ``(A,)``.
        policy: Optimal weight at each ``(t, state, grid point)``, shape ``(T, K, n)``.
        value: Success probability at each ``(t, state, grid point)``, shape ``(T + 1, K, n)``.
        horizon: Number of periods ``T``.
        contribution: Amount added to wealth at the end of each period.
    """

    grid: Array
    actions: Array
    policy: Array
    value: Array
    horizon: int
    contribution: float

    @property
    def n_states(self) -> int:
        return self.policy.shape[1]

    def grid_index(self, wealth: Array) -> Array:
        """Nearest grid point in log wealth (same rule as :class:`DPSolution`)."""
        mids = np.sqrt(self.grid[:-1] * self.grid[1:])
        return np.clip(np.searchsorted(mids, np.asarray(wealth, dtype=float), side="right"), 0, self.grid.size - 1)

    def action(self, t: int, wealth: Array, state: Array) -> Array:
        """Optimal weight at time ``t`` for each (wealth, state index) pair."""
        return self.policy[t, np.asarray(state, dtype=int), self.grid_index(wealth)]

    def value_at(self, t: int, wealth: float, state: int) -> float:
        """Success probability at time ``t`` in ``state``, linearly interpolated in wealth."""
        return float(np.interp(wealth, self.grid, self.value[t, state]))

    def in_state(self, state: int) -> DPSolution:
        """The policy table for one fixed state, as a :class:`DPSolution` (for plotting)."""
        return DPSolution(self.grid, self.actions, self.policy[:, state], self.value[:, state],
                          goal_objective(np.inf), self.horizon, self.contribution)


def _interp_rows(x: Array, xp: Array, table: Array, rows_lo: Array, rows_hi: Array, w_hi: Array) -> Array:
    """``(1 - w) * table[lo](x) + w * table[hi](x)``, linear in ``x`` and flat beyond the ends.

    ``x`` has shape ``(n, M)``; ``rows_*`` and ``w_hi`` have shape ``(M,)`` (one
    pair of rows per scenario). Same end treatment as ``np.interp``.
    """
    pos = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, xp.size - 2)
    frac = np.clip((x - xp[pos]) / (xp[pos + 1] - xp[pos]), 0.0, 1.0)

    def read(rows: Array) -> Array:
        f0 = table[rows[None, :], pos]
        return f0 + frac * (table[rows[None, :], pos + 1] - f0)

    out = read(rows_lo)
    if np.any(w_hi > 0):
        out = out + w_hi[None, :] * (read(rows_hi) - out)
    return out


def solve_with_state(
    step: Callable[[int, int], StateStep],
    n_states: int,
    actions: Array,
    grid: Array,
    goal: float,
    horizon: int,
    contribution: float = 0.0,
    tie_tol: float = 1e-12,
    zero_tol: float | None = None,
) -> StateDPSolution:
    """Goal-probability DP with state ``(t, wealth, s)``, ``s`` an exogenous state in ``0..K-1``.

    Bellman equation::

        V_t(W, s) = max_a  sum_j p_j [ (1 - w_j) V_{t+1}(W'_j, lo_j) + w_j V_{t+1}(W'_j, hi_j) ]

    with ``W'_j = W * R_a,j + c`` and ``(p_j, R_j, lo_j, hi_j, w_j)`` from
    ``step(t, s)``. With one state whose next state is itself, this is
    :func:`solve` with the goal objective; the tests check that.

    Args:
        step: ``step(t, s)`` -> scenarios and next-state weights for the step from ``t`` to ``t + 1``.
        n_states: Number of exogenous states ``K``.
        actions, grid, horizon, contribution, tie_tol, zero_tol: as in :func:`solve`.
        goal: Target wealth.
    """
    actions = np.asarray(actions, dtype=float)
    grid = np.asarray(grid, dtype=float)
    objective = goal_objective(goal)
    zero_tol = tie_tol if zero_tol is None else zero_tol
    n = grid.size
    value = np.empty((horizon + 1, n_states, n))
    policy_idx = np.empty((horizon, n_states, n), dtype=int)
    value[horizon] = objective.terminal(grid)[None, :]

    for t in range(horizon - 1, -1, -1):
        last_step = t == horizon - 1
        for s in range(n_states):
            st = step(t, s)
            port = st.scenarios.portfolio(actions)
            probs = st.scenarios.probs
            lo, hi = np.asarray(st.lo, dtype=int), np.asarray(st.hi, dtype=int)
            w_hi = np.asarray(st.w_hi, dtype=float)
            q = np.empty((actions.size, n))
            for k in range(actions.size):
                w_next = grid[:, None] * port[k][None, :] + contribution
                if last_step:
                    v = objective.terminal(w_next)
                else:
                    v = _interp_rows(w_next, grid, value[t + 1], lo, hi, w_hi)
                q[k] = np.einsum("nm,m->n", v, probs)
            np.clip(q, 0.0, 1.0, out=q)
            choice = _choose(q, _action_risk(port, probs), "goal", tie_tol, zero_tol)
            policy_idx[t, s] = choice
            value[t, s] = q[choice, np.arange(n)]

    return StateDPSolution(grid, actions, actions[policy_idx], value, horizon, contribution)
