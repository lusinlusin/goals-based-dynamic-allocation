"""Adjust the saving every year, not just the allocation.

Section 6 found that re-solving barely changes the outcome; what it changes is
the plan's forecast, and acting on a worse forecast means saving more. This
module turns that into a rule and prices it.

**Annual review.** At the start of each year, with current wealth and the years
left, look up the success probability the plan reports for each saving level
on a grid ($0 to $40,000 a year). Each level assumes that amount is saved every
remaining year, with the allocation re-optimised for it. Save the smallest
amount whose reported probability reaches a target (linear interpolation
between grid levels; the top of the grid is a cap). Invest according to the
policy for the grid level just above. The rule runs again next year, so saving
goes up after bad years and down after good ones.

**Tables.** One DP solve per saving level. Row ``t`` of a 20-year solve is the
table for ``20 - t`` years left, so one solve per level covers every review
date. For a fixed mix (60/40) the "DP" has a single action and the table is
just that mix's success probability from each (year, wealth).

The comparisons hold the success probability fixed (75%, as in the saving curve
of section 1) and ask how much saving each approach needs on average.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from . import analysis, dp, market, metrics, resolve
from .analysis import Investor, World
from .market import Assumptions

Array = np.ndarray

#: Saving levels the review can choose between (per year). The top level is the cap.
SAVING_GRID = np.arange(0.0, 40_001.0, 2_500.0)
TARGET = 0.75


@dataclass(frozen=True)
class SavingTables:
    """Success probability and stock weight for every saving level, year and wealth.

    Attributes:
        levels: Saving per year, shape ``(K,)``, increasing.
        grid: Wealth grid, shape ``(n,)``.
        value: Reported success probability, shape ``(K, T + 1, n)``.
        policy: Stock weight, shape ``(K, T, n)``.
    """

    levels: Array
    grid: Array
    value: Array
    policy: Array

    def at(self, saving: float) -> "SavingTables":
        """The table for one level only (a fixed-saving plan)."""
        k = int(np.flatnonzero(np.isclose(self.levels, saving))[0])
        return SavingTables(self.levels[k:k + 1], self.grid, self.value[k:k + 1], self.policy[k:k + 1])


def _solve_level(investor: Investor, assumptions: Assumptions, saving: float, mix: float | None) -> tuple[Array, Array]:
    """Value and policy of the goal DP with this saving (``mix``: a single fixed stock weight)."""
    inv = replace(investor, contribution=float(saving))
    world = World(assumptions)
    if mix is None:
        sol = analysis.solve_policy(inv, world)
    else:
        sol = dp.solve(world.scenarios(), np.array([mix]), analysis.wealth_grid(inv), dp.goal_objective(inv.goal),
                       inv.horizon, inv.contribution, zero_tol=analysis.GOAL_ZERO_TOL)
    return sol.value, sol.policy


def build_tables(investor: Investor, jobs: Sequence[tuple[Assumptions, Sequence[float], float | None]],
                 parallel: bool = True) -> list[SavingTables]:
    """One :class:`SavingTables` per ``(assumptions, saving levels, mix)``, solved together."""
    flat = [(investor, a, float(c), mix) for a, levels, mix in jobs for c in levels]
    out = analysis.run_parallel(_solve_level, flat) if parallel else [_solve_level(*j) for j in flat]
    grid = analysis.wealth_grid(investor)
    tables, i = [], 0
    for _, levels, _ in jobs:
        chunk = out[i:i + len(levels)]
        i += len(levels)
        tables.append(SavingTables(np.asarray(levels, dtype=float), grid,
                                   np.stack([v for v, _ in chunk]), np.stack([p for _, p in chunk])))
    return tables


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def choose_saving(estimates: Array, levels: Array, target: float) -> tuple[Array, Array]:
    """Smallest saving whose reported success reaches ``target``, for each path.

    Args:
        estimates: Reported success for each path and level, shape ``(N, K)``.
        levels: Saving levels, shape ``(K,)``.
        target: Success probability to reach.

    Returns:
        ``(saving, k)``: the saving (interpolated between levels, capped at the
        top level) and the index of the level whose policy is used.
    """
    n, k_levels = estimates.shape
    ok = estimates >= target
    reachable = ok.any(axis=1)
    k = np.where(reachable, ok.argmax(axis=1), k_levels - 1)
    lo = np.maximum(k - 1, 0)
    rows = np.arange(n)
    e_lo, e_hi = estimates[rows, lo], estimates[rows, k]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(e_hi > e_lo, (target - e_lo) / (e_hi - e_lo), 1.0)
    frac = np.clip(frac, 0.0, 1.0)
    saving = levels[lo] + frac * (levels[k] - levels[lo])
    saving = np.where(k == 0, levels[0], saving)
    saving = np.where(reachable, saving, levels[-1])
    return saving, k


@dataclass(frozen=True)
class ReviewResult:
    """Simulated paths under a saving rule: wealth ``(N, T+1)``, stock weight and saving ``(N, T)``."""

    wealth: Array
    weights: Array
    saving: Array

    @property
    def terminal(self) -> Array:
        return self.wealth[:, -1]


def simulate_review(tables_by_year: Sequence[SavingTables], r_s: Array, r_b: Array, w0: float,
                    target: float) -> ReviewResult:
    """Run the annual review along given return paths.

    ``tables_by_year[t]`` holds what the investor believes in year ``t`` (the
    same tables every year, or new ones after a change in the assumptions).
    With single-level tables this is a fixed-saving plan.
    """
    n_paths, horizon = r_s.shape
    wealth = np.empty((n_paths, horizon + 1))
    weights = np.empty((n_paths, horizon))
    saving = np.empty((n_paths, horizon))
    wealth[:, 0] = w0
    for t in range(horizon):
        tb = tables_by_year[t]
        w = wealth[:, t]
        est = np.column_stack([np.interp(w, tb.grid, tb.value[k, t]) for k in range(tb.levels.size)])
        c, k = choose_saving(est, tb.levels, target)
        mids = np.sqrt(tb.grid[:-1] * tb.grid[1:])
        idx = np.clip(np.searchsorted(mids, w, side="right"), 0, tb.grid.size - 1)
        a = tb.policy[k, t, idx]
        weights[:, t], saving[:, t] = a, c
        wealth[:, t + 1] = w * (a * r_s[:, t] + (1.0 - a) * r_b[:, t]) + c
    return ReviewResult(wealth, weights, saving)


def calibrate_target(tables: SavingTables, r_s: Array, r_b: Array, investor: Investor, success: float = TARGET,
                     tol: float = 1e-4) -> float:
    """Review target that makes the *realised* success probability equal ``success`` (bisection)."""
    horizon = r_s.shape[1]

    def realised(tau: float) -> float:
        res = simulate_review([tables] * horizon, r_s, r_b, investor.w0, tau)
        return float((res.terminal >= investor.goal).mean())

    lo, hi = 0.30, 0.995
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if realised(mid) >= success else (mid, hi)
    return hi


def summarize(res: ReviewResult, goal: float, split: int | None = None) -> dict[str, float]:
    """Success, saving statistics (per path: average yearly saving) and the share of years at the cap."""
    p, se = metrics.success_probability(res.terminal, goal)
    per_path = res.saving.mean(axis=1)
    # Saving at the end of year t discounted at the base-case bond return, as the level yearly
    # amount with the same present value (so early and late dollars are compared fairly).
    disc = 1.0 / (1.0 + market.BASE.mu_bond) ** np.arange(1, res.saving.shape[1] + 1)
    out = {
        "success": p,
        "success_se": se,
        "avg_saving": float(per_path.mean()),
        "pv_level_saving": float((res.saving @ disc).mean() / disc.sum()),
        "saving_p10": float(np.quantile(per_path, 0.10)),
        "saving_p90": float(np.quantile(per_path, 0.90)),
        "share_years_at_cap": float((res.saving >= SAVING_GRID[-1] - 1e-6).mean()),
        "avg_stock_weight": float(res.weights.mean()),
        "saving_by_year_median": np.median(res.saving, axis=0).tolist(),
        "saving_by_year_p10": np.quantile(res.saving, 0.10, axis=0).tolist(),
        "saving_by_year_p90": np.quantile(res.saving, 0.90, axis=0).tolist(),
    }
    if split is not None:
        out["avg_saving_before"] = float(res.saving[:, :split].mean())
        out["avg_saving_after"] = float(res.saving[:, split:].mean())
    return out


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def experiment(investor: Investor, n_paths: int = analysis.SIM_PATHS, target: float = TARGET,
               shock_year: int = 10, shocked_mu: float = 0.05) -> dict[str, Any]:
    """Saving needed for ``target`` success: fixed or reviewed every year, with 60/40 or the dynamic allocation.

    Then the same rule when the expected stock return drops in ``shock_year``
    (noticed at once; the tables switch to the new assumptions).
    """
    base = market.BASE
    shocked = base.shifted(f"Stock return {shocked_mu:.0%} from year {shock_year}", mu_stock=shocked_mu)
    fixed_dp, fixed_mix = analysis.run_parallel(
        analysis.required_saving, [(investor, World(base), target, "dp"), (investor, World(base), target, "60/40")])
    levels = list(SAVING_GRID)
    dyn, mix, dyn_shock, fixed_tabs, fixed_tabs_shock = build_tables(investor, [
        (base, levels, None), (base, levels, 0.6), (shocked, levels, None), (base, [fixed_dp], None),
        (shocked, [fixed_dp], None)])
    fixed_mix_tabs = build_tables(investor, [(base, [fixed_mix], 0.6)])[0]

    horizon = investor.horizon
    r_s, r_b = World(base).paths(n_paths, horizon)
    tau_dyn = calibrate_target(dyn, r_s, r_b, investor, target)
    tau_mix = calibrate_target(mix, r_s, r_b, investor, target)

    def run(tabs: Sequence[SavingTables], tau: float, paths: tuple[Array, Array], split: int | None = None) -> dict:
        return summarize(simulate_review(tabs, *paths, investor.w0, tau), investor.goal, split)

    base_paths = (r_s, r_b)
    base_rows = {
        "Fixed 60/40, fixed saving": run([fixed_mix_tabs] * horizon, 0.0, base_paths),
        "Fixed 60/40, saving reviewed every year": run([mix] * horizon, tau_mix, base_paths),
        "Dynamic allocation, fixed saving": run([fixed_tabs] * horizon, 0.0, base_paths),
        "Dynamic allocation, saving reviewed every year": run([dyn] * horizon, tau_dyn, base_paths),
    }

    truth = resolve.changing(base, horizon, shock_year, shocked.label, mu_stock=shocked_mu)
    shock_paths = resolve.paths(truth, n_paths)

    def switch(before: SavingTables, after: SavingTables) -> list[SavingTables]:
        return [before] * shock_year + [after] * (horizon - shock_year)

    shock_rows = {
        f"Fixed ${investor.contribution:,.0f}": run(switch(dyn.at(investor.contribution), dyn_shock.at(investor.contribution)),
                                                   0.0, shock_paths, shock_year),
        f"Fixed ${round(fixed_dp, -2):,.0f} (the 75% plan)": run(switch(fixed_tabs, fixed_tabs_shock), 0.0, shock_paths, shock_year),
        "Reviewed every year": run(switch(dyn, dyn_shock), tau_dyn, shock_paths, shock_year),
    }
    return {
        "target": target,
        "fixed_saving_dp": fixed_dp,
        "fixed_saving_60_40": fixed_mix,
        "review_target_dp": tau_dyn,
        "review_target_60_40": tau_mix,
        "cap": float(SAVING_GRID[-1]),
        "base": base_rows,
        "shock": {"label": shocked.label, "year": shock_year, "rows": shock_rows},
    }
