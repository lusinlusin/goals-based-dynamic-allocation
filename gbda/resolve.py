"""Return assumptions that change during the plan: re-solve, and what it is worth.

The setting: a 20-year plan is solved under the base case, and in year 10 the
expected stock return drops (or volatility rises). Three ways to run the plan:

1. **Keep the original table.** Never look at the assumptions again.
2. **Re-solve.** At the start of every year, solve the remaining years again
   with the assumptions believed at that time, starting from wherever wealth is
   (rolling re-optimisation, receding horizon). Each re-solve assumes the new
   assumptions will then hold to the end.
3. **Hindsight.** Solve once at the outset, knowing when and how the
   assumptions will change. Not available in practice; it is the upper bound.

All three are scored on the same return paths, which follow the true
(changing) assumptions. A check comes for free: re-solving with *unchanged*
assumptions must give back the remaining rows of the original table exactly
(optimality principle), so years in which nothing changed cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from . import analysis, dp, metrics, policy, simulate
from .analysis import Investor, World
from .market import Assumptions

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Assumption paths and return paths
# --------------------------------------------------------------------------- #
def changing(base: Assumptions, horizon: int, year: int, label: str = "", **changes: float) -> list[Assumptions]:
    """Year-by-year assumptions: ``base`` before ``year``, ``base`` with ``changes`` from ``year`` on."""
    new = base.shifted(label or f"changed in year {year}", **changes)
    return [base] * year + [new] * (horizon - year)


def paths(assumption_path: Sequence[Assumptions], n_paths: int, seed: int = analysis.SIM_SEED) -> tuple[Array, Array]:
    """Lognormal return paths whose assumptions follow ``assumption_path`` (one entry per year).

    Every distinct assumption set is drawn with the same seed and the columns
    are spliced, so the paths share their shocks with ``World(a).paths(...)``
    for any single ``a`` (common random numbers across experiments).
    """
    horizon = len(assumption_path)
    r_s = np.empty((n_paths, horizon))
    r_b = np.empty((n_paths, horizon))
    for a in dict.fromkeys(assumption_path):  # distinct, in order
        cols = [t for t, x in enumerate(assumption_path) if x == a]
        s, b = World(a).paths(n_paths, horizon, seed)
        r_s[:, cols], r_b[:, cols] = s[:, cols], b[:, cols]
    return r_s, r_b


# --------------------------------------------------------------------------- #
# The three ways to run the plan
# --------------------------------------------------------------------------- #
def rolling_tables(
    investor: Investor,
    beliefs: Sequence[Assumptions],
    years: Sequence[int] | None = None,
    parallel: bool = True,
) -> list[policy.PolicyTable]:
    """Re-solve the remaining years at the start of each year in ``years`` (default: every year).

    ``beliefs[t]`` is what the investor believes at the start of year ``t`` and
    is assumed to hold from then to the goal date. Year 0 is always solved.
    """
    horizon = investor.horizon
    years = sorted(set(range(horizon) if years is None else years) | {0})
    jobs = [(replace(investor, horizon=horizon - t), World(beliefs[t]), "goal", t) for t in years]
    if parallel:
        return analysis.run_parallel(policy.solve_policy, jobs)
    return [policy.solve_policy(*job) for job in jobs]


def change_years(beliefs: Sequence[Assumptions]) -> list[int]:
    """Years in which the belief differs from the year before (plus year 0)."""
    return [0] + [t for t in range(1, len(beliefs)) if beliefs[t] != beliefs[t - 1]]


def hindsight(investor: Investor, assumption_path: Sequence[Assumptions]) -> dp.DPSolution:
    """One DP solved at the outset with the true year-by-year assumptions."""
    cache: dict[Assumptions, dp.ReturnScenarios] = {}
    scenarios = [cache.setdefault(a, World(a).scenarios()) for a in assumption_path]
    return dp.solve(
        scenarios,
        analysis.ACTIONS,
        analysis.wealth_grid(investor),
        dp.goal_objective(investor.goal),
        investor.horizon,
        investor.contribution,
        zero_tol=analysis.GOAL_ZERO_TOL,
    )


def catch_up_saving(
    investor: Investor,
    base: Assumptions,
    shock: "Shock",
    target: float,
    n_paths: int = 50_000,
    high: float = 60_000.0,
    tol: float = 100.0,
) -> float:
    """Yearly saving from the shock year on that brings the success probability back to ``target``.

    The change is noticed at once; the remaining years are re-solved with the new
    assumptions *and* the new saving amount. Bisection on forward-simulated
    success; ``nan`` if even ``high`` is not enough.
    """
    horizon, year = investor.horizon, shock.year
    truth = changing(base, horizon, year, shock.label, **shock.changes)
    r_s, r_b = paths(truth, n_paths)
    original = policy.solve_policy(investor, World(base))

    def success(c: float) -> float:
        rest = policy.solve_policy(replace(investor, horizon=horizon - year, contribution=c), World(truth[year]),
                                   start_year=year)
        saving = np.r_[np.full(year, investor.contribution), np.full(horizon - year, c)]
        sim = simulate.simulate(policy.as_policy([original, rest]), r_s, r_b, investor.w0, saving)
        return metrics.success_probability(sim.terminal, investor.goal)[0]

    low = investor.contribution
    if success(high) < target:
        return float("nan")
    while high - low > tol:
        mid = 0.5 * (low + high)
        low, high = (low, mid) if success(mid) >= target else (mid, high)
    return float(high)


def tail_matches(original: policy.PolicyTable, resolved: policy.PolicyTable) -> bool:
    """Does a table re-solved with unchanged assumptions equal the rest of the original? (It must.)"""
    s = resolved.start_year
    a, b = original.solution, resolved.solution
    return bool(np.array_equal(a.policy[s:], b.policy) and np.array_equal(a.value[s:], b.value))


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Shock:
    """A change in the assumptions part-way through the plan."""

    label: str
    year: int
    changes: dict[str, float]
    lags: tuple[int, ...] = (0,)


def experiment(
    investor: Investor,
    base: Assumptions,
    shock: Shock,
    n_paths: int = analysis.SIM_PATHS,
    annual: bool = False,
) -> dict[str, Any]:
    """Score the original table, re-solving (noticed ``lag`` years late) and hindsight.

    Args:
        investor: The plan.
        base: Assumptions the plan starts with (true before ``shock.year``).
        shock: When and how the assumptions change.
        n_paths: Forward paths.
        annual: Re-solve literally every year. Otherwise only in years when the
            belief changes, which gives the same tables (checked by
            :func:`tail_matches` and the tests) at a fraction of the cost.

    Returns:
        Success probability, standard error and paired difference against the
        original table for every approach, plus each table's own forecast of
        success at the shock year against what then happens.
    """
    horizon = investor.horizon
    truth = changing(base, horizon, shock.year, shock.label, **shock.changes)
    r_s, r_b = paths(truth, n_paths)

    def score(pol: simulate.Policy) -> simulate.SimResult:
        return simulate.simulate(pol, r_s, r_b, investor.w0, investor.contribution)

    original = policy.solve_policy(investor, World(base))
    sims: dict[str, simulate.SimResult] = {"Keep the original table": score(policy.as_policy(original))}
    forecasts: dict[str, float] = {}
    w_shock = sims["Keep the original table"].wealth[:, shock.year]
    forecasts["Keep the original table"] = float(original.success_estimate(shock.year, w_shock).mean())

    checks: dict[str, bool] = {}
    for lag in shock.lags:
        beliefs = truth[: shock.year] + [base] * lag + truth[shock.year + lag:]
        tables = rolling_tables(investor, beliefs, None if annual else change_years(beliefs))
        name = "Re-solve every year" if lag == 0 else f"Re-solve, change noticed {lag} years late"
        sims[name] = score(policy.as_policy(tables))
        # Every re-solve made before anything changed must equal the original table's tail.
        checks[name] = all(tail_matches(original, tb) for tb in tables[1:] if tb.start_year < shock.year + lag)
        noticed = next(tb for tb in tables if tb.start_year == shock.year + lag)
        w = sims[name].wealth[:, noticed.start_year]
        forecasts[name] = float(noticed.success_estimate(noticed.start_year, w).mean())

    sol = hindsight(investor, truth)
    sims["Hindsight: changes known from the start"] = score(simulate.dp_policy(sol))
    w = sims["Hindsight: changes known from the start"].wealth[:, shock.year]
    forecasts["Hindsight: changes known from the start"] = float(np.interp(w, sol.grid, sol.value[shock.year]).mean())
    sims["Fixed 100% stock"] = score(simulate.fixed_mix(1.0))
    sims["Fixed 60/40"] = score(simulate.fixed_mix(0.6))

    hit0 = sims["Keep the original table"].terminal >= investor.goal
    rows: dict[str, dict[str, float]] = {}
    for name, sim in sims.items():
        hit = sim.terminal >= investor.goal
        p, se = metrics.success_probability(sim.terminal, investor.goal)
        diff, se_diff, _ = metrics.paired_difference(hit, hit0)
        late = sim.weights[:, shock.year:]
        rows[name] = {
            "success": p,
            "success_se": se,
            "vs_original": diff,
            "vs_original_se": se_diff,
            "avg_stock_after_change": float(late.mean()),
            # The table's own estimate, averaged over paths, in the year the change is acted on
            # (compare with "success": what then actually happens).
            "forecast_at_change": forecasts.get(name, float("nan")),
        }
    return {
        "shock": {"label": shock.label, "year": shock.year, "changes": shock.changes},
        "before": {"mu_stock": base.mu_stock, "sigma_stock": base.sigma_stock},
        "rows": rows,
        "tail_checks": checks,
        "weights_by_year": {name: sim.weights.mean(axis=0).tolist() for name, sim in sims.items()},
    }
