"""The stock/bond goal problem and the experiments built on it.

Everything here is a thin layer over :mod:`gbda.dp` (solve), :mod:`gbda.market`
(returns), :mod:`gbda.simulate` (forward paths) and :mod:`gbda.metrics`.
Each experiment is a plain function of picklable inputs that returns numbers,
so a batch of them can be spread over processes with :func:`run_parallel`.

Two seeds are used on purpose: the DP optimises against one set of return
scenarios (``DP_SEED``) and every policy is then scored on fresh paths
(``SIM_SEED``). The reported numbers are therefore out-of-sample.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Sequence

import numpy as np

from . import dp, market, metrics, simulate
from .market import Assumptions, History

Array = np.ndarray

DP_SEED = 1
SIM_SEED = 2024
ACTIONS = np.round(np.linspace(0.0, 1.0, 11), 2)
DP_SCENARIOS = 4000
GRID_POINTS = 300
SIM_PATHS = 100_000
GLIDE_START, GLIDE_END = 0.9, 0.3
#: A success probability at or below this is treated as "out of reach" (hold all stock).
#: With a few thousand scenarios, anything smaller is sampling noise.
GOAL_ZERO_TOL = 1e-4


# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Investor:
    """Who is investing: starting wealth, yearly saving, horizon and target."""

    w0: float = 50_000.0
    contribution: float = 10_000.0
    horizon: int = 20
    goal: float = 500_000.0

    def required_return(self) -> float:
        """Constant annual return at which the plan lands exactly on the goal."""
        from scipy.optimize import brentq

        def gap(r: float) -> float:
            w = self.w0
            for _ in range(self.horizon):
                w = w * (1.0 + r) + self.contribution
            return w - self.goal

        return float(brentq(gap, -0.5, 1.0))


@dataclass(frozen=True)
class World:
    """How annual returns are generated.

    ``dist`` is ``"lognormal"``, ``"student_t"`` (fat tails, same mean and
    volatility) or ``"bootstrap"`` (blocks of consecutive historical years from
    ``history``; ``assumptions`` is then only a label).
    """

    assumptions: Assumptions = market.BASE
    dist: Literal["lognormal", "student_t", "bootstrap"] = "lognormal"
    df: float = 5.0
    history: History | None = None
    block: int = 5

    @property
    def label(self) -> str:
        return self.assumptions.label

    def scenarios(self, m: int = DP_SCENARIOS, seed: int = DP_SEED, moment_match: bool = True) -> dp.ReturnScenarios:
        """One-year return scenarios for the DP expectation."""
        if self.dist == "bootstrap":
            assert self.history is not None
            # One year ahead, the bootstrap distribution is simply the set of historical years.
            return dp.ReturnScenarios.equally_weighted(self.history.stock, self.history.bond)
        rs, rb = market.sample_returns(self.assumptions, m, seed, self.dist, self.df, moment_match=moment_match)
        return dp.ReturnScenarios.equally_weighted(rs, rb)

    def paths(self, n_paths: int, horizon: int, seed: int = SIM_SEED) -> tuple[Array, Array]:
        """Forward return paths, shape ``(n_paths, horizon)`` each."""
        if self.dist == "bootstrap":
            assert self.history is not None
            return market.block_bootstrap(self.history, n_paths, horizon, seed, self.block)
        return market.sample_returns(self.assumptions, (n_paths, horizon), seed, self.dist, self.df)


def make_objective(name: str, investor: Investor) -> dp.Objective:
    """``"goal"``, ``"log"`` or ``"power:<gamma>"`` -> objective for this investor."""
    if name == "goal":
        return dp.goal_objective(investor.goal)
    if name == "log":
        return dp.log_objective(scale=investor.goal)
    if name.startswith("power:"):
        return dp.power_objective(float(name.split(":")[1]), scale=investor.goal)
    raise ValueError(f"unknown objective {name!r}")


def wealth_grid(investor: Investor, n: int = GRID_POINTS) -> Array:
    """Log-spaced wealth grid from far below to far above the goal, with the goal on it."""
    return dp.log_grid(investor.goal / 250.0, investor.goal * 40.0, n, include=investor.goal)


def solve_policy(
    investor: Investor,
    world: World,
    objective: str = "goal",
    max_stock: float = 1.0,
    m: int = DP_SCENARIOS,
    seed: int = DP_SEED,
    n_grid: int = GRID_POINTS,
    moment_match: bool = True,
) -> dp.DPSolution:
    """Solve the DP for one investor, one return model and one objective.

    ``max_stock`` caps the stock weight (a risk limit or a regulatory limit).
    """
    actions = ACTIONS[ACTIONS <= max_stock + 1e-9]
    return dp.solve(
        world.scenarios(m, seed, moment_match),
        actions,
        wealth_grid(investor, n_grid),
        make_objective(objective, investor),
        investor.horizon,
        investor.contribution,
        zero_tol=GOAL_ZERO_TOL,
    )


def run(policy: simulate.Policy, investor: Investor, paths: tuple[Array, Array]) -> simulate.SimResult:
    """Simulate a policy for this investor along the given return paths."""
    return simulate.simulate(policy, paths[0], paths[1], investor.w0, investor.contribution)


def glide(investor: Investor) -> simulate.Policy:
    """The reference glide path: stock weight falling linearly from 90% to 30%."""
    return simulate.glide_path(GLIDE_START, GLIDE_END, investor.horizon)


def run_parallel(fn: Callable[..., Any], jobs: Sequence[tuple], workers: int | None = None) -> list[Any]:
    """Run ``fn(*job)`` for every job across processes, keeping the order."""
    workers = workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    if workers <= 1:
        return [fn(*job) for job in jobs]
    # One numerical thread per worker: the processes already use every core.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, *job) for job in jobs]
        return [f.result() for f in futures]


# --------------------------------------------------------------------------- #
# One experiment = one function returning plain numbers
# --------------------------------------------------------------------------- #
def goal_case(
    investor: Investor,
    world: World,
    plan_world: World | None = None,
    max_stock: float = 1.0,
    n_paths: int = SIM_PATHS,
    m: int = DP_SCENARIOS,
    dp_seed: int = DP_SEED,
    moment_match: bool = True,
) -> dict[str, float]:
    """Solve the goal DP and score it, with static benchmarks, on the same paths.

    Args:
        investor: The plan.
        world: The return model the policies are *scored* in.
        plan_world: The return model the DP *optimises* against. Defaults to
            ``world``. Passing a different one measures the cost of planning
            with the wrong assumptions.
        max_stock: Cap on the stock weight for the DP and the fixed mixes.
        n_paths: Forward paths.
        m: DP scenarios.
        dp_seed: Seed of the DP scenario set.
        moment_match: Moment-match the DP scenario set.

    Returns:
        DP value at the start, forward-simulated success and tail statistics for
        the DP policy, plus success of 60/40, the glide path and the best fixed mix.
    """
    solution = solve_policy(investor, plan_world or world, "goal", max_stock, m, dp_seed, moment_match=moment_match)
    paths = world.paths(n_paths, investor.horizon)
    sim = run(simulate.dp_policy(solution), investor, paths)
    out = metrics.summarize(sim.terminal, investor.goal, sim.weights)
    out["dp_value"] = solution.value_at(0, investor.w0)
    out["initial_stock_weight"] = float(solution.action(0, investor.w0))

    fixed = {}
    for a in ACTIONS[ACTIONS <= max_stock + 1e-9]:
        fixed[float(a)] = metrics.success_probability(run(simulate.fixed_mix(a), investor, paths).terminal, investor.goal)[0]
    best = max(fixed, key=fixed.get)  # type: ignore[arg-type]
    out["best_fixed_weight"] = best
    out["best_fixed_success"] = fixed[best]
    out["fixed_60_40_success"] = fixed.get(0.6, float("nan"))
    out["all_stock_success"] = fixed.get(1.0, float("nan"))
    out["glide_success"] = metrics.success_probability(run(glide(investor), investor, paths).terminal, investor.goal)[0]
    return out


def required_saving(
    investor: Investor,
    world: World,
    target: float = 0.75,
    strategy: Literal["dp", "60/40"] = "dp",
    low: float = 0.0,
    high: float = 60_000.0,
    tol: float = 100.0,
    n_paths: int = 50_000,
) -> float:
    """Yearly saving needed to reach ``target`` success probability (bisection, forward-simulated)."""
    paths = world.paths(n_paths, investor.horizon)

    def success(c: float) -> float:
        inv = replace(investor, contribution=c)
        if strategy == "dp":
            policy = simulate.dp_policy(solve_policy(inv, world, "goal"))
        else:
            policy = simulate.fixed_mix(0.6)
        return metrics.success_probability(run(policy, inv, paths).terminal, inv.goal)[0]

    if success(high) < target:
        return float("nan")
    while high - low > tol:
        mid = 0.5 * (low + high)
        if success(mid) >= target:
            high = mid
        else:
            low = mid
    return float(high)


def dp_sampling_case(investor: Investor, world: World, m: int, dp_seed: int, moment_match: bool) -> dict[str, float]:
    """DP estimate vs forward-simulated success for one DP scenario draw (precision study)."""
    res = goal_case(investor, world, n_paths=SIM_PATHS, m=m, dp_seed=dp_seed, moment_match=moment_match)
    return {"m": m, "seed": dp_seed, "moment_match": float(moment_match), "dp_value": res["dp_value"], "success": res["success"]}
