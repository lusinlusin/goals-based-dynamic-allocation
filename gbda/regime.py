"""Two market states (regimes) in the DP state: the policy plans for the market switching.

The market is in a *good* or a *bad* state each year. Stocks have a higher
expected return and lower volatility in the good state; bonds are the same in
both. At the end of each year the state stays the same with probability
``stay`` (0.9: a state lasts ten years on average) and switches otherwise.
The investor sees the current state at the start of each year (a
simplification: in practice it has to be inferred from data).

The DP state becomes ``(year, wealth, regime)``; the next regime does not
depend on this year's return, so in :func:`gbda.dp.solve_with_state` every
scenario moves to regime 0 or 1 with the transition probabilities.

The two states are calibrated so that their long-run mix has the base case's
mean and volatility (7% and 16%). A planner who ignores the regimes and uses
the base case therefore has the right one-year averages; what it misses is
which state the market is in now and that states persist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import analysis, dp, metrics, simulate
from .analysis import Investor, World
from .market import Assumptions

Array = np.ndarray
GOOD, BAD = 0, 1
#: Seed offset for the regime chains (keeps the return shocks equal to the base case).
REGIME_SEED_OFFSET = 104729


@dataclass(frozen=True)
class RegimeModel:
    """Two return regimes and how they switch.

    Attributes:
        good, bad: Return assumptions in each state.
        stay_good, stay_bad: Probability of staying in the same state next year.
    """

    good: Assumptions = field(default_factory=Assumptions)
    bad: Assumptions = field(default_factory=Assumptions)
    stay_good: float = 0.9
    stay_bad: float = 0.9

    @property
    def transition(self) -> Array:
        """``P[i, j]`` = probability of state ``j`` next year given state ``i`` now."""
        return np.array([[self.stay_good, 1.0 - self.stay_good], [1.0 - self.stay_bad, self.stay_bad]])

    @property
    def stationary(self) -> Array:
        """Long-run share of years in each state."""
        leave_g, leave_b = 1.0 - self.stay_good, 1.0 - self.stay_bad
        if leave_g + leave_b == 0:
            return np.array([0.5, 0.5])
        return np.array([leave_b, leave_g]) / (leave_g + leave_b)

    def with_stay(self, stay: float) -> "RegimeModel":
        """Same regimes, different persistence (used to plan with a wrong transition probability)."""
        return RegimeModel(self.good, self.bad, stay, stay)

    # ---- the DP --------------------------------------------------------------- #
    def stepper(self, m: int = analysis.DP_SCENARIOS, seed: int = analysis.DP_SEED) -> "RegimeSteps":
        """Scenario sets for the DP: one per regime, drawn with the same shocks."""
        return RegimeSteps(World(self.good).scenarios(m, seed), World(self.bad).scenarios(m, seed), self.transition)

    def solve(self, investor: Investor, m: int = analysis.DP_SCENARIOS, seed: int = analysis.DP_SEED) -> dp.StateDPSolution:
        """Goal-probability policy for every (year, wealth, regime)."""
        return dp.solve_with_state(
            self.stepper(m, seed),
            2,
            analysis.ACTIONS,
            analysis.wealth_grid(investor),
            investor.goal,
            investor.horizon,
            investor.contribution,
            zero_tol=analysis.GOAL_ZERO_TOL,
        )

    # ---- the world ------------------------------------------------------------ #
    def regimes(self, n_paths: int, horizon: int, start: int | str = "stationary", seed: int = analysis.SIM_SEED) -> Array:
        """Regime of each path in each year, shape ``(N, T)``.

        ``start`` is ``GOOD``, ``BAD`` or ``"stationary"`` (drawn from the long-run shares).
        """
        rng = np.random.default_rng(seed + REGIME_SEED_OFFSET)
        u = rng.random((n_paths, horizon))
        out = np.empty((n_paths, horizon), dtype=int)
        out[:, 0] = (u[:, 0] < self.stationary[BAD]).astype(int) if start == "stationary" else int(start)
        p_bad_next = self.transition[:, BAD]
        for t in range(1, horizon):
            out[:, t] = (u[:, t] < p_bad_next[out[:, t - 1]]).astype(int)
        return out

    def paths(self, regimes: Array, seed: int = analysis.SIM_SEED) -> tuple[Array, Array]:
        """Returns along the given regime paths. Both regimes use the same shocks."""
        n_paths, horizon = regimes.shape
        gs, gb = World(self.good).paths(n_paths, horizon, seed)
        bs, bb = World(self.bad).paths(n_paths, horizon, seed)
        bad = regimes == BAD
        return np.where(bad, bs, gs), np.where(bad, bb, gb)


@dataclass(frozen=True)
class RegimeSteps:
    """``step(t, regime)`` for :func:`gbda.dp.solve_with_state` (picklable)."""

    good: dp.ReturnScenarios
    bad: dp.ReturnScenarios
    transition: Array

    def __call__(self, t: int, s: int) -> dp.StateStep:
        sc = self.good if s == GOOD else self.bad
        m = sc.r_a.size
        return dp.StateStep(sc, np.full(m, GOOD), np.full(m, BAD), np.full(m, self.transition[s, BAD]))


def calibrated(base: Assumptions = Assumptions(), gap: float = 0.03, sigma_good: float = 0.12, stay: float = 0.9) -> RegimeModel:
    """Good/bad stock regimes whose 50/50 long-run mix has the base mean and volatility.

    Expected stock return ``base +/- gap``; volatility ``sigma_good`` in the good
    state and whatever makes the mixture's variance equal ``base.sigma_stock^2``
    in the bad one: ``(sg^2 + sb^2)/2 + gap^2 = sigma^2``.
    """
    sigma_bad = float(np.sqrt(2.0 * (base.sigma_stock**2 - gap**2) - sigma_good**2))
    good = base.shifted("Good regime", mu_stock=base.mu_stock + gap, sigma_stock=sigma_good)
    bad = base.shifted("Bad regime", mu_stock=base.mu_stock - gap, sigma_stock=sigma_bad)
    return RegimeModel(good, bad, stay, stay)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def regime_policy(solution: dp.StateDPSolution, regimes: Array) -> simulate.Policy:
    """Forward policy that looks up the table for each path's current regime."""

    def pol(t: int, wealth: Array) -> Array:
        return solution.action(t, wealth, regimes[:, t])

    return pol


def _solve_job(model: RegimeModel, investor: Investor) -> dp.StateDPSolution:
    return model.solve(investor)


def experiment(
    investor: Investor,
    model: RegimeModel,
    assumed_stay: tuple[float, ...] = (0.5, 0.75, 0.95, 0.99),
    n_paths: int = analysis.SIM_PATHS,
) -> dict[str, Any]:
    """Regime-aware policy against a regime-blind one, by starting regime; cost of a wrong ``stay``.

    The regime-blind policy is the base-case DP (it plans on the long-run
    average returns). Every policy is scored on the same paths.
    """
    stays = (model.stay_good,) + tuple(assumed_stay)
    models = [model] + [model.with_stay(q) for q in assumed_stay]
    solutions = analysis.run_parallel(_solve_job, [(m_, investor) for m_ in models])
    aware = solutions[0]
    blind = analysis.solve_policy(investor, World(Assumptions()))

    by_start: dict[str, dict[str, dict[str, float]]] = {}
    for start_name, start in (("good", GOOD), ("bad", BAD), ("long-run mix", "stationary")):
        reg = model.regimes(n_paths, investor.horizon, start)
        r_s, r_b = model.paths(reg)
        pols = {
            "Regime-aware DP": regime_policy(aware, reg),
            "Regime-blind DP (base case)": simulate.dp_policy(blind),
            "Fixed 100% stock": simulate.fixed_mix(1.0),
            "Fixed 60/40": simulate.fixed_mix(0.6),
        }
        for q, sol in zip(stays[1:], solutions[1:]):
            pols[f"Regime-aware, planned with stay = {q:g}"] = regime_policy(sol, reg)
        sims = {k: simulate.simulate(p, r_s, r_b, investor.w0, investor.contribution) for k, p in pols.items()}
        hit_aware = sims["Regime-aware DP"].terminal >= investor.goal
        rows: dict[str, dict[str, float]] = {}
        for name, sim in sims.items():
            hit = sim.terminal >= investor.goal
            p, se = metrics.success_probability(sim.terminal, investor.goal)
            diff, se_diff, _ = metrics.paired_difference(hit, hit_aware)
            rows[name] = {"success": p, "success_se": se, "vs_aware": diff, "vs_aware_se": se_diff,
                          "avg_stock_weight": float(sim.weights.mean())}
        if start != "stationary":
            rows["Regime-aware DP"]["dp_value"] = aware.value_at(0, investor.w0, int(start))
        rows["Regime-blind DP (base case)"]["dp_value"] = blind.value_at(0, investor.w0)
        by_start[start_name] = rows

    return {
        "good": {"mu_stock": model.good.mu_stock, "sigma_stock": model.good.sigma_stock},
        "bad": {"mu_stock": model.bad.mu_stock, "sigma_stock": model.bad.sigma_stock},
        "stay": model.stay_good,
        "by_start": by_start,
        "policy": {"grid": aware.grid.tolist(), "good": aware.policy[:, GOOD].tolist(),
                   "bad": aware.policy[:, BAD].tolist(), "blind": blind.policy.tolist()},
    }
