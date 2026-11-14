"""Two market states (regimes) in the DP state: the policy plans for the market switching.

The market is in a *good* or a *bad* state each year. Stocks have a higher
expected return and lower volatility in the good state; bonds are the same in
both. At the end of each year the state stays the same with probability
``stay`` (0.9: a state lasts ten years on average) and switches otherwise.
The investor sees the current state at the start of each year. That is a
simplification; the second half of this module drops it: the state has to be
inferred from stock returns, and the DP state carries the probability that the
market is in the bad state.

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

from . import analysis, dp, market, metrics, simulate
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


# =========================================================================== #
# The state cannot be seen: infer it from stock returns
# =========================================================================== #
# The investor knows the two regimes and how they switch, but not which one the
# market is in. The DP state becomes (year, wealth, b) with b = P(bad state this
# year | returns so far). Each year b is updated from the stock return by Bayes'
# rule and then moved forward one step with the transition probabilities. It is
# the same structure as learning the expected return (param_uncertainty), except
# that the thing being learned keeps changing.
def _log_stock(a: Assumptions) -> tuple[float, float]:
    return market.lognormal_params(a.mu_stock, a.sigma_stock)


def update_belief(model: RegimeModel, p_bad: Array | float, log_stock: Array) -> Array:
    """P(bad state next year), from P(bad) this year and this year's log stock return.

    Bayes' rule on the stock return (bonds are the same in both states, and the
    investor ignores what the stock-bond correlation could add), then one step of
    the Markov chain.
    """
    (mg, sg), (mb, sb) = _log_stock(model.good), _log_stock(model.bad)
    x = np.asarray(log_stock, dtype=float)
    like_good = np.exp(-0.5 * ((x - mg) / sg) ** 2) / sg
    like_bad = np.exp(-0.5 * ((x - mb) / sb) ** 2) / sb
    p = np.asarray(p_bad, dtype=float)
    post = p * like_bad / (p * like_bad + (1.0 - p) * like_good)
    return post * model.stay_bad + (1.0 - post) * (1.0 - model.stay_good)


def filter_beliefs(model: RegimeModel, r_s: Array, prior: float) -> Array:
    """P(bad state) at the start of each year along each path, shape ``(N, T)``."""
    n_paths, horizon = r_s.shape
    out = np.empty((n_paths, horizon))
    b = np.full(n_paths, float(prior))
    for t in range(horizon):
        out[:, t] = b
        b = update_belief(model, b, np.log(r_s[:, t]))
    return out


def belief_grid(n_beliefs: int) -> Array:
    """Beliefs P(bad) used by the DP: evenly spaced on [0, 1], so 0, 1/2 and 1 are on it for odd ``n``."""
    return np.linspace(0.0, 1.0, n_beliefs)


@dataclass(frozen=True)
class HiddenSteps:
    """``step(t, b)`` for the hidden-state DP (picklable).

    At belief ``p = grid[b]`` next year's returns are a mixture: the good-state
    scenarios with total weight ``1 - p`` and the bad-state ones with ``p``. After
    each scenario the belief moves to :func:`update_belief` of that scenario's
    stock return, which lands between two grid points.
    """

    model: RegimeModel
    good: dp.ReturnScenarios
    bad: dp.ReturnScenarios
    grid: Array

    def __call__(self, t: int, b: int) -> dp.StateStep:
        p = float(self.grid[b])
        m = self.good.r_a.size
        sc = dp.ReturnScenarios(np.r_[self.good.r_a, self.bad.r_a], np.r_[self.good.r_b, self.bad.r_b],
                                np.r_[np.full(m, (1.0 - p) / m), np.full(self.bad.r_a.size, p / self.bad.r_a.size)])
        nxt = update_belief(self.model, p, np.log(sc.r_a))
        k = self.grid.size
        pos = np.clip(nxt * (k - 1), 0.0, k - 1.0)
        lo = np.minimum(np.floor(pos).astype(int), k - 2)
        return dp.StateStep(sc, lo, lo + 1, pos - lo)


def solve_hidden(model: RegimeModel, investor: Investor, n_beliefs: int = 11, m: int = 2000,
                 seed: int = analysis.DP_SEED) -> dp.StateDPSolution:
    """Goal-probability policy for every (year, wealth, belief that the state is bad)."""
    steps = HiddenSteps(model, World(model.good).scenarios(m, seed), World(model.bad).scenarios(m, seed),
                        belief_grid(n_beliefs))
    return dp.solve_with_state(steps, n_beliefs, analysis.ACTIONS, analysis.wealth_grid(investor), investor.goal,
                               investor.horizon, investor.contribution, zero_tol=analysis.GOAL_ZERO_TOL)


def hidden_policy(model: RegimeModel, solution: dp.StateDPSolution, r_s: Array, prior: float) -> simulate.Policy:
    """Forward policy that filters each path's belief from its stock returns and looks up the table."""
    k = solution.n_states
    idx = np.rint(filter_beliefs(model, r_s, prior) * (k - 1)).astype(int)

    def pol(t: int, wealth: Array) -> Array:
        return solution.action(t, wealth, idx[:, t])

    return pol


def _solve_kind(kind: str, model: RegimeModel, investor: Investor, n_beliefs: int, m: int) -> dp.StateDPSolution:
    """Process-pool entry point for the two solves in :func:`hidden_experiment`."""
    return solve_hidden(model, investor, n_beliefs, m) if kind == "hidden" else model.solve(investor)


def hidden_experiment(investor: Investor, model: RegimeModel, n_paths: int = analysis.SIM_PATHS,
                      n_beliefs: int = 11, m: int = 2000) -> dict[str, Any]:
    """Seeing the state, inferring it from returns, ignoring it: scored on the same regime paths.

    Three starts: the first state known to be good, known to be bad (later
    states hidden in both), or drawn from the long-run mix and unknown.
    ``m`` scenarios per state for the hidden-state DP, so each belief point sees
    ``2 m`` (the same 4,000 as every other DP here by default); the
    regime-aware DP uses the usual 4,000 per state.
    """
    hidden, aware = analysis.run_parallel(
        _solve_kind, [("hidden", model, investor, n_beliefs, m), ("aware", model, investor, n_beliefs, m)])
    blind = analysis.solve_policy(investor, World(Assumptions()))
    k = hidden.n_states
    seen, inferred, ignored = "Sees the state", "Infers the state from returns", "Ignores the state (base case)"
    by_start: dict[str, dict[str, dict[str, float]]] = {}
    tracking: dict[str, float] = {}
    for name, start, prior in (("good", GOOD, 0.0), ("bad", BAD, 1.0),
                               ("long-run mix", "stationary", float(model.stationary[BAD]))):
        reg = model.regimes(n_paths, investor.horizon, start)
        r_s, r_b = model.paths(reg)
        pols = {
            seen: regime_policy(aware, reg),
            inferred: hidden_policy(model, hidden, r_s, prior),
            ignored: simulate.dp_policy(blind),
            "Fixed 100% stock": simulate.fixed_mix(1.0),
            "Fixed 60/40": simulate.fixed_mix(0.6),
        }
        sims = {key: simulate.simulate(p, r_s, r_b, investor.w0, investor.contribution) for key, p in pols.items()}
        hit_blind = sims[ignored].terminal >= investor.goal
        rows: dict[str, dict[str, float]] = {}
        for key, sim in sims.items():
            hit = sim.terminal >= investor.goal
            p, se = metrics.success_probability(sim.terminal, investor.goal)
            diff, se_diff, _ = metrics.paired_difference(hit, hit_blind)
            rows[key] = {"success": p, "success_se": se, "vs_ignore": diff, "vs_ignore_se": se_diff,
                         "avg_stock_weight": float(sim.weights.mean())}
        rows[inferred]["dp_value"] = hidden.value_at(0, investor.w0, int(round(prior * (k - 1))))
        rows[ignored]["dp_value"] = blind.value_at(0, investor.w0)
        if start != "stationary":
            rows[seen]["dp_value"] = aware.value_at(0, investor.w0, int(start))
        by_start[name] = rows
        if start == "stationary":  # how well the filter tracks the state, from year 1 on
            all_beliefs = filter_beliefs(model, r_s, prior)
            beliefs, bad = all_beliefs[:, 1:], reg[:, 1:] == BAD
            switches = (np.diff(reg, axis=1) != 0).sum(axis=1)
            example = int(np.argmax(switches >= 2))  # first path with at least two switches, for the figure
            tracking = {
                "share_of_years_called_right": float(((beliefs > 0.5) == bad).mean()),
                "mean_belief_when_bad": float(beliefs[bad].mean()),
                "mean_belief_when_good": float(beliefs[~bad].mean()),
                "example_regimes": reg[example].tolist(),
                "example_beliefs": all_beliefs[example].tolist(),
                "example_stock_returns": (r_s[example] - 1.0).tolist(),
            }
    return {"by_start": by_start, "tracking": tracking, "n_beliefs": n_beliefs, "dp_scenarios": m,
            "policy_year5": {"grid": hidden.grid.tolist(), "beliefs": belief_grid(k).tolist(),
                             "weights": hidden.policy[5].tolist()}}

