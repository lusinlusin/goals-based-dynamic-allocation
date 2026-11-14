"""The expected stock return is not known: each path draws its own, and the investor learns it.

**World.** Each path draws its stock return parameter once, then all 20 years of
returns from it. On the log scale the stock return is ``x_t = m + sigma * z_t``
with ``m ~ N(m_hat, tau^2)`` drawn once per path. With
``tau = s / (1 + mu_hat)`` the expected annual return ``mu = exp(m + sigma^2/2) - 1``
is approximately ``N(mu_hat, s^2)``. Bonds and volatilities are known. With
``s = 0`` the paths are exactly the base-case paths.

**Investor.** Starts from the same prior and updates it every year from the
stock's log return (normal prior, normal returns with known volatility: the
posterior stays normal)::

    v_t      = 1 / (1/tau^2 + t/sigma^2)                    posterior variance, known in advance
    m_{t+1}  = m_t + v_t/(v_t + sigma^2) * (x_t - m_t)      posterior mean, moves with returns

Next year's log stock return is then ``N(m_t, sigma^2 + v_t)``: wider than the
base case by the remaining parameter uncertainty, and centred on the current
belief. The DP state becomes ``(year, wealth, m_t)`` with ``m_t`` on a grid; it
is solved by :func:`gbda.dp.solve_with_state`. The investor learns from the
stock return alone (bond returns would add a little information through the
correlation; ignored).

Why the time dimension matters: over ``T`` years the variance of the
cumulative log return is ``T sigma^2 + T^2 tau^2``. The noise part grows with
``T``, the parameter part with ``T^2``, so parameter uncertainty is a small
share of one year's risk and a large share of a long horizon's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import analysis, dp, metrics, simulate
from .analysis import Investor
from .market import Assumptions, lognormal_params, standard_shocks

Array = np.ndarray

#: Seed offset for the per-path parameter draw (keeps the return shocks equal to the base case).
PARAM_SEED_OFFSET = 7919


@dataclass(frozen=True)
class UncertainStock:
    """Base-case assumptions plus uncertainty ``s`` about the expected stock return.

    Attributes:
        assumptions: Centre of the prior (``mu_stock``) and everything else, taken as known.
        s: Standard deviation of the expected annual stock return (e.g. 0.02).
        n_beliefs: Points on the belief grid for the DP.
        width: The belief grid spans ``m_hat +/- width * tau``.
    """

    assumptions: Assumptions = field(default_factory=Assumptions)
    s: float = 0.02
    n_beliefs: int = 13
    width: float = 3.0

    @property
    def stock_log(self) -> tuple[float, float]:
        """``(m_hat, sigma)`` of the stock's log return at the prior centre."""
        return lognormal_params(self.assumptions.mu_stock, self.assumptions.sigma_stock)

    @property
    def tau(self) -> float:
        """Prior standard deviation of the log-return location ``m``."""
        return self.s / (1.0 + self.assumptions.mu_stock)

    def posterior_var(self, t: int) -> float:
        """Variance of the belief about ``m`` after ``t`` years of returns."""
        if self.tau == 0:
            return 0.0
        sigma = self.stock_log[1]
        return 1.0 / (1.0 / self.tau**2 + t / sigma**2)

    def belief_grid(self) -> Array:
        """Grid of beliefs ``m_t`` used by the DP (a single point when ``s = 0``)."""
        m_hat = self.stock_log[0]
        if self.tau == 0:
            return np.array([m_hat])
        return m_hat + self.tau * np.linspace(-self.width, self.width, self.n_beliefs)

    # ---- the DP --------------------------------------------------------------- #
    def stepper(self, m: int, seed: int) -> "BeliefSteps":
        """Scenario generator for :func:`gbda.dp.solve_with_state`."""
        return BeliefSteps(self, *standard_shocks(m, seed, moment_match=True))

    def solve(
        self,
        investor: Investor,
        m: int = 2000,
        seed: int = analysis.DP_SEED,
        n_grid: int = analysis.GRID_POINTS,
    ) -> dp.StateDPSolution:
        """Goal-probability policy for every (year, wealth, belief)."""
        return dp.solve_with_state(
            self.stepper(m, seed),
            self.belief_grid().size,
            analysis.ACTIONS,
            analysis.wealth_grid(investor, n_grid),
            investor.goal,
            investor.horizon,
            investor.contribution,
            zero_tol=analysis.GOAL_ZERO_TOL,
        )

    # ---- the world and the beliefs along it ----------------------------------- #
    def paths(self, n_paths: int, horizon: int, seed: int = analysis.SIM_SEED) -> tuple[Array, Array, Array]:
        """Return paths in which each path has its own stock parameter.

        Returns:
            ``(stock gross returns, bond gross returns, mu of each path)``; the
            return arrays have shape ``(n_paths, horizon)``.
        """
        a = self.assumptions
        m_hat, sigma = self.stock_log
        mb, sb = lognormal_params(a.mu_bond, a.sigma_bond)
        z1, z2 = standard_shocks((n_paths, horizon), seed)
        m = m_hat + self.tau * np.random.default_rng(seed + PARAM_SEED_OFFSET).standard_normal(n_paths)
        r_s = np.exp(m[:, None] + sigma * z1)
        r_b = np.exp(mb + sb * (a.rho * z1 + np.sqrt(1.0 - a.rho * a.rho) * z2))
        return r_s, r_b, np.expm1(m + 0.5 * sigma**2)

    def beliefs(self, r_s: Array) -> Array:
        """Belief ``m_t`` at the start of each year along each path, shape ``(N, T)``."""
        m_hat, sigma = self.stock_log
        n_paths, horizon = r_s.shape
        out = np.empty((n_paths, horizon))
        b = np.full(n_paths, m_hat)
        for t in range(horizon):
            out[:, t] = b
            v = self.posterior_var(t)
            b = b + v / (v + sigma**2) * (np.log(r_s[:, t]) - b)
        return out

    def belief_index(self, beliefs: Array) -> Array:
        """Nearest point of the belief grid."""
        grid = self.belief_grid()
        if grid.size == 1:
            return np.zeros(np.shape(beliefs), dtype=int)
        step = grid[1] - grid[0]
        return np.clip(np.rint((beliefs - grid[0]) / step), 0, grid.size - 1).astype(int)


@dataclass(frozen=True)
class BeliefSteps:
    """``step(t, b)`` for the learning DP: predictive returns at belief ``b`` and where the belief goes.

    The same standard normal shocks ``(e1, e2)`` are used for every year and
    belief (common random numbers), as the base DP does.
    """

    model: UncertainStock
    e1: Array
    e2: Array

    def __call__(self, t: int, b: int) -> dp.StateStep:
        a = self.model.assumptions
        grid = self.model.belief_grid()
        _, sigma = self.model.stock_log
        mb, sb = lognormal_params(a.mu_bond, a.sigma_bond)
        v = self.model.posterior_var(t)
        if v == 0:  # no parameter uncertainty: exactly the base-case scenarios
            sd, c = sigma, a.rho
        else:
            sd = np.sqrt(sigma**2 + v)
            c = a.rho * sigma / sd  # correlation of bonds with the predictive stock shock
        x_s = grid[b] + sd * self.e1
        x_b = mb + sb * (c * self.e1 + np.sqrt(1.0 - c * c) * self.e2)
        scenarios = dp.ReturnScenarios.equally_weighted(np.exp(x_s), np.exp(x_b))
        if grid.size == 1:
            zeros = np.zeros(self.e1.size, dtype=int)
            return dp.StateStep(scenarios, zeros, zeros, np.zeros(self.e1.size))
        nxt = grid[b] + v / (v + sigma**2) * (x_s - grid[b])
        pos = np.clip((nxt - grid[0]) / (grid[1] - grid[0]), 0.0, grid.size - 1.0)
        lo = np.minimum(np.floor(pos).astype(int), grid.size - 2)
        return dp.StateStep(scenarios, lo, lo + 1, pos - lo)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def learning_policy(model: UncertainStock, solution: dp.StateDPSolution, r_s: Array) -> simulate.Policy:
    """Forward policy that tracks each path's belief and looks up the (year, wealth, belief) table."""
    idx = model.belief_index(model.beliefs(r_s))

    def pol(t: int, wealth: Array) -> Array:
        return solution.action(t, wealth, idx[:, t])

    return pol


def case(investor: Investor, model: UncertainStock, n_paths: int = analysis.SIM_PATHS, m: int = 2000) -> dict[str, Any]:
    """Score the learning policy and a policy that ignores the uncertainty, for one ``s``.

    "Ignores the uncertainty" is the same solver with ``s = 0`` (identical to
    the base-case DP at this resolution). Both are scored on the same paths, in
    which every path has its own expected stock return drawn with this ``s``.
    """
    r_s, r_b, mu = model.paths(n_paths, investor.horizon)
    ignore = UncertainStock(model.assumptions, 0.0)
    sol_learn = model.solve(investor, m)
    sol_ignore = ignore.solve(investor, m)
    pols = {
        "Learns the expected return": learning_policy(model, sol_learn, r_s),
        "Ignores the uncertainty": learning_policy(ignore, sol_ignore, r_s),
        "Fixed 100% stock": simulate.fixed_mix(1.0),
        "Fixed 60/40": simulate.fixed_mix(0.6),
    }
    sims = {k: simulate.simulate(p, r_s, r_b, investor.w0, investor.contribution) for k, p in pols.items()}
    hit_ignore = sims["Ignores the uncertainty"].terminal >= investor.goal
    deciles = np.quantile(mu, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(deciles, mu, side="right") - 1, 0, 9)
    rows: dict[str, dict[str, float]] = {}
    for name, sim in sims.items():
        hit = sim.terminal >= investor.goal
        p, se = metrics.success_probability(sim.terminal, investor.goal)
        diff, se_diff, _ = metrics.paired_difference(hit, hit_ignore)
        if np.ptp(mu) == 0:  # s = 0: every path has the same parameter
            by_decile = np.full(10, p)
        else:
            by_decile = np.array([hit[bins == k].mean() for k in range(10)])
        rows[name] = {
            "success": p,
            "success_se": se,
            "vs_ignore": diff,
            "vs_ignore_se": se_diff,
            "avg_stock_weight": float(sim.weights.mean()),
            "success_lowest_mu_decile": float(by_decile[0]),
            "success_highest_mu_decile": float(by_decile[-1]),
        }
    centre = model.belief_grid().size // 2
    rows["Learns the expected return"]["dp_value"] = sol_learn.value_at(0, investor.w0, centre)
    rows["Ignores the uncertainty"]["dp_value"] = sol_ignore.value_at(0, investor.w0, 0)
    return {
        "s": model.s,
        "rows": rows,
        "mu_deciles": deciles.tolist(),
        "weights_by_year": {k: sim.weights.mean(axis=0).tolist() for k, sim in sims.items()},
        # For the figure: stock weight by wealth in year 10 at the lowest, central and highest belief.
        "policy_year10": {
            "grid": sol_learn.grid.tolist(),
            "beliefs_mu": np.expm1(model.belief_grid() + 0.5 * model.stock_log[1] ** 2).tolist(),
            "weights": sol_learn.policy[10].tolist(),
        },
    }


def variance_share(s: float, sigma: float, horizons: tuple[int, ...] = (1, 5, 10, 20, 30), mu: float = 0.07) -> dict[int, float]:
    """Share of the variance of the cumulative log stock return that comes from not knowing ``mu``.

    ``T^2 tau^2 / (T sigma_log^2 + T^2 tau^2)`` with ``tau = s / (1 + mu)``.
    """
    _, sig = lognormal_params(mu, sigma)
    tau = s / (1.0 + mu)
    return {T: T * T * tau * tau / (T * sig * sig + T * T * tau * tau) for T in horizons}
