"""A two-asset coin-flip market where the answers can be worked out by hand.

Each period a fair coin is flipped and both assets react to the same flip:

    =======  =====  =====
    asset    heads  tails
    =======  =====  =====
    asset 1  1.40   0.70
    asset 2  0.80   1.15
    =======  =====  =====

With weight ``u`` in asset 1 the portfolio earns ``0.8 + 0.6u`` on heads and
``1.15 - 0.45u`` on tails. At ``u = 1/3`` both equal 1: a riskless portfolio.

Known results used to check the engine:

* log utility: the optimal policy is the fixed mix ``u* = 11/18``;
* power utility: also a fixed mix, with a closed form (:func:`optimal_weight`);
* goal probability: the policy is *not* a fixed mix, and beats the best fixed mix by a wide margin.

The four uses of Monte Carlo exercised here are (A) evaluating a given policy,
(B) optimising a fixed mix by simulation, (C) replacing the exact expectation in
the DP by a sample average, and (D) verifying a DP policy out of sample.

Run ``python -m gbda.binomial`` to print the comparison against reference numbers
and write the validation figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from . import dp, metrics, plots, simulate

Array = np.ndarray

HEADS = (1.40, 0.80)  # gross returns of (asset 1, asset 2) on heads
TAILS = (0.70, 1.15)  # ... on tails
RISKLESS_WEIGHT = 1.0 / 3.0
LOG_OPTIMAL_WEIGHT = 11.0 / 18.0

#: Reference values the code has to reproduce.
REFERENCE_POWER_WEIGHTS = {0.5: 0.889, 1.0: 0.611, 2.0: 0.472, 5.0: 0.388, 10.0: 0.361}
REFERENCE_BEST_FIXED_GOAL = 0.412
REFERENCE_DP_GOAL = 0.765


# --------------------------------------------------------------------------- #
# The market
# --------------------------------------------------------------------------- #
def portfolio_returns(u: float | Array) -> tuple[Array, Array]:
    """Gross portfolio return on heads and on tails for weight ``u`` in asset 1."""
    u = np.asarray(u, dtype=float)
    return HEADS[1] + (HEADS[0] - HEADS[1]) * u, TAILS[1] + (TAILS[0] - TAILS[1]) * u


def exact_scenarios() -> dp.ReturnScenarios:
    """The two outcomes with probability 1/2 each (exact expectation)."""
    return dp.ReturnScenarios(
        r_a=np.array([HEADS[0], TAILS[0]]),
        r_b=np.array([HEADS[1], TAILS[1]]),
        probs=np.array([0.5, 0.5]),
    )


def flip(shape: int | tuple[int, ...], seed: int) -> Array:
    """Fair coin flips; ``True`` means heads."""
    return np.random.default_rng(seed).random(shape) < 0.5


def returns_from_flips(heads: Array) -> tuple[Array, Array]:
    """Gross returns of asset 1 and asset 2 for an array of coin flips."""
    return np.where(heads, HEADS[0], TAILS[0]), np.where(heads, HEADS[1], TAILS[1])


def sampled_scenarios(m: int, seed: int) -> dp.ReturnScenarios:
    """Monte Carlo expectation from ``m`` simulated coin flips.

    Averaging over ``m`` draws that take only two values is the same as weighting
    the two outcomes by their sample frequencies, so the draws are collapsed to
    that form (identical result, far less work).
    """
    share_heads = float(flip(m, seed).mean())
    return dp.ReturnScenarios(
        r_a=np.array([HEADS[0], TAILS[0]]),
        r_b=np.array([HEADS[1], TAILS[1]]),
        probs=np.array([share_heads, 1.0 - share_heads]),
    )


# --------------------------------------------------------------------------- #
# Closed-form answers
# --------------------------------------------------------------------------- #
def expected_log_growth(u: float | Array, base: float = np.e) -> Array:
    """Exact ``E[log R_u]`` per period (natural log by default; pass ``base=10`` for log10)."""
    h, t = portfolio_returns(u)
    return (0.5 * np.log(h) + 0.5 * np.log(t)) / np.log(base)


def optimal_weight(gamma: float) -> float:
    """Optimal fixed weight in asset 1 under CRRA utility with risk aversion ``gamma``.

    First-order condition: ``0.6 * h^-gamma = 0.45 * t^-gamma``, i.e.
    ``t / h = 0.75^(1/gamma)``, solved for ``u`` and clipped to ``[0, 1]``.
    ``gamma = 1`` (log) gives ``11/18``; as ``gamma`` grows the weight tends to 1/3.
    """
    k = 0.75 ** (1.0 / gamma)
    return float(np.clip((1.15 - 0.8 * k) / (0.45 + 0.6 * k), 0.0, 1.0))


def optimal_weight_numeric(gamma: float) -> float:
    """Same as :func:`optimal_weight`, by direct numerical maximisation (independent check)."""

    def neg_utility(u: float) -> float:
        h, t = portfolio_returns(u)
        if gamma == 1:
            return -float(0.5 * np.log(h) + 0.5 * np.log(t))
        return -float((0.5 * h ** (1 - gamma) + 0.5 * t ** (1 - gamma)) / (1 - gamma))

    res = optimize.minimize_scalar(neg_utility, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10})
    return float(res.x)


# --------------------------------------------------------------------------- #
# Use A: evaluate a given policy by forward simulation
# --------------------------------------------------------------------------- #
def simulate_fixed_mix_log10(u: float, n_paths: int, n_flips: int, seed: int) -> Array:
    """log10 wealth paths of a fixed mix (start at 0), shape ``(n_paths, n_flips + 1)``."""
    heads = flip((n_paths, n_flips), seed)
    h, t = portfolio_returns(u)
    step = np.where(heads, np.log10(h), np.log10(t))
    out = np.zeros((n_paths, n_flips + 1))
    np.cumsum(step, axis=1, out=out[:, 1:])
    return out


def exact_share_ahead(u: float, n_flips: int) -> float:
    """Exact probability that a fixed mix ends above its starting wealth."""
    h, t = (float(x) for x in portfolio_returns(u))
    lh, lt = np.log(h), np.log(t)
    if lh == lt:
        return float(lh > 0)
    k = np.arange(n_flips + 1)
    ahead = k * lh + (n_flips - k) * lt > 0
    return float(stats.binom.pmf(k, n_flips, 0.5)[ahead].sum())


def summarize_log10_paths(paths: Array) -> dict[str, float]:
    """Median, 5%/95% quantiles of final log10 wealth, share of paths that end ahead,
    and the simulated mean log10 growth per period with its standard error."""
    final = paths[:, -1]
    n_flips = paths.shape[1] - 1
    per_period = final / n_flips
    return {
        "median": float(np.median(final)),
        "p05": float(np.quantile(final, 0.05)),
        "p95": float(np.quantile(final, 0.95)),
        "share_ahead": float((final > 0).mean()),
        "growth_per_period": float(per_period.mean()),
        "growth_se": float(per_period.std(ddof=1) / np.sqrt(final.size)),
    }


# --------------------------------------------------------------------------- #
# Use B: find the best fixed mix by simulation
# --------------------------------------------------------------------------- #
def mc_best_fixed_weight(
    m: int,
    seed: int,
    weights: Array | None = None,
    common_random_numbers: bool = True,
) -> tuple[float, Array]:
    """Estimate ``argmax_u E[ln R_u]`` from ``m`` simulated flips.

    Args:
        m: Number of coin flips per weight.
        seed: Random seed.
        weights: Candidate weights (default ``0, 0.01, ..., 1``).
        common_random_numbers: If True every weight is scored on the same flips.
            If False each weight gets its own flips, and the comparison between
            neighbouring weights is swamped by sampling noise.

    Returns:
        ``(best weight, estimated objective for every candidate)``.
    """
    weights = np.linspace(0.0, 1.0, 101) if weights is None else np.asarray(weights)
    h, t = portfolio_returns(weights)
    if common_random_numbers:
        share_heads = flip(m, seed).mean()
        est = share_heads * np.log(h) + (1.0 - share_heads) * np.log(t)
    else:
        share_heads = np.random.default_rng(seed).binomial(m, 0.5, size=weights.size) / m
        est = share_heads * np.log(h) + (1.0 - share_heads) * np.log(t)
    return float(weights[int(np.argmax(est))]), est


# --------------------------------------------------------------------------- #
# Use C: DP with exact or sampled expectations (utility objectives)
# --------------------------------------------------------------------------- #
def solve_utility_dp(
    gamma: float,
    horizon: int = 10,
    scenarios: dp.ReturnScenarios | None = None,
    weights: Array | None = None,
    grid: Array | None = None,
) -> dp.DPSolution:
    """DP for log (``gamma = 1``) or power utility in the coin-flip market.

    With ``scenarios=None`` the expectation is exact; pass :func:`sampled_scenarios`
    for the Monte Carlo version.
    """
    scenarios = exact_scenarios() if scenarios is None else scenarios
    weights = np.linspace(0.0, 1.0, 1001) if weights is None else weights
    grid = dp.log_grid(0.05, 20.0, 60) if grid is None else grid
    objective = dp.log_objective() if gamma == 1 else dp.power_objective(gamma)
    return dp.solve(scenarios, weights, grid, objective, horizon)


# --------------------------------------------------------------------------- #
# Use D: goal probability
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GoalProblem:
    """Maximise ``P(W_T >= goal)`` starting from ``w0`` over ``horizon`` flips."""

    horizon: int = 20
    w0: float = 1.0
    goal: float = 1.5


def best_fixed_mix_goal(problem: GoalProblem, weights: Array | None = None) -> tuple[float, float]:
    """Best fixed weight for the goal problem and its exact success probability."""
    weights = np.linspace(0.0, 1.0, 1001) if weights is None else weights
    k = np.arange(problem.horizon + 1)
    pmf = stats.binom.pmf(k, problem.horizon, 0.5)
    h, t = portfolio_returns(weights[:, None])
    final = problem.w0 * h**k * t ** (problem.horizon - k)
    prob = ((final >= problem.goal - 1e-12) * pmf).sum(axis=1)
    best = int(np.argmax(prob))
    return float(weights[best]), float(prob[best])


def complete_market_bound(problem: GoalProblem) -> float:
    """Upper bound on the success probability if any payoff could be bought.

    The market is complete with state price ``q = 3/7`` per head (and zero
    interest, from the riskless mix). Without the ``0 <= u <= 1`` limit the best
    plan buys a payoff of ``goal`` on the cheapest paths per unit of probability
    (most heads first) until the budget ``w0`` runs out. The constrained DP
    cannot do better than this.
    """
    q = 3.0 / 7.0
    n = problem.horizon
    budget = problem.w0 / problem.goal
    prob = 0.0
    for k in range(n, -1, -1):
        price = stats.binom.pmf(k, n, q)  # cost of paying 1 on every path with k heads
        real = stats.binom.pmf(k, n, 0.5)
        if price <= budget:
            prob += real
            budget -= price
        else:
            prob += real * budget / price
            break
    return float(prob)


def solve_goal_dp(
    problem: GoalProblem,
    weight_step: float = 0.01,
    n_grid: int = 400,
    interp: str = "linear",
    include_riskless: bool = False,
    scenarios: dp.ReturnScenarios | None = None,
) -> dp.DPSolution:
    """DP for the goal problem.

    Outcomes are discrete, so the value function is a staircase and the result
    depends on how fine the weight and wealth grids are. ``interp="floor"`` gives
    a guaranteed lower bound, ``"ceil"`` an upper bound.
    """
    weights = np.round(np.arange(0.0, 1.0 + 1e-9, weight_step), 10)
    if include_riskless:
        weights = np.unique(np.append(weights, RISKLESS_WEIGHT))
    grid = dp.log_grid(0.01, 100.0, n_grid, include=problem.goal)
    return dp.solve(
        exact_scenarios() if scenarios is None else scenarios,
        weights,
        grid,
        dp.goal_objective(problem.goal),
        problem.horizon,
        interp=interp,  # type: ignore[arg-type]
    )


def dp_value_at_start(solution: dp.DPSolution, w0: float) -> float:
    """DP success probability at the starting wealth (read the same way the solver reads it)."""
    if solution.lookup == "nearest":
        return solution.value_at(0, w0)
    return float(solution.value[0][solution.grid_index(w0)])


def verify_goal_mc(solution: dp.DPSolution, problem: GoalProblem, n_paths: int, seed: int) -> tuple[float, float]:
    """Run the DP policy on fresh coin flips; return success probability and standard error."""
    r1, r2 = returns_from_flips(flip((n_paths, problem.horizon), seed))
    sim = simulate.simulate(simulate.dp_policy(solution), r1, r2, problem.w0)
    return metrics.success_probability(sim.terminal, problem.goal - 1e-12)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _figure_paths(out: Path, n_paths: int, n_flips: int) -> dict[str, dict[str, float]]:
    """Use A figure: log10 wealth bands for three fixed mixes. Returns the summary table."""
    import matplotlib.pyplot as plt

    cases = {"All in asset 1": (1.0, plots.ORANGE), "All in asset 2": (0.0, plots.AQUA), "Fixed mix 11/18": (LOG_OPTIMAL_WEIGHT, plots.BLUE)}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(n_flips + 1)
    table: dict[str, dict[str, float]] = {}
    for name, (u, color) in cases.items():
        paths = simulate_fixed_mix_log10(u, n_paths, n_flips, seed=11)
        lo, mid, hi = np.quantile(paths, [0.05, 0.5, 0.95], axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(x, mid, color=color, label=name)
        ax.text(n_flips * 1.01, mid[-1], name, color=plots.INK_2, fontsize=9, va="center")
        table[name] = summarize_log10_paths(paths) | {
            "exact_growth": float(expected_log_growth(u, base=10)),
            "exact_share_ahead": exact_share_ahead(u, n_flips),
        }
    ax.axhline(0, color=plots.AXIS, lw=1)
    ax.set_xlim(0, n_flips * 1.22)
    ax.set_xticks(np.arange(0, n_flips + 1, 1000))
    ax.set_xlabel("Number of coin flips")
    ax.set_ylabel("log10 of wealth")
    ax.set_title(f"Fixed mixes over {n_flips:,} flips: median and 5–95% band of {n_paths:,} paths")
    ax.legend(loc="lower left")
    plots.save(fig, out / "binomial_paths.png")
    return table


def _figure_mc_optimisation(out: Path) -> dict[int, tuple[float, float]]:
    """Use B figure: common random numbers vs independent samples, and error vs sample size."""
    import matplotlib.pyplot as plt

    weights = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    m = 1000
    _, est_crn = mc_best_fixed_weight(m, seed=5, common_random_numbers=True)
    _, est_ind = mc_best_fixed_weight(m, seed=5, common_random_numbers=False)
    ax.plot(weights, est_ind * 1000, color=plots.ORANGE, lw=1.2, label="Separate flips per weight")
    ax.plot(weights, est_crn * 1000, color=plots.BLUE, label="Same flips for all weights")
    ax.plot(weights, expected_log_growth(weights) * 1000, color=plots.INK, lw=1.2, ls=(0, (4, 3)), label="Exact")
    ax.axvline(LOG_OPTIMAL_WEIGHT, color=plots.AXIS, lw=1)
    ax.text(LOG_OPTIMAL_WEIGHT + 0.01, ax.get_ylim()[0], "11/18", color=plots.INK_2, fontsize=9, va="bottom")
    ax.set_xlabel("Weight in asset 1")
    ax.set_ylabel("Estimated E[ln R] per period (‰)")
    ax.set_title(f"Scoring every weight with M = {m:,} flips")
    ax.legend(loc="upper left")

    ax = axes[1]
    sizes = [100, 1000, 10_000, 100_000]
    spread: dict[int, tuple[float, float]] = {}
    rng = np.random.default_rng(0)
    for i, m in enumerate(sizes):
        picks = np.array([mc_best_fixed_weight(m, seed=1000 + 37 * r + m)[0] for r in range(200)])
        spread[m] = (float(np.mean(np.abs(picks - LOG_OPTIMAL_WEIGHT))), float(picks.std()))
        ax.scatter(i + rng.uniform(-0.18, 0.18, picks.size), picks, s=14, color=plots.BLUE, alpha=0.35, lw=0)
        ax.text(i, 1.03, f"mean abs. error {spread[m][0]:.3f}", ha="center", color=plots.INK_2, fontsize=8.5)
    ax.axhline(LOG_OPTIMAL_WEIGHT, color=plots.INK, lw=1.2, ls=(0, (4, 3)))
    ax.set_xticks(range(len(sizes)), [f"M = {m:,}" for m in sizes])
    ax.set_ylim(-0.02, 1.1)
    ax.set_ylabel("Estimated best weight (200 repeats)")
    ax.set_title("Estimated best weight closes in on 11/18 as M grows")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    plots.save(fig, out / "binomial_mc_optimisation.png")
    return spread


def _figure_goal_policy(out: Path, solution: dp.DPSolution, problem: GoalProblem) -> None:
    """Use D figure: the goal policy over time and wealth."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    mesh = plots.policy_heatmap(
        ax, solution.grid, solution.policy, "Goal policy in the coin-flip market: weight in asset 1",
        goal=problem.goal, w_range=(0.3, 3.0),
    )
    ax.set_ylabel("Wealth (start = 1)")
    ax.set_yticks([0.3, 0.5, 1.0, 1.5, 2.0, 3.0], ["0.3", "0.5", "1.0", "1.5", "2.0", "3.0"])
    ax.minorticks_off()
    ax.set_xlabel("Period")
    ax.set_xticks(range(0, problem.horizon + 1, 5))
    cb = fig.colorbar(mesh, ax=ax, ticks=np.linspace(0, 1, 6))
    cb.set_label("Weight in asset 1")
    cb.outline.set_visible(False)
    plots.save(fig, out / "binomial_goal_policy.png")


def main(figure_dir: str | Path = "figures/validation") -> None:
    """Print every check against its reference value and write the validation figures."""
    plots.apply_style()
    out = Path(figure_dir)
    line = "-" * 78

    print(line + "\nFixed-mix optimum by risk aversion (reference vs closed form vs numeric)\n" + line)
    for gamma, ref in REFERENCE_POWER_WEIGHTS.items():
        print(f"  gamma = {gamma:>4g}   reference {ref:.3f}   closed form {optimal_weight(gamma):.4f}   "
              f"numeric {optimal_weight_numeric(gamma):.4f}")
    print(f"  expected log10 growth per period (‰): u=1 {expected_log_growth(1, 10) * 1e3:+.2f}, "
          f"u=0 {expected_log_growth(0, 10) * 1e3:+.2f}, u=11/18 {expected_log_growth(LOG_OPTIMAL_WEIGHT, 10) * 1e3:+.2f}")

    print(line + "\nUse A. Forward simulation of fixed mixes (2,000 paths x 4,000 flips, log10 wealth)\n" + line)
    table = _figure_paths(out, n_paths=2000, n_flips=4000)
    for name, row in table.items():
        print(f"  {name:<16} median {row['median']:+6.1f}   5–95% {row['p05']:+6.1f} to {row['p95']:+6.1f}   "
              f"ends ahead {row['share_ahead']:6.1%} (exact {row['exact_share_ahead']:.1%})   "
              f"growth/period {row['growth_per_period'] * 1e3:+.2f}‰ ± {row['growth_se'] * 1e3:.2f} (exact {row['exact_growth'] * 1e3:+.2f}‰)")

    print(line + "\nUse B. Best fixed mix found by simulation (target 11/18 = 0.611)\n" + line)
    spread = _figure_mc_optimisation(out)
    for m, (err, sd) in spread.items():
        print(f"  M = {m:>7,}   mean abs. error {err:.3f}   std of estimate {sd:.3f}")

    print(line + "\nUse C. DP policy: exact vs sampled expectation (10 periods)\n" + line)
    for gamma in (1.0, 2.0, 5.0):
        exact = solve_utility_dp(gamma)
        flat = float(exact.policy.max() - exact.policy.min())
        row = f"  gamma = {gamma:g}: exact DP weight {exact.policy[0, 0]:.3f} (spread across time and wealth {flat:.1e})"
        for m in (100, 1000, 10_000, 100_000):
            sampled = solve_utility_dp(gamma, scenarios=sampled_scenarios(m, seed=3))
            row += f" | M={m:,}: {sampled.policy[0, 0]:.3f}"
        print(row)

    print(line + "\nUse D. Goal probability: T = 20, W0 = 1, goal = 1.5\n" + line)
    problem = GoalProblem()
    u_fixed, p_fixed = best_fixed_mix_goal(problem)
    print(f"  best fixed mix: u = {u_fixed:.2f}, success {p_fixed:.1%}   (reference about {REFERENCE_BEST_FIXED_GOAL:.1%})")
    coarse = solve_goal_dp(problem, weight_step=0.01, n_grid=400, interp="linear")
    p_mc, se = verify_goal_mc(coarse, problem, n_paths=100_000, seed=21)
    print(f"  DP, weight step 0.01, 400 wealth points: {dp_value_at_start(coarse, problem.w0):.1%}   "
          f"(reference about {REFERENCE_DP_GOAL:.1%});  forward check on 100,000 paths: {p_mc:.2%} ± {se:.2%}")
    print("  refining the grids (lower bound / upper bound / forward check of the lower-bound policy):")
    fine = coarse
    for step, n_grid in ((0.01, 2000), (0.002, 8000), (0.001, 16000)):
        low = solve_goal_dp(problem, step, n_grid, "floor", include_riskless=True)
        high = solve_goal_dp(problem, step, n_grid, "ceil", include_riskless=True)
        p_mc, se = verify_goal_mc(low, problem, n_paths=100_000, seed=21)
        print(f"    weight step {step:<5} {n_grid:>6,} points: {dp_value_at_start(low, problem.w0):.2%} / "
              f"{dp_value_at_start(high, problem.w0):.2%} / {p_mc:.2%} ± {se:.2%}")
        fine = low
    print(f"  bound if any payoff could be bought (no 0–1 limit on the weight): {complete_market_bound(problem):.1%}")
    _figure_goal_policy(out, fine, problem)
    print(line + f"\nFigures written to {out}/")


if __name__ == "__main__":
    main()
