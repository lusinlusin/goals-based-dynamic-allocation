"""Checks on the stock/bond engine: each one has a known right answer."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from gbda import analysis as an
from gbda import dp, market, metrics, simulate

INVESTOR = an.Investor()
BASE_WORLD = an.World()


# ---- Policy shape when there are no contributions --------------------------- #
@pytest.mark.parametrize("objective", ["log", "power:2", "power:5", "power:10"])
def test_no_contributions_gives_a_constant_stock_weight(objective: str) -> None:
    """Without yearly saving, CRRA policies do not depend on wealth or on time."""
    investor = replace(INVESTOR, contribution=0.0)
    solution = an.solve_policy(investor, BASE_WORLD, objective, m=2000)
    assert solution.policy.min() == solution.policy.max()


@pytest.mark.parametrize("objective,gamma", [("log", 1.0), ("power:5", 5.0)])
def test_no_contribution_weight_equals_one_period_grid_search(objective: str, gamma: float) -> None:
    """The DP choice equals a direct search over the 11 weights of the one-period problem."""
    investor = replace(INVESTOR, contribution=0.0)
    scenarios = BASE_WORLD.scenarios(2000)
    port = scenarios.portfolio(an.ACTIONS)
    score = np.log(port).mean(axis=1) if gamma == 1 else (port ** (1 - gamma)).mean(axis=1) / (1 - gamma)
    solution = an.solve_policy(investor, BASE_WORLD, objective, m=2000)
    assert np.all(solution.policy[-1] == an.ACTIONS[int(np.argmax(score))])
    assert np.all(solution.policy[0] == an.ACTIONS[int(np.argmax(score))])


def test_contributions_make_low_wealth_investors_hold_more_stock() -> None:
    """Future savings act like a bond holding, so the weight falls as wealth rises."""
    solution = an.solve_policy(INVESTOR, BASE_WORLD, "power:5", m=2000)
    row = solution.policy[5]
    band = (solution.grid > 2e4) & (solution.grid < 2e6)
    assert np.all(np.diff(row[band]) <= 1e-12)
    assert row[band][0] > row[band][-1]


# ---- Goal objective ------------------------------------------------------------ #
def test_far_above_goal_means_all_bonds_and_near_certain_success() -> None:
    investor = replace(INVESTOR, w0=5_000_000.0)
    solution = an.solve_policy(investor, BASE_WORLD, "goal", m=2000)
    assert solution.action(0, investor.w0) == 0.0
    sim = an.run(simulate.dp_policy(solution), investor, BASE_WORLD.paths(20_000, investor.horizon))
    assert metrics.success_probability(sim.terminal, investor.goal)[0] > 0.999
    assert sim.weights.mean() < 0.01


def test_far_below_goal_means_all_stock() -> None:
    solution = an.solve_policy(INVESTOR, BASE_WORLD, "goal", m=2000)
    assert solution.action(INVESTOR.horizon - 1, 100_000.0) == 1.0
    assert solution.action(INVESTOR.horizon - 3, 20_000.0) == 1.0


def test_goal_policy_cuts_risk_once_ahead() -> None:
    solution = an.solve_policy(INVESTOR, BASE_WORLD, "goal", m=2000)
    late = INVESTOR.horizon - 2
    assert solution.action(late, 300_000.0) > solution.action(late, 520_000.0)


def test_value_function_is_a_probability_and_rises_with_wealth() -> None:
    solution = an.solve_policy(INVESTOR, BASE_WORLD, "goal", m=2000)
    assert solution.value.min() >= 0.0 and solution.value.max() <= 1.0
    assert np.all(np.diff(solution.value[0]) >= -1e-9)


def test_dp_estimate_agrees_with_forward_simulation() -> None:
    """Different seeds for the DP scenarios and the forward paths: an out-of-sample check."""
    assert an.DP_SEED != an.SIM_SEED
    result = an.goal_case(INVESTOR, BASE_WORLD)
    assert abs(result["dp_value"] - result["success"]) < 0.02
    assert result["success"] >= result["best_fixed_success"]


def test_success_never_falls_when_saving_more() -> None:
    probs = []
    for c in (0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0):
        res = an.goal_case(replace(INVESTOR, contribution=c), BASE_WORLD, n_paths=20_000, m=2000)
        probs.append(res["success"])
    assert all(b >= a for a, b in zip(probs, probs[1:]))
    assert probs[-1] > probs[0]


def test_stock_cap_cannot_help() -> None:
    free = an.goal_case(INVESTOR, BASE_WORLD, n_paths=20_000, m=2000)["dp_value"]
    capped = an.goal_case(INVESTOR, BASE_WORLD, max_stock=0.4, n_paths=20_000, m=2000)["dp_value"]
    assert capped <= free


# ---- Simulation ------------------------------------------------------------------ #
def test_zero_volatility_reproduces_deterministic_compounding() -> None:
    flat = market.Assumptions(mu_stock=0.07, sigma_stock=0.0, mu_bond=0.03, sigma_bond=0.0, rho=0.0)
    paths = market.sample_returns(flat, (100, 20), seed=3)
    sim = an.run(simulate.fixed_mix(0.6), INVESTOR, paths)
    w = INVESTOR.w0
    for _ in range(INVESTOR.horizon):
        w = w * (1 + 0.6 * 0.07 + 0.4 * 0.03) + INVESTOR.contribution
    assert np.allclose(sim.terminal, w, rtol=1e-12)


def test_glide_path_weights_and_required_return() -> None:
    policy = simulate.glide_path(0.9, 0.3, 20)
    assert policy(0, np.ones(2))[0] == pytest.approx(0.9)
    assert policy(19, np.ones(2))[0] == pytest.approx(0.3)
    r = INVESTOR.required_return()
    w = INVESTOR.w0
    for _ in range(INVESTOR.horizon):
        w = w * (1 + r) + INVESTOR.contribution
    assert w == pytest.approx(INVESTOR.goal, rel=1e-9)


# ---- Return generators ------------------------------------------------------------- #
@pytest.mark.parametrize("dist", ["lognormal", "student_t"])
def test_generators_hit_the_assumed_mean_volatility_and_correlation(dist: str) -> None:
    rs, rb = market.sample_returns(market.BASE, 1_000_000, seed=7, dist=dist)
    assert rs.min() > 0 and rb.min() > 0
    assert rs.mean() - 1 == pytest.approx(0.07, abs=1e-3)
    assert rs.std() == pytest.approx(0.16, abs=2e-3)
    assert rb.mean() - 1 == pytest.approx(0.03, abs=5e-4)
    assert rb.std() == pytest.approx(0.05, abs=1e-3)
    assert np.corrcoef(np.log(rs), np.log(rb))[0, 1] == pytest.approx(0.2, abs=0.01)


def test_student_t_has_fatter_tails_than_lognormal() -> None:
    normal, _ = market.sample_returns(market.BASE, 1_000_000, seed=7)
    fat, _ = market.sample_returns(market.BASE, 1_000_000, seed=7, dist="student_t", df=5)
    assert stats.kurtosis(np.log(fat)) > stats.kurtosis(np.log(normal)) + 1.0
    assert np.mean(fat < 0.65) > 2 * np.mean(normal < 0.65)  # years worse than -35%


def test_moment_matched_scenarios_have_exact_underlying_moments() -> None:
    rs, rb = market.sample_returns(market.BASE, 4000, seed=1, moment_match=True)
    ms, ss = market.lognormal_params(0.07, 0.16)
    zs = (np.log(rs) - ms) / ss
    assert zs.mean() == pytest.approx(0.0, abs=1e-12) and zs.std() == pytest.approx(1.0, abs=1e-12)
    assert np.corrcoef(np.log(rs), np.log(rb))[0, 1] == pytest.approx(0.2, abs=1e-10)
    assert rs.mean() - 1 == pytest.approx(0.07, abs=5e-4)


def test_compound_and_arithmetic_conversions_are_inverses() -> None:
    mu = market.arithmetic_from_compound(0.058, 0.16)
    assert mu > 0.058
    assert market.compound_from_arithmetic(mu, 0.16) == pytest.approx(0.058, abs=1e-12)


def test_history_and_block_bootstrap() -> None:
    history = market.load_history()
    assert history.years[0] == 1928 and history.years[-1] == 2025 and history.years.size == 98
    rs, rb = market.block_bootstrap(history, n_paths=500, horizon=20, seed=4, block=5)
    assert rs.shape == rb.shape == (500, 20)
    pairs = set(zip(np.round(history.stock, 6), np.round(history.bond, 6)))
    assert all((a, b) in pairs for a, b in zip(np.round(rs[:20].ravel(), 6), np.round(rb[:20].ravel(), 6)))
    # Inside a block, consecutive draws are consecutive calendar years.
    first = int(np.where(np.isclose(history.stock, rs[0, 0]) & np.isclose(history.bond, rb[0, 0]))[0][0])
    assert rs[0, 1] == pytest.approx(history.stock[(first + 1) % 98])


def test_recentred_history_matches_target_moments_and_keeps_order() -> None:
    history = market.load_history()
    shaped = history.recentred(market.BASE)
    assert shaped.stock.mean() - 1 == pytest.approx(0.07, abs=1e-9)
    assert shaped.stock.std() == pytest.approx(0.16, abs=1e-9)
    assert shaped.bond.mean() - 1 == pytest.approx(0.03, abs=1e-9)
    assert stats.spearmanr(history.stock, shaped.stock)[0] == pytest.approx(1.0)


def test_equilibrium_returns_make_the_market_portfolio_optimal() -> None:
    eq = market.equilibrium_assumptions(stock_market_weight=0.6, risk_aversion=2.5, cash_rate=0.03)
    cov_sb = eq.rho * eq.sigma_stock * eq.sigma_bond
    cov = np.array([[eq.sigma_stock**2, cov_sb], [cov_sb, eq.sigma_bond**2]])
    excess = np.array([eq.mu_stock - 0.03, eq.mu_bond - 0.03])
    assert np.allclose(np.linalg.solve(cov, excess) / 2.5, [0.6, 0.4])


# ---- Metrics ---------------------------------------------------------------------- #
def test_success_probability_and_standard_error() -> None:
    terminal = np.array([400.0] * 25 + [600.0] * 75)
    p, se = metrics.success_probability(terminal, 500.0)
    assert p == 0.75 and se == pytest.approx(np.sqrt(0.75 * 0.25 / 100))


def test_certainty_equivalent_properties() -> None:
    assert metrics.certainty_equivalent(np.full(10, 3e5), 5.0) == pytest.approx(3e5)
    wealth = np.random.default_rng(0).lognormal(np.log(4e5), 0.5, 50_000)
    ces = [metrics.certainty_equivalent(wealth, g) for g in (1.0, 2.0, 5.0, 10.0)]
    assert all(a > b for a, b in zip(ces, ces[1:]))  # more risk-averse -> lower CE
    assert ces[0] < wealth.mean() and ces[-1] > wealth.min()
    assert ces[0] == pytest.approx(np.exp(np.log(wealth).mean()))


def test_paired_standard_error_is_smaller_on_common_paths() -> None:
    paths = BASE_WORLD.paths(20_000, INVESTOR.horizon)
    a = an.run(simulate.fixed_mix(0.6), INVESTOR, paths).terminal >= INVESTOR.goal
    b = an.run(an.glide(INVESTOR), INVESTOR, paths).terminal >= INVESTOR.goal
    _, paired, independent = metrics.paired_difference(a, b)
    assert paired < independent


@pytest.mark.parametrize("objective,gamma", [("power:5", 5.0), ("power:10", 10.0)])
def test_utility_policy_has_the_best_certainty_equivalent_under_its_own_utility(objective: str, gamma: float) -> None:
    """Out of sample, the DP policy for a utility beats every fixed mix and the glide path on that utility."""
    paths = BASE_WORLD.paths(20_000, INVESTOR.horizon)
    solution = an.solve_policy(INVESTOR, BASE_WORLD, objective, m=2000)
    own = metrics.certainty_equivalent(an.run(simulate.dp_policy(solution), INVESTOR, paths).terminal, gamma)
    rivals = [simulate.fixed_mix(a) for a in an.ACTIONS] + [an.glide(INVESTOR)]
    best_rival = max(metrics.certainty_equivalent(an.run(p, INVESTOR, paths).terminal, gamma) for p in rivals)
    assert own > best_rival
