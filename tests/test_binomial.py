"""Checks on the coin-flip market, where the right answers are known in advance."""

import numpy as np
import pytest

from gbda import binomial as bn
from gbda import dp


def test_riskless_mix_at_one_third() -> None:
    heads, tails = bn.portfolio_returns(1 / 3)
    assert heads == pytest.approx(1.0) and tails == pytest.approx(1.0)


@pytest.mark.parametrize("gamma,reference", list(bn.REFERENCE_POWER_WEIGHTS.items()))
def test_fixed_mix_optimum_matches_reference(gamma: float, reference: float) -> None:
    assert bn.optimal_weight(gamma) == pytest.approx(reference, abs=6e-4)
    assert bn.optimal_weight_numeric(gamma) == pytest.approx(bn.optimal_weight(gamma), abs=1e-6)


def test_log_optimum_is_11_over_18_and_weights_fall_towards_one_third() -> None:
    assert bn.optimal_weight(1.0) == pytest.approx(11 / 18)
    weights = [bn.optimal_weight(g) for g in (0.5, 1, 2, 5, 10, 100, 1e4)]
    assert all(a > b for a, b in zip(weights, weights[1:]))
    assert weights[-1] == pytest.approx(1 / 3, abs=1e-4)


def test_expected_log10_growth_matches_published_per_mille_values() -> None:
    assert bn.expected_log_growth(1.0, base=10) * 1e3 == pytest.approx(-4.39, abs=0.01)
    assert bn.expected_log_growth(0.0, base=10) * 1e3 == pytest.approx(-18.1, abs=0.05)
    assert bn.expected_log_growth(11 / 18, base=10) * 1e3 == pytest.approx(4.48, abs=0.01)


# ---- Use A: forward simulation of a given policy --------------------------- #
@pytest.mark.parametrize("u", [1.0, 0.0, 11 / 18])
def test_simulated_growth_within_standard_error_of_exact(u: float) -> None:
    summary = bn.summarize_log10_paths(bn.simulate_fixed_mix_log10(u, n_paths=2000, n_flips=4000, seed=11))
    exact = float(bn.expected_log_growth(u, base=10))
    assert abs(summary["growth_per_period"] - exact) < 3 * summary["growth_se"]


def test_positive_mean_return_asset_still_loses_on_most_paths() -> None:
    heads, tails = bn.portfolio_returns(1.0)
    assert 0.5 * heads + 0.5 * tails > 1.0  # +5% arithmetic mean per period
    assert bn.exact_share_ahead(1.0, 4000) == pytest.approx(0.032, abs=0.002)
    assert bn.exact_share_ahead(11 / 18, 4000) > 0.9999


# ---- Use B: optimising a fixed mix by simulation --------------------------- #
def test_mc_optimum_converges_to_11_over_18() -> None:
    errors = {
        m: np.mean([abs(bn.mc_best_fixed_weight(m, seed=s)[0] - 11 / 18) for s in range(50)])
        for m in (100, 10_000, 1_000_000)
    }
    assert errors[100] > errors[10_000] > errors[1_000_000]
    assert errors[1_000_000] < 0.01


def test_common_random_numbers_give_a_smooth_objective() -> None:
    _, crn = bn.mc_best_fixed_weight(1000, seed=5, common_random_numbers=True)
    _, independent = bn.mc_best_fixed_weight(1000, seed=5, common_random_numbers=False)
    roughness = lambda y: np.abs(np.diff(y, 2)).mean()  # noqa: E731
    assert roughness(independent) > 50 * roughness(crn)


# ---- Use C: DP with exact and sampled expectations -------------------------- #
@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_dp_policy_is_the_same_fixed_mix_everywhere(gamma: float) -> None:
    solution = bn.solve_utility_dp(gamma, horizon=10)
    assert solution.policy.min() == solution.policy.max()  # no dependence on time or wealth
    assert solution.policy[0, 0] == pytest.approx(bn.optimal_weight(gamma), abs=5e-4)


def test_log_value_function_is_alpha_times_periods_plus_log_wealth() -> None:
    horizon = 10
    solution = bn.solve_utility_dp(1.0, horizon=horizon)
    alpha = float(bn.expected_log_growth(11 / 18))
    expected = alpha * horizon + np.log(solution.grid)
    assert np.allclose(solution.value[0], expected, atol=1e-6)


def test_sampled_dp_converges_to_exact_dp() -> None:
    exact = bn.solve_utility_dp(1.0, horizon=5)
    gaps = []
    for m in (100, 10_000, 1_000_000):
        runs = [bn.solve_utility_dp(1.0, horizon=5, scenarios=bn.sampled_scenarios(m, seed=s)) for s in range(20)]
        gaps.append(np.mean([abs(r.policy[0, 0] - exact.policy[0, 0]) for r in runs]))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 0.01


# ---- Use D: goal probability -------------------------------------------------- #
PROBLEM = bn.GoalProblem()


def test_best_fixed_mix_reaches_goal_41_percent() -> None:
    _, prob = bn.best_fixed_mix_goal(PROBLEM)
    assert prob == pytest.approx(0.4119, abs=1e-4)  # needs 11 or more heads out of 20


def test_goal_dp_reproduces_reference_on_reference_grid() -> None:
    solution = bn.solve_goal_dp(PROBLEM, weight_step=0.01, n_grid=400, interp="linear")
    value = bn.dp_value_at_start(solution, PROBLEM.w0)
    assert value == pytest.approx(bn.REFERENCE_DP_GOAL, abs=0.01)
    simulated, se = bn.verify_goal_mc(solution, PROBLEM, n_paths=100_000, seed=21)
    assert simulated == pytest.approx(bn.REFERENCE_DP_GOAL, abs=0.01)
    assert abs(simulated - value) < 3 * se + 0.005


def test_goal_dp_bounds_bracket_and_forward_simulation_agrees() -> None:
    low = bn.solve_goal_dp(PROBLEM, 0.01, 2000, "floor", include_riskless=True)
    high = bn.solve_goal_dp(PROBLEM, 0.01, 2000, "ceil", include_riskless=True)
    v_low, v_high = bn.dp_value_at_start(low, PROBLEM.w0), bn.dp_value_at_start(high, PROBLEM.w0)
    simulated, se = bn.verify_goal_mc(low, PROBLEM, n_paths=100_000, seed=21)
    assert v_low <= v_high <= bn.complete_market_bound(PROBLEM)
    assert simulated >= v_low - 3 * se  # the lower-bound policy delivers at least its DP value
    assert simulated <= v_high + 3 * se
    assert v_low > 0.78  # far above the best fixed mix (41%)


def test_goal_policy_gambles_when_behind_and_locks_in_when_ahead() -> None:
    solution = bn.solve_goal_dp(PROBLEM, 0.01, 2000, "floor", include_riskless=True)
    last = solution.policy[-1]
    behind = last[(solution.grid > 1.5 / 1.4) & (solution.grid < 1.2)]
    ahead = last[solution.grid >= 1.5]
    assert np.all(behind > 0.5)
    assert ahead[0] == pytest.approx(1 / 3)
    assert np.all(dp.goal_objective(1.5).terminal(solution.grid[solution.grid >= 1.5]) == 1.0)
