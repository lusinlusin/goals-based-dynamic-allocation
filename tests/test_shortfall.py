"""Penalising the size of a miss: the objective, its limits, and the comparison helper."""

import numpy as np
import pytest

from gbda import analysis as an
from gbda import dp, shortfall

INVESTOR = an.Investor()
G = INVESTOR.goal


def test_zero_penalty_is_the_goal_objective() -> None:
    goal = an.solve_policy(INVESTOR, an.World(), "goal", m=1000)
    for name in ("shortfall:1:0", "shortfall:2:0"):
        sol = an.solve_policy(INVESTOR, an.World(), name, m=1000)
        assert np.array_equal(sol.policy, goal.policy) and np.array_equal(sol.value, goal.value)


def test_terminal_reward() -> None:
    w = np.array([0.5 * G, 0.9 * G, G, 2 * G])
    assert dp.shortfall_objective(G, 1.0, 1.0).terminal(w) == pytest.approx([-0.5, -0.1, 1.0, 1.0])
    assert dp.shortfall_objective(G, 4.0, 2.0).terminal(w) == pytest.approx([-1.0, -0.04, 1.0, 1.0])


def test_linear_penalty_rewards_expected_wealth_far_below_the_goal() -> None:
    """Last year, $100K: the goal is out of reach in every scenario, so a linear penalty on the
    shortfall is just minus expected wealth, and the stock (higher expected return) wins."""
    sol = an.solve_policy(INVESTOR, an.World(), "shortfall:1:8", m=1000)
    assert sol.action(INVESTOR.horizon - 1, 100_000.0) == 1.0


def test_squared_penalty_cuts_risk_when_short_of_the_goal_near_the_date() -> None:
    goal = an.solve_policy(INVESTOR, an.World(), "goal", m=1000)
    squared = an.solve_policy(INVESTOR, an.World(), "shortfall:2:32", m=1000)
    t, w = INVESTOR.horizon - 1, 350_000.0  # 30% short with one year to go
    assert goal.action(t, w) == 1.0
    assert squared.action(t, w) < 1.0
    assert squared.action(t, 60_000.0) == 1.0  # far behind: still all stock


def test_success_at_reads_the_efficient_side_of_a_family() -> None:
    rows = [{"p10": 300.0, "success": 0.60}, {"p10": 310.0, "success": 0.50}, {"p10": 320.0, "success": 0.30},
            {"p10": 315.0, "success": 0.10}]  # the last point is dominated
    assert shortfall.success_at(rows, 305.0) == pytest.approx(0.55)
    assert shortfall.success_at(rows, 295.0) == pytest.approx(0.60)
    assert np.isnan(shortfall.success_at(rows, 325.0))
