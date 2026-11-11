"""The three-period example: the engine must reproduce the numbers worked out by hand."""

import pytest

from gbda import worked_example as we

SOLUTION = we.solve()


def test_engine_matches_the_hand_calculation_at_every_tabulated_wealth_level() -> None:
    for t, levels in we.NODES.items():
        for wealth in levels:
            assert we.engine_value(SOLUTION, t, wealth) == pytest.approx(we.hand_value(t, wealth)[0])


def test_best_achievable_chance_is_five_eighths() -> None:
    assert we.hand_value(0, we.W0)[0] == pytest.approx(5 / 8)
    assert we.engine_value(SOLUTION, 0, we.W0) == pytest.approx(5 / 8)
    assert we.path_success(SOLUTION) == pytest.approx(5 / 8)  # all eight coin sequences


def test_fixed_rules_do_worse() -> None:
    assert we.path_success(we.RISK) == pytest.approx(4 / 8)
    assert we.path_success(we.CASH) == 0.0


def test_policy_takes_risk_when_behind_and_stops_when_ahead() -> None:
    assert SOLUTION.action(0, 1.0) == we.RISK
    assert SOLUTION.action(1, 1.4) == we.CASH  # goal secured after one head
    assert SOLUTION.action(1, 0.7) == we.RISK
    assert SOLUTION.action(2, 0.98) == we.RISK
    assert SOLUTION.action(2, 1.4) == we.CASH


def test_hand_tables_quoted_in_the_write_up() -> None:
    assert we.hand_value(1, 1.4)[1:] == pytest.approx((1.0, 0.75))  # hold cash vs take risk at 1.4
    assert we.hand_value(1, 0.7)[1:] == pytest.approx((0.0, 0.25))
    assert we.hand_value(0, 1.0)[1:] == pytest.approx((0.5, 0.625))
