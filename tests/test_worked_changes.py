"""The three-period example under each method: hand fractions, engine and every quoted number agree."""

from fractions import Fraction as F

import numpy as np
import pytest

from gbda import worked_changes as wc

RISK, CASH = wc.RISK, wc.CASH


def state_value(sol, t: int, wealth: float, s: int) -> float:
    return float(sol.value[t, s, sol.grid_index(wealth)])


# ---- 1. Re-solve ---------------------------------------------------------------- #
def test_resolve_numbers_in_the_write_up() -> None:
    r = wc.resolve_results()
    assert r["promised"] == F(5, 8)
    assert (r["old_at_1.4"], r["old_at_0.7"]) == (1, F(1, 4))
    assert (r["new_at_1.4"], r["new_at_0.7"]) == (1, 0)  # 0.7 x 1.3 x 1.3 = 1.183 < 1.3
    assert r["keep"] == r["resolve"] == r["late"] == r["hindsight"] == F(1, 2)
    assert r["hindsight_value"] == F(1, 2)


def test_resolve_engine_matches_the_hand_calculation() -> None:
    e = wc.resolve_engine()
    assert np.array_equal(e["unchanged"].policy, e["original"].policy[1:])  # optimality principle
    assert np.array_equal(e["unchanged"].value, e["original"].value[1:])
    for w in (1.4, 0.7):
        assert e["resolved"].value[0][e["resolved"].grid_index(w)] == float(wc.coin_value(1, F(w), wc.NEW_COIN)[0])
        assert e["hindsight"].value[1][e["hindsight"].grid_index(w)] == float(wc.coin_value(1, F(w), wc.NEW_COIN)[0])
    assert e["hindsight"].value[0][e["hindsight"].grid_index(1.0)] == pytest.approx(0.5)


# ---- 2. Parameter uncertainty ----------------------------------------------------- #
def test_beliefs_follow_bayes_rule() -> None:
    assert wc.belief_good(1, 1) == F(3, 4) and wc.p_next_heads(1, 1) == F(5, 8)
    assert wc.belief_good(0, 1) == F(1, 4) and wc.p_next_heads(0, 1) == F(3, 8)
    assert wc.belief_good(2, 2) == F(9, 10) and wc.p_next_heads(2, 2) == F(7, 10)
    assert wc.p_next_heads(1, 2) == F(1, 2) and wc.p_next_heads(0, 2) == F(3, 10)
    # Averaged over what could have been seen, the next flip is still heads half the time.
    from math import comb

    for n in range(3):
        seen = {h: F(comb(n, h)) / 2 * (wc.P_GOOD**h * (1 - wc.P_GOOD) ** (n - h) + wc.P_BAD**h * (1 - wc.P_BAD) ** (n - h))
                for h in range(n + 1)}
        assert sum(seen.values()) == 1
        assert sum(p * wc.p_next_heads(h, n) for h, p in seen.items()) == F(1, 2)


def test_learning_gives_the_same_policy_and_the_same_success() -> None:
    assert wc.learn_value(1, F(7, 10), 0)[1:] == (0, F(3, 16))
    assert wc.learn_value(0, wc.W0, 0)[1:] == (F(1, 2), F(19, 32))
    for t, w, h in wc.ON_PATH_2:
        assert wc.learn_action(t, w, h) == wc.ignore_action(t, w, h)
    assert wc.success_unknown_coin(wc.learn_action) == wc.success_unknown_coin(wc.ignore_action) == F(19, 32)
    assert wc.success_unknown_coin(wc.learn_action, "good") == F(57, 64)
    assert wc.success_unknown_coin(wc.learn_action, "bad") == F(19, 64)
    assert wc.success_unknown_coin(lambda t, w, h: RISK) == F(1, 2)  # always risk: unchanged


def test_learning_engine_matches_the_hand_calculation() -> None:
    sol = wc.learn_engine()
    for t, w, h in wc.ON_PATH_2:
        assert state_value(sol, t, float(w), h) == pytest.approx(float(wc.learn_value(t, w, h)[0]))
        assert sol.action(t, float(w), h) == wc.learn_action(t, w, h)


# ---- 3. Regimes ------------------------------------------------------------------- #
def test_regime_numbers_in_the_write_up() -> None:
    assert wc.regime_value(1, F(1), wc.GOOD)[1:] == (F(5, 8), F(3, 4))
    assert wc.regime_value(1, F(1), wc.BAD)[1:] == (F(3, 8), F(1, 4))
    assert wc.regime_value(1, F(7, 10), wc.GOOD)[0] == F(15, 32)
    assert wc.regime_value(1, F(7, 10), wc.BAD)[0] == F(3, 32)
    assert wc.regime_value(0, wc.W0, wc.GOOD)[1:] == (F(21, 32), F(27, 32))
    assert wc.regime_value(0, wc.W0, wc.BAD)[1:] == (F(15, 32), F(25, 64))
    assert wc.regime_action(0, wc.W0, wc.BAD) == CASH  # bad state: hold cash and wait


def test_regime_success_aware_against_blind() -> None:
    def blind(t, w, s):
        return wc.coin_action(t, w, wc.OLD_COIN)

    assert wc.success_regimes(wc.regime_action, wc.GOOD) == wc.success_regimes(blind, wc.GOOD) == F(27, 32)
    assert wc.success_regimes(wc.regime_action, wc.BAD) == F(15, 32)
    assert wc.success_regimes(blind, wc.BAD) == F(25, 64)
    assert wc.success_regimes(wc.regime_action) == F(21, 32)
    assert wc.success_regimes(blind) == F(79, 128)
    # Planning with no persistence picks the same actions here, so it costs nothing.
    assert wc.success_regimes(lambda t, w, s: wc.regime_action(t, w, s, F(1, 2))) == F(21, 32)


def test_regime_engine_matches_the_hand_calculation() -> None:
    sol = wc.regime_engine()
    nodes = {0: (1.0,), 1: (1.4, 1.0, 0.7), 2: (1.4, 1.0, 0.98, 0.7)}
    for t, levels in nodes.items():
        for w in levels:
            for s in (wc.GOOD, wc.BAD):
                assert state_value(sol, t, w, s) == pytest.approx(float(wc.regime_value(t, F(w), s)[0]))
                if wc.regime_value(t, F(w), s)[0] > 0:
                    assert sol.action(t, w, s) == wc.regime_action(t, F(w), s)


# ---- Comparison ------------------------------------------------------------------- #
def test_comparison_table() -> None:
    rows = {r.method[0]: r for r in wc.comparison()}
    assert (rows["1"].ignore_success, rows["1"].handle_success, rows["1"].handle_says) == (F(1, 2), F(1, 2), F(1, 2))
    assert (rows["2"].ignore_success, rows["2"].handle_success, rows["2"].handle_says) == (F(19, 32),) * 3
    assert (rows["3"].ignore_success, rows["3"].handle_success, rows["3"].handle_says) == (F(79, 128), F(21, 32), F(21, 32))
    assert [r.policy_changes for r in rows.values()] == [False, False, True]
    assert all(r.ignore_says == F(5, 8) for r in rows.values())
