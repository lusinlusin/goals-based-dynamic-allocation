"""The three-period example when assumptions change: one small calculation per method.

Same problem as :mod:`gbda.worked_example`: start with 1, goal 1.3, three
decisions (times 0, 1, 2), each time hold cash (x1) or take a coin-flip risk
(x1.4 on heads, x0.7 on tails). With a fair coin the best policy reaches the
goal 62.5% of the time. Three changes, one per method:

1. **Re-solve.** From the second period on, heads pays only x1.3. The investor
   is told at time 1 and solves the last two periods again.
2. **Parameter uncertainty.** The coin is *good* (heads 3/4) or *bad* (heads
   1/4) for all three periods, each with probability 1/2. Nobody knows which;
   the investor learns from the flips (Bayes' rule).
3. **Regimes.** Each period the coin is good or bad, the investor sees which
   before deciding, and it stays the same next period with probability 3/4.

In 2 and 3 a single flip is still heads with probability 1/2 on average, as in
the original example; what changes is how the flips hang together.

Everything is computed twice: by hand (exact fractions, direct recursion, every
sequence listed) and by the project's DP engine on the same problem. The tests
check that the two agree. ``python -m gbda.worked_changes`` prints the tables
used in ``docs/worked_example_changes.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
from typing import Callable

import numpy as np

from . import dp
from . import worked_example as we

F = Fraction
W0, GOAL, UP, DOWN = F(1), F(13, 10), F(7, 5), F(7, 10)
PERIODS = we.PERIODS
CASH, RISK = we.CASH, we.RISK
HALF = F(1, 2)


def reached(w: Fraction) -> Fraction:
    return F(int(w >= GOAL))


def best(hold: Fraction, risk: Fraction) -> tuple[Fraction, float]:
    """Value and action. Ties go to cash, unless the goal is out of reach (the engine's rule)."""
    if risk > hold:
        return risk, RISK
    if hold > risk:
        return hold, CASH
    return hold, (CASH if hold > 0 else RISK)


def _grid() -> np.ndarray:
    return dp.log_grid(0.2, 5.0, 4000, include=float(GOAL))


def _coin(p_heads: float, up: float = float(UP)) -> dp.ReturnScenarios:
    return dp.ReturnScenarios(np.array([up, float(DOWN)]), np.array([1.0, 1.0]), np.array([p_heads, 1.0 - p_heads]))


# =========================================================================== #
# 1. Re-solve: heads pays x1.3 from the second period on
# =========================================================================== #
NEW_UP = F(13, 10)
OLD_COIN = (UP, UP, UP)
NEW_COIN = (UP, NEW_UP, NEW_UP)  # what actually happens: period 1 at x1.4, then x1.3


@lru_cache(maxsize=None)
def coin_value(t: int, w: Fraction, ups: tuple[Fraction, ...]) -> tuple[Fraction, Fraction, Fraction]:
    """``(V, hold, risk)`` at time ``t`` and wealth ``w`` for a fair coin whose heads pays ``ups[t]``."""
    if t == PERIODS:
        r = reached(w)
        return r, r, r
    hold = coin_value(t + 1, w, ups)[0]
    risk = HALF * coin_value(t + 1, w * ups[t], ups)[0] + HALF * coin_value(t + 1, w * DOWN, ups)[0]
    return best(hold, risk)[0], hold, risk


def coin_action(t: int, w: Fraction, ups: tuple[Fraction, ...]) -> float:
    _, hold, risk = coin_value(t, w, ups)
    return best(hold, risk)[1]


def success_coin(policy: Callable[[int, Fraction], float], ups: tuple[Fraction, ...]) -> Fraction:
    """Exact success probability: all eight sequences, fair coin, heads pays ``ups[t]``."""
    total = F(0)
    for heads in product((True, False), repeat=PERIODS):
        w = W0
        for t, h in enumerate(heads):
            if policy(t, w) == RISK:
                w *= ups[t] if h else DOWN
        total += reached(w) / 2**PERIODS
    return total


def resolve_results() -> dict[str, Fraction]:
    """Keep the table, re-solve at time 1, re-solve one period late, hindsight."""

    def switch(at: int) -> Callable[[int, Fraction], float]:
        return lambda t, w: coin_action(t, w, OLD_COIN if t < at else NEW_COIN)

    return {
        "promised": coin_value(0, W0, OLD_COIN)[0],
        "old_at_1.4": coin_value(1, F(7, 5), OLD_COIN)[0],
        "old_at_0.7": coin_value(1, F(7, 10), OLD_COIN)[0],
        "new_at_1.4": coin_value(1, F(7, 5), NEW_COIN)[0],
        "new_at_0.7": coin_value(1, F(7, 10), NEW_COIN)[0],
        "keep": success_coin(switch(PERIODS), NEW_COIN),
        "resolve": success_coin(switch(1), NEW_COIN),
        "late": success_coin(switch(2), NEW_COIN),
        "hindsight": success_coin(switch(0), NEW_COIN),
        "hindsight_value": coin_value(0, W0, NEW_COIN)[0],
    }


def resolve_engine() -> dict[str, dp.DPSolution]:
    """The same tables from the engine: original, re-solved at time 1, hindsight (per-period coins)."""
    args = (np.array([CASH, RISK]), _grid(), dp.goal_objective(float(GOAL)))
    return {
        "original": dp.solve(_coin(0.5), *args, PERIODS, interp="floor"),
        "resolved": dp.solve(_coin(0.5, float(NEW_UP)), *args, PERIODS - 1, interp="floor"),
        "unchanged": dp.solve(_coin(0.5), *args, PERIODS - 1, interp="floor"),
        "hindsight": dp.solve([_coin(0.5, float(u)) for u in NEW_COIN], *args, PERIODS, interp="floor"),
    }


# =========================================================================== #
# 2. Parameter uncertainty: a good or a bad coin, unknown, kept for all periods
# =========================================================================== #
P_GOOD, P_BAD = F(3, 4), F(1, 4)


def belief_good(heads: int, flips: int) -> Fraction:
    """Probability that the coin is good after ``heads`` heads in ``flips`` flips (prior 1/2)."""
    like_good = P_GOOD**heads * (1 - P_GOOD) ** (flips - heads)
    like_bad = P_BAD**heads * (1 - P_BAD) ** (flips - heads)
    return like_good / (like_good + like_bad)


def p_next_heads(heads: int, flips: int) -> Fraction:
    """Probability the next flip is heads, given what has been seen."""
    b = belief_good(heads, flips)
    return b * P_GOOD + (1 - b) * P_BAD


@lru_cache(maxsize=None)
def learn_value(t: int, w: Fraction, heads: int) -> tuple[Fraction, Fraction, Fraction]:
    """``(V, hold, risk)`` for the investor who learns the coin; ``heads`` = heads in the ``t`` flips so far.

    The coin is flipped every period whether or not the investor holds the
    risky asset (the market moves either way), so holding also teaches something.
    """
    if t == PERIODS:
        r = reached(w)
        return r, r, r
    p = p_next_heads(heads, t)
    hold = p * learn_value(t + 1, w, heads + 1)[0] + (1 - p) * learn_value(t + 1, w, heads)[0]
    risk = p * learn_value(t + 1, w * UP, heads + 1)[0] + (1 - p) * learn_value(t + 1, w * DOWN, heads)[0]
    return best(hold, risk)[0], hold, risk


def learn_action(t: int, w: Fraction, heads: int) -> float:
    _, hold, risk = learn_value(t, w, heads)
    return best(hold, risk)[1]


def ignore_action(t: int, w: Fraction, heads: int) -> float:
    """The original policy: treats every flip as a fair coin, looks only at time and wealth."""
    return coin_action(t, w, OLD_COIN)


def success_unknown_coin(policy: Callable[[int, Fraction, int], float], coin: str | None = None) -> Fraction:
    """Exact success probability when the coin is good or bad (``None``: each with probability 1/2)."""
    coins = {"good": P_GOOD, "bad": P_BAD}
    chosen = [coin] if coin else list(coins)
    total = F(0)
    for name in chosen:
        p = coins[name]
        for flips in product((True, False), repeat=PERIODS):
            prob, w, heads = F(1), W0, 0
            for t, h in enumerate(flips):
                prob *= p if h else 1 - p
                if policy(t, w, heads) == RISK:
                    w *= UP if h else DOWN
                heads += h
            total += prob * reached(w)
    return total / len(chosen)


def learn_engine() -> dp.StateDPSolution:
    """The engine with a second state variable: number of heads seen so far (0..3)."""

    def step(t: int, s: int) -> dp.StateStep:
        p = float(p_next_heads(min(s, t), t))  # states with s > t cannot occur; any coin will do
        up, stay = min(s + 1, PERIODS), s
        return dp.StateStep(_coin(p), np.array([up, stay]), np.array([up, stay]), np.zeros(2))

    return dp.solve_with_state(step, PERIODS + 1, np.array([CASH, RISK]), _grid(), float(GOAL), PERIODS)


# =========================================================================== #
# 3. Regimes: the coin switches between good and bad, and the investor sees it
# =========================================================================== #
GOOD, BAD = 0, 1
P_STATE = {GOOD: P_GOOD, BAD: P_BAD}
STAY = F(3, 4)


def move(s: int, s_next: int, stay: Fraction = STAY) -> Fraction:
    return stay if s == s_next else 1 - stay


@lru_cache(maxsize=None)
def regime_value(t: int, w: Fraction, s: int, stay: Fraction = STAY) -> tuple[Fraction, Fraction, Fraction]:
    """``(V, hold, risk)`` in state ``s`` (seen before deciding); next state drawn with ``stay``."""
    if t == PERIODS:
        r = reached(w)
        return r, r, r

    def later(w_next: Fraction) -> Fraction:
        return sum(move(s, s2, stay) * regime_value(t + 1, w_next, s2, stay)[0] for s2 in (GOOD, BAD))

    p = P_STATE[s]
    hold = later(w)
    risk = p * later(w * UP) + (1 - p) * later(w * DOWN)
    return best(hold, risk)[0], hold, risk


def regime_action(t: int, w: Fraction, s: int, stay: Fraction = STAY) -> float:
    _, hold, risk = regime_value(t, w, s, stay)
    return best(hold, risk)[1]


def success_regimes(policy: Callable[[int, Fraction, int], float], start: int | None = None) -> Fraction:
    """Exact success probability; ``start`` = first state, ``None`` = good or bad with probability 1/2."""
    starts = [start] if start is not None else [GOOD, BAD]
    total = F(0)
    for s0 in starts:
        for later_states in product((GOOD, BAD), repeat=PERIODS - 1):
            states = (s0,) + later_states
            p_states = np.prod([move(a, b) for a, b in zip(states[:-1], states[1:])], dtype=object)
            for flips in product((True, False), repeat=PERIODS):
                prob, w = F(p_states), W0
                for t, (s, h) in enumerate(zip(states, flips)):
                    prob *= P_STATE[s] if h else 1 - P_STATE[s]
                    if policy(t, w, s) == RISK:
                        w *= UP if h else DOWN
                total += prob * reached(w)
    return total / len(starts)


def regime_engine(stay: float = float(STAY)) -> dp.StateDPSolution:
    """The engine with the market state as the second state variable."""

    def step(t: int, s: int) -> dp.StateStep:
        to_bad = 1.0 - stay if s == GOOD else stay
        return dp.StateStep(_coin(float(P_STATE[s])), np.array([GOOD, GOOD]), np.array([BAD, BAD]),
                            np.array([to_bad, to_bad]))

    return dp.solve_with_state(step, 2, np.array([CASH, RISK]), _grid(), float(GOAL), PERIODS)


# =========================================================================== #
# Summary
# =========================================================================== #
@dataclass(frozen=True)
class Row:
    """One method's numbers for the comparison table."""

    method: str
    ignore_success: Fraction
    handle_success: Fraction
    ignore_says: Fraction
    handle_says: Fraction
    policy_changes: bool


def comparison() -> list[Row]:
    r = resolve_results()
    learn = success_unknown_coin(learn_action)
    blind = success_regimes(lambda t, w, s: coin_action(t, w, OLD_COIN))
    aware = success_regimes(regime_action)
    aware_says = HALF * (regime_value(0, W0, GOOD)[0] + regime_value(0, W0, BAD)[0])
    return [
        Row("1. Re-solve", r["keep"], r["resolve"], r["promised"], HALF * (r["new_at_1.4"] + r["new_at_0.7"]), False),
        Row("2. Parameter uncertainty", success_unknown_coin(ignore_action), learn, r["promised"],
            learn_value(0, W0, 0)[0], any(learn_action(t, w, h) != ignore_action(t, w, h) for t, w, h in ON_PATH_2)),
        Row("3. Regimes", blind, aware, r["promised"], aware_says, regime_action(0, W0, BAD) != RISK),
    ]


#: Wealth and heads seen at each time on the paths the policies actually visit (method 2).
ON_PATH_2 = [(0, W0, 0), (1, F(7, 5), 1), (1, F(7, 10), 0), (2, F(7, 5), 1), (2, F(7, 5), 2),
             (2, F(49, 50), 1), (2, F(49, 100), 0)]


def pct(x: Fraction) -> str:
    return f"{float(x):.1%}"


def main() -> None:
    """Print every table in docs/worked_example_changes.md."""
    r = resolve_results()
    print("1. Re-solve: heads pays x1.3 from period 2 (told at time 1)")
    print(f"   original table at time 1: 1.4 -> {r['old_at_1.4']}, 0.7 -> {r['old_at_0.7']}")
    print(f"   re-solved table at time 1: 1.4 -> {r['new_at_1.4']}, 0.7 -> {r['new_at_0.7']}")
    print(f"   success: keep {pct(r['keep'])}, re-solve {pct(r['resolve'])}, late {pct(r['late'])}, "
          f"hindsight {pct(r['hindsight'])}; plan promised {pct(r['promised'])}")

    print("\n2. Parameter uncertainty: good (3/4) or bad (1/4) coin, unknown")
    for heads, flips in ((0, 0), (1, 1), (0, 1), (2, 2), (1, 2), (0, 2)):
        print(f"   {heads} heads in {flips} flips: P(good) = {belief_good(heads, flips)}, "
              f"P(next heads) = {p_next_heads(heads, flips)}")
    for t, w, h in ((1, F(7, 10), 0), (1, F(1), 1), (1, F(1), 0), (0, W0, 0)):
        v, hold, risk = learn_value(t, w, h)
        print(f"   t={t} W={float(w):g} heads={h}: hold {hold}, risk {risk}, V {v}")
    for name, pol in (("learns", learn_action), ("ignores", ignore_action), ("always risk", lambda t, w, h: RISK)):
        print(f"   {name:12s} overall {pct(success_unknown_coin(pol))}, good coin "
              f"{pct(success_unknown_coin(pol, 'good'))}, bad coin {pct(success_unknown_coin(pol, 'bad'))}")

    print("\n3. Regimes: good/bad coin each period, seen, stays with probability 3/4")
    for t, w in ((1, F(7, 5)), (1, F(1)), (1, F(7, 10)), (0, W0)):
        for s in (GOOD, BAD):
            v, hold, risk = regime_value(t, w, s)
            print(f"   t={t} W={float(w):g} {'good' if s == GOOD else 'bad '}: hold {hold}, risk {risk}, "
                  f"V {v} -> {'risk' if regime_action(t, w, s) == RISK else 'hold'}")
    for name, pol in (("aware", regime_action), ("blind", lambda t, w, s: coin_action(t, w, OLD_COIN))):
        print(f"   {name}: start good {pct(success_regimes(pol, GOOD))}, start bad {pct(success_regimes(pol, BAD))}, "
              f"either {pct(success_regimes(pol))}")
    wrong = success_regimes(lambda t, w, s: regime_action(t, w, s, HALF))
    print(f"   planned with stay = 1/2 (no persistence): {pct(wrong)}")

    print("\nComparison")
    for row in comparison():
        print(f"   {row.method:26s} ignore {pct(row.ignore_success)} (says {pct(row.ignore_says)}), "
              f"handle {pct(row.handle_success)} (says {pct(row.handle_says)}), policy changes: {row.policy_changes}")


if __name__ == "__main__":
    main()
