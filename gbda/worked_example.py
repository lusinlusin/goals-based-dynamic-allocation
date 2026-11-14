"""A three-period problem small enough to solve by hand, solved by the same engine.

Setting: start with wealth 1, three periods, goal 1.3. Each period choose one of
two actions:

* **hold cash** — wealth stays the same;
* **take risk** — a fair coin: wealth is multiplied by 1.4 on heads, 0.7 on tails.

The objective is the probability of ending at or above the goal. Backward
induction gives 62.5%; staying in the risky asset throughout gives 50%. The gap
comes from one move: after an early head the investor is already above the goal
and stops taking risk.

``python -m gbda.worked_example`` prints the hand tables next to the engine's
numbers and draws the decision tree used in ``docs/worked_example.md``.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import dp, plots

Array = np.ndarray

W0 = 1.0
GOAL = 1.3
PERIODS = 3
UP, DOWN = 1.4, 0.7
CASH, RISK = 0.0, 1.0
ACTION_NAMES = {CASH: "hold cash", RISK: "take risk"}


def scenarios(up: float = UP, down: float = DOWN) -> dp.ReturnScenarios:
    """Heads and tails with probability 1/2: the risky asset moves, cash does not."""
    return dp.ReturnScenarios(r_a=np.array([up, down]), r_b=np.array([1.0, 1.0]), probs=np.array([0.5, 0.5]))


def solve(n_grid: int = 4000, up: float = UP, periods: int = PERIODS) -> dp.DPSolution:
    """Solve the three-period problem with the project's DP engine.

    The value function is a staircase in wealth, so the grid is read with
    ``interp="floor"``. Every wealth level that can occur is at least 5% away from
    a step, far more than the grid spacing, so the values at those levels are exact.

    ``up`` and ``periods`` change the coin and the number of periods left; they
    are used to re-solve part-way through (see :func:`resolve_demo`).
    """
    grid = dp.log_grid(0.2, 5.0, n_grid, include=GOAL)
    return dp.solve(scenarios(up), np.array([CASH, RISK]), grid, dp.goal_objective(GOAL), periods, interp="floor")


def engine_value(solution: dp.DPSolution, t: int, wealth: float) -> float:
    """Success probability the engine assigns to ``wealth`` at time ``t``."""
    return float(solution.value[t][solution.grid_index(wealth)])


# --------------------------------------------------------------------------- #
# The same calculation done the way one would do it on paper
# --------------------------------------------------------------------------- #
def hand_value(t: int, wealth: float) -> tuple[float, float, float]:
    """Backward induction by direct recursion, no grid.

    Returns:
        ``(value, value if holding cash now, value if taking risk now)``.
    """
    if t == PERIODS:
        reached = float(wealth >= GOAL - 1e-12)
        return reached, reached, reached
    hold = hand_value(t + 1, wealth)[0]
    risk = 0.5 * hand_value(t + 1, wealth * UP)[0] + 0.5 * hand_value(t + 1, wealth * DOWN)[0]
    return max(hold, risk), hold, risk


def path_success(policy: dp.DPSolution | float | Callable[[int, float], float], ups: Sequence[float] | None = None) -> float:
    """Exact success probability of a policy by listing all eight coin sequences.

    ``policy`` is a DP solution, a fixed weight in the risky asset, or any
    ``policy(t, wealth) -> weight``. ``ups`` gives the heads multiplier in each
    period (default ``UP`` throughout); tails is always ``DOWN``.
    """
    ups = [UP] * PERIODS if ups is None else list(ups)
    wins = 0
    for heads in product((True, False), repeat=PERIODS):
        wealth = W0
        for t, h in enumerate(heads):
            if isinstance(policy, float):
                a = policy
            elif isinstance(policy, dp.DPSolution):
                a = float(policy.action(t, wealth))
            else:
                a = float(policy(t, wealth))
            wealth *= a * (ups[t] if h else DOWN) + (1.0 - a)
        wins += wealth >= GOAL - 1e-12
    return wins / 2**PERIODS


# --------------------------------------------------------------------------- #
# The coin changes part-way through: re-solve the remaining periods
# --------------------------------------------------------------------------- #
#: After the first period it turns out heads pays x1.3, not x1.4.
NEW_UP = 1.3


def resolve_demo() -> dict[str, float]:
    """Re-solve at time 1 with the new coin and compare with keeping the original table.

    At time 1 with wealth 0.7 the original table still says "take risk, 25%".
    Re-solved with heads at x1.3, the last period's threshold becomes 1.3 / 1.3
    = 1.0; one head from 0.7 reaches only 0.91, so the chance is 0. The action
    does not change (the goal is out of reach either way); the forecast does.
    The same check as in the full model also holds here: re-solving at time 1
    with the *old* coin gives back the original table's last two periods.
    """
    original = solve()
    resolved = solve(up=NEW_UP, periods=PERIODS - 1)
    unchanged = solve(periods=PERIODS - 1)
    ups = [UP] + [NEW_UP] * (PERIODS - 1)

    def rolling(t: int, wealth: float) -> float:
        return float(original.action(0, wealth)) if t == 0 else float(resolved.action(t - 1, wealth))

    return {
        "original_forecast_at_0.7": engine_value(original, 1, 0.7),
        "resolved_forecast_at_0.7": engine_value(resolved, 0, 0.7),
        "original_forecast_at_1.4": engine_value(original, 1, 1.4),
        "resolved_forecast_at_1.4": engine_value(resolved, 0, 1.4),
        "plan_promised": engine_value(original, 0, W0),
        "keep_table_success": path_success(original, ups),
        "resolve_success": path_success(rolling, ups),
        "unchanged_resolve_matches": float(
            np.array_equal(unchanged.policy, original.policy[1:]) and np.array_equal(unchanged.value, original.value[1:])
        ),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
#: Wealth levels worth tabulating at each decision date (on and off the optimal path).
NODES = {2: (1.96, 1.4, 1.0, 0.98, 0.7, 0.49), 1: (1.4, 1.0, 0.7), 0: (1.0,)}


def _figure_tree(path: Path, solution: dp.DPSolution) -> None:
    """Draw the tree that the optimal policy actually follows."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    # (time, y position, wealth, parent index, edge label)
    nodes = [
        (0, 3.0, 1.0, None, ""),
        (1, 4.5, 1.4, 0, "heads"),
        (1, 1.5, 0.7, 0, "tails"),
        (2, 4.5, 1.4, 1, "hold"),
        (2, 2.4, 0.98, 2, "heads"),
        (2, 0.6, 0.49, 2, "tails"),
        (3, 4.5, 1.4, 3, "hold"),
        (3, 3.0, 1.372, 4, "heads"),
        (3, 1.8, 0.686, 4, "tails"),
        (3, 0.6, 0.686, 5, "heads (best case)"),
    ]
    colors = {CASH: plots.BLUE, RISK: plots.ORANGE}
    for t, y, wealth, parent, label in nodes:
        if parent is not None:
            pt, py = nodes[parent][0], nodes[parent][1]
            ax.plot([pt + 0.16, t - 0.16], [py, y], color=plots.AXIS, lw=1.5, zorder=1)
            ax.text((pt + t) / 2, (py + y) / 2 + 0.16, label, ha="center", va="bottom", fontsize=9, color=plots.INK_2)
        if t < PERIODS:
            value = engine_value(solution, t, wealth)
            action = float(solution.action(t, wealth))
            out_of_reach = value == 0.0
            face = plots.GREY if out_of_reach else colors[action]
            ax.scatter([t], [y], s=1500, color=face, edgecolor=plots.SURFACE, lw=2, zorder=2)
            ax.text(t, y, f"{wealth:g}", ha="center", va="center", fontsize=11, color="white", fontweight="bold", zorder=3)
            what = "out of reach" if out_of_reach else ACTION_NAMES[action]
            ax.text(t, y - 0.52, f"{what}\nchance {value:.1%}".replace(".0%", "%"), ha="center", va="top",
                    fontsize=9, color=plots.INK)
        else:
            reached = wealth >= GOAL
            ax.scatter([t], [y], s=1500, color=plots.SURFACE, edgecolor=plots.INK if reached else plots.AXIS, lw=2, zorder=2)
            ax.text(t, y, f"{wealth:g}", ha="center", va="center", fontsize=11, color=plots.INK, zorder=3)
            ax.text(t + 0.22, y, "goal reached" if reached else "missed", ha="left", va="center", fontsize=9.5,
                    color=plots.INK, fontweight="bold" if reached else "normal")
    ax.set_xticks(range(PERIODS + 1), ["Start", "After 1 period", "After 2 periods", "End"])
    ax.set_xlim(-0.4, 3.75)
    ax.set_ylim(-0.5, 5.3)
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, color=c) for c in (plots.ORANGE, plots.BLUE, plots.GREY)]
    ax.legend(handles, ["Take risk", "Hold cash", "Goal out of reach"], loc="upper left", ncol=3,
              bbox_to_anchor=(0.0, 1.02), columnspacing=1.6, handletextpad=0.3)
    ax.set_title(f"Three periods, goal {GOAL}: wealth, best action and chance of reaching the goal", pad=28)
    plots.save(fig, path)


def main(figure_path: str | Path = "figures/worked_example_tree.png") -> None:
    """Print the hand tables beside the engine's values and draw the tree."""
    plots.apply_style()
    solution = solve()
    print(f"Start {W0}, goal {GOAL}, {PERIODS} periods; risky asset x{UP} or x{DOWN}, cash x1\n")
    for t in (2, 1, 0):
        print(f"Decision at time {t} ({PERIODS - t} period(s) left)")
        print("  wealth   hold cash   take risk   best        engine")
        for wealth in NODES[t]:
            best, hold, risk = hand_value(t, wealth)
            choice = "either" if hold == risk else ("hold cash" if hold > risk else "take risk")
            print(f"  {wealth:<7g}  {hold:<10.3f}  {risk:<10.3f}  {choice:<10}  {engine_value(solution, t, wealth):.3f}")
        print()
    print(f"Dynamic policy, all 8 coin sequences: {path_success(solution):.1%}")
    print(f"Always take risk:                    {path_success(RISK):.1%}")
    print(f"Always hold cash:                    {path_success(CASH):.1%}")
    demo = resolve_demo()
    print(f"\nAfter period 1 heads turns out to pay x{NEW_UP}; re-solve the last two periods:")
    print(f"  wealth 0.7: original table {demo['original_forecast_at_0.7']:.0%}, re-solved {demo['resolved_forecast_at_0.7']:.0%}")
    print(f"  wealth 1.4: original table {demo['original_forecast_at_1.4']:.0%}, re-solved {demo['resolved_forecast_at_1.4']:.0%}")
    print(f"  plan promised {demo['plan_promised']:.1%}; keeping the table gets {demo['keep_table_success']:.1%}, "
          f"re-solving gets {demo['resolve_success']:.1%}")
    print(f"  re-solving with the old coin gives back the original table: {bool(demo['unchanged_resolve_matches'])}")
    _figure_tree(Path(figure_path), solution)
    print(f"\nTree written to {figure_path}")


if __name__ == "__main__":
    main()
