"""Count the size of a miss, not only whether it happens.

The goal objective ``P(W_T >= G)`` is indifferent to how far short a miss
falls, so an investor who is behind holds all stocks to the end (section 3).
The usual fix is a blended objective::

    maximise  P(W_T >= G) - eta * E[ ((G - W_T)^+ / G)^p ]

with ``p = 1`` (expected shortfall) or ``p = 2`` (expected squared shortfall).
Below the goal ``(G - W)^+ = G - W``, so with ``p = 1`` the penalty is linear in
wealth: it rewards *expected* wealth, and stocks have the higher expected
return. Only ``p = 2`` penalises a large miss more than a small one.

This module traces both penalties over ``eta`` and puts them next to the
other ways of trading success for safety: a cap on the stock weight, power
utility and fixed mixes. Everything is scored on the same forward paths, on
two axes: the chance of reaching the goal and how bad the bad outcomes are.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import analysis, simulate
from .analysis import Investor, World

Array = np.ndarray

LINEAR_ETAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
QUADRATIC_ETAS = (0.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0)
CAPS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
GAMMAS = (2.0, 3.0, 5.0, 7.0, 10.0, 15.0)


def outcomes(sim: simulate.SimResult, goal: float) -> dict[str, float]:
    """Success and the size of misses for one simulated strategy."""
    w = sim.terminal
    short = np.maximum(goal - w, 0.0)
    miss = w < goal
    p = float((~miss).mean())
    q05, q10, q50, q90 = np.quantile(w, [0.05, 0.10, 0.50, 0.90])
    weights_missed = sim.weights[miss]
    return {
        "success": p,
        "success_se": float(np.sqrt(p * (1 - p) / w.size)),
        "p05": float(q05),
        "p10": float(q10),
        "median": float(q50),
        "p90": float(q90),
        "mean_wealth_if_missed": float(w[miss].mean()) if miss.any() else float("nan"),
        "expected_shortfall": float(short.mean()),
        "rms_shortfall": float(np.sqrt((short**2).mean())),
        "avg_stock_weight": float(sim.weights.mean()),
        "avg_stock_last5_if_missed": float(weights_missed[:, -5:].mean()) if miss.any() else float("nan"),
    }


def case(investor: Investor, objective: str, max_stock: float, n_paths: int) -> dict[str, Any]:
    """Solve one objective (optionally with a cap) and score it on the base-case paths."""
    sol = analysis.solve_policy(investor, World(), objective, max_stock)
    paths = World().paths(n_paths, investor.horizon)
    sim = analysis.run(simulate.dp_policy(sol), investor, paths)
    out = outcomes(sim, investor.goal)
    out["objective"] = objective
    out["max_stock"] = max_stock
    if objective.startswith("shortfall") or objective == "goal":
        out["dp_value"] = sol.value_at(0, investor.w0)
    return out


def experiment(investor: Investor, n_paths: int = analysis.SIM_PATHS) -> dict[str, Any]:
    """Every family of strategies, scored on the same base-case paths."""
    jobs = (
        [(investor, f"shortfall:1:{eta:g}", 1.0, n_paths) for eta in LINEAR_ETAS]
        + [(investor, f"shortfall:2:{eta:g}", 1.0, n_paths) for eta in QUADRATIC_ETAS]
        + [(investor, "goal", cap, n_paths) for cap in CAPS]
        + [(investor, f"power:{g:g}", 1.0, n_paths) for g in GAMMAS]
    )
    res = analysis.run_parallel(case, jobs)
    i = 0
    families: dict[str, list[dict[str, Any]]] = {}
    for name, params, key in (("Linear shortfall penalty", LINEAR_ETAS, "eta"),
                              ("Squared shortfall penalty", QUADRATIC_ETAS, "eta"),
                              ("Goal with a stock cap", CAPS, "cap"),
                              ("Power utility", GAMMAS, "gamma")):
        rows = res[i:i + len(params)]
        i += len(params)
        for row, value in zip(rows, params):
            row[key] = value
        families[name] = rows
    paths = World().paths(n_paths, investor.horizon)
    fixed = []
    for a in analysis.ACTIONS:
        row = outcomes(analysis.run(simulate.fixed_mix(float(a)), investor, paths), investor.goal)
        row["weight"] = float(a)
        fixed.append(row)
    families["Fixed mix"] = fixed
    return {"families": families}


def success_at(rows: list[dict[str, Any]], p10: float) -> float:
    """Success of a family of strategies at a given 10th percentile of final wealth.

    Linear interpolation along the family's points, ordered by the 10th
    percentile, keeping only the part of the curve where raising it costs
    success (the efficient side). ``nan`` if the family never reaches ``p10``.
    """
    pts = sorted((r["p10"], r["success"]) for r in rows)
    frontier: list[tuple[float, float]] = []
    for x, y in reversed(pts):  # from the safest point back: keep points with more success
        if not frontier or y > frontier[-1][1]:
            frontier.append((x, y))
    frontier.reverse()
    xs = np.array([x for x, _ in frontier])
    ys = np.array([y for _, y in frontier])
    if p10 < xs[0]:
        return float(ys[0])  # the family already does better than this level at its best-success point
    if p10 > xs[-1]:
        return float("nan")
    return float(np.interp(p10, xs, ys))
