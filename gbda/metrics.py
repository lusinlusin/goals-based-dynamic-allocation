"""Summary statistics for simulated terminal wealth."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def success_probability(terminal: Array, goal: float) -> tuple[float, float]:
    """Share of paths ending at or above ``goal`` and its standard error ``sqrt(p(1-p)/N)``."""
    hit = np.asarray(terminal) >= goal
    p = float(hit.mean())
    return p, float(np.sqrt(p * (1.0 - p) / hit.size))


def certainty_equivalent(terminal: Array, gamma: float) -> float:
    """Certain amount with the same expected CRRA utility: ``U^{-1}(E[U(W_T)])``.

    ``gamma = 1`` is log utility (the CE is then the geometric mean of ``W_T``).
    Computed on wealth relative to its median to stay inside floating-point range.
    """
    w = np.asarray(terminal, dtype=float)
    scale = float(np.median(w))
    x = w / scale
    if gamma == 1:
        return scale * float(np.exp(np.mean(np.log(x))))
    return scale * float(np.mean(x ** (1.0 - gamma)) ** (1.0 / (1.0 - gamma)))


def paired_difference(x: Array, y: Array) -> tuple[float, float, float]:
    """Mean of ``x - y`` with two standard errors.

    Returns:
        ``(difference, se_paired, se_independent)``. ``se_paired`` is valid when
        ``x`` and ``y`` were simulated on the same return paths (common random
        numbers); ``se_independent`` is what the standard error would be had the
        two been simulated on separate paths.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    diff = x - y
    se_paired = float(diff.std(ddof=1) / np.sqrt(n))
    se_indep = float(np.sqrt(x.var(ddof=1) / n + y.var(ddof=1) / n))
    return float(diff.mean()), se_paired, se_indep


def paths_needed(difference: float, sd_per_path: float, z: float = 1.96) -> int:
    """Paths needed for a difference of this size to be ``z`` standard errors from zero."""
    if difference == 0:
        return int(1e18)
    return int(np.ceil((z * sd_per_path / abs(difference)) ** 2))


def summarize(
    terminal: Array,
    goal: float,
    weights: Array | None = None,
    ce_gammas: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
) -> dict[str, float]:
    """One row of the strategy comparison table.

    Args:
        terminal: Terminal wealth per path.
        goal: Target wealth.
        weights: Optional stock weights, shape ``(N, T)``, for the average weight.
        ce_gammas: Risk-aversion levels to report certainty equivalents for.

    Returns:
        Success probability and standard error, wealth quantiles, the average
        shortfall when the goal is missed, certainty equivalents, average stock weight.
    """
    w = np.asarray(terminal, dtype=float)
    p, se = success_probability(w, goal)
    q05, q10, q50, q90 = np.quantile(w, [0.05, 0.10, 0.50, 0.90])
    missed = w[w < goal]
    row = {
        "success": p,
        "success_se": se,
        "p05": float(q05),
        "p10": float(q10),
        "median": float(q50),
        "p90": float(q90),
        "mean_wealth_if_missed": float(missed.mean()) if missed.size else float("nan"),
    }
    for g in ce_gammas:
        row[f"ce_gamma{g:g}"] = certainty_equivalent(w, g)
    if weights is not None:
        row["avg_stock_weight"] = float(np.mean(weights))
    return row
