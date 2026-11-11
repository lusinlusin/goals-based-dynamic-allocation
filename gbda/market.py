"""Return assumptions and return generators for the stock/bond problem.

Conventions
-----------
* ``mu`` is the expected annual *simple* return (arithmetic mean), ``sigma`` the
  standard deviation of the annual simple return. Published long-term forecasts
  are usually quoted as compound (geometric) returns; convert them with
  :func:`arithmetic_from_compound` first. At 16% volatility the two differ by
  about 1.2 percentage points, which is as large as the differences this project studies.
* ``rho`` is the correlation between the two assets' log returns.
* Generators return *gross* returns (1 + simple return), always positive.
* Every generator takes a seed. The same seed gives the same underlying shocks
  for any assumption set, so two assumption sets are compared on common random numbers.

Three distribution shapes are supported: lognormal (base case), Student-t log
returns (fat tails, same mean and volatility), and a block bootstrap of historical
years (fat tails plus whatever year-to-year dependence the history contains).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import optimize, stats

Array = np.ndarray
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "us_annual_returns_1928_2025.csv"

#: Shocks beyond this many standard deviations are clipped in the Student-t generator
#: (exp of an unbounded t variable has no finite mean).
T_CLIP = 10.0


# --------------------------------------------------------------------------- #
# Assumption sets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Assumptions:
    """Annual return assumptions for stocks and bonds (see module docstring for conventions)."""

    mu_stock: float = 0.07
    sigma_stock: float = 0.16
    mu_bond: float = 0.03
    sigma_bond: float = 0.05
    rho: float = 0.2
    label: str = "Base case"

    def shifted(self, label: str, **changes: float) -> "Assumptions":
        """Copy with some fields replaced."""
        return replace(self, label=label, **changes)


BASE = Assumptions()


def lognormal_params(mu: float, sigma: float) -> tuple[float, float]:
    """Mean and standard deviation of the log return that give simple-return mean ``mu`` and sd ``sigma``."""
    s2 = np.log1p((sigma / (1.0 + mu)) ** 2)
    return float(np.log1p(mu) - 0.5 * s2), float(np.sqrt(s2))


def arithmetic_from_compound(compound: float, sigma: float) -> float:
    """Expected simple return implied by a compound (geometric) return, for lognormal returns.

    Uses ``median gross return = 1 + compound`` and ``sd = sigma``: solves
    ``(1 + mu) / sqrt(1 + (sigma / (1 + mu))^2) = 1 + compound``.
    """
    g = 1.0 + compound
    m2 = 0.5 * (g * g + np.sqrt(g**4 + 4.0 * g * g * sigma * sigma))
    return float(np.sqrt(m2) - 1.0)


def compound_from_arithmetic(mu: float, sigma: float) -> float:
    """Compound (geometric) return implied by an arithmetic mean, for lognormal returns."""
    m, _ = lognormal_params(mu, sigma)
    return float(np.expm1(m))


# ---- four ways to set the expected returns ---------------------------------- #
def historical_assumptions(
    history: "History", start: int | None = None, end: int | None = None, label: str | None = None
) -> Assumptions:
    """Sample means, volatilities and correlation over a window of the historical record."""
    h = history.window(start, end)
    return Assumptions(
        mu_stock=float(h.stock.mean() - 1.0),
        sigma_stock=float(h.stock.std(ddof=1)),
        mu_bond=float(h.bond.mean() - 1.0),
        sigma_bond=float(h.bond.std(ddof=1)),
        rho=float(np.corrcoef(np.log(h.stock), np.log(h.bond))[0, 1]),
        label=label or f"Historical average {h.years[0]}–{h.years[-1]}",
    )


def building_block_assumptions(
    dividend_yield: float,
    earnings_growth: float,
    valuation_change: float,
    bond_yield: float,
    risk: Assumptions = BASE,
    label: str = "Building blocks",
) -> Assumptions:
    """Stocks = dividend yield + earnings growth + valuation change; bonds = current yield.

    The sums are compound returns and are converted to arithmetic means using the
    volatilities in ``risk`` (which also supplies the correlation).
    """
    stock = dividend_yield + earnings_growth + valuation_change
    return Assumptions(
        mu_stock=arithmetic_from_compound(stock, risk.sigma_stock),
        sigma_stock=risk.sigma_stock,
        mu_bond=arithmetic_from_compound(bond_yield, risk.sigma_bond),
        sigma_bond=risk.sigma_bond,
        rho=risk.rho,
        label=label,
    )


def equilibrium_assumptions(
    stock_market_weight: float = 0.6,
    risk_aversion: float = 2.5,
    cash_rate: float = 0.03,
    risk: Assumptions = BASE,
    label: str = "Market equilibrium",
) -> Assumptions:
    """Returns that make the market portfolio optimal (reverse optimisation).

    Excess returns are ``risk_aversion * Covariance * market weights`` — the
    starting point of Black–Litterman — added to a cash rate.
    """
    w = np.array([stock_market_weight, 1.0 - stock_market_weight])
    cov_sb = risk.rho * risk.sigma_stock * risk.sigma_bond
    cov = np.array([[risk.sigma_stock**2, cov_sb], [cov_sb, risk.sigma_bond**2]])
    excess = risk_aversion * cov @ w
    return Assumptions(
        mu_stock=float(cash_rate + excess[0]),
        sigma_stock=risk.sigma_stock,
        mu_bond=float(cash_rate + excess[1]),
        sigma_bond=risk.sigma_bond,
        rho=risk.rho,
        label=label,
    )


def compound_forecast_assumptions(
    stock_compound: float, bond_compound: float, risk: Assumptions = BASE, label: str = "Published forecast"
) -> Assumptions:
    """Assumption set from compound-return forecasts, e.g. a published long-term outlook."""
    return Assumptions(
        mu_stock=arithmetic_from_compound(stock_compound, risk.sigma_stock),
        sigma_stock=risk.sigma_stock,
        mu_bond=arithmetic_from_compound(bond_compound, risk.sigma_bond),
        sigma_bond=risk.sigma_bond,
        rho=risk.rho,
        label=label,
    )


# --------------------------------------------------------------------------- #
# Parametric generators
# --------------------------------------------------------------------------- #
def _student_shock_moments(m: float, s: float, df: float) -> tuple[float, float]:
    """Mean and sd of ``exp(m + s * Z)`` where ``Z`` is a unit-variance t shock clipped at ``T_CLIP``."""
    scale = np.sqrt((df - 2.0) / df)
    z = np.linspace(-T_CLIP, T_CLIP, 20001)
    pdf = stats.t.pdf(z / scale, df) / scale
    tail = stats.t.sf(T_CLIP / scale, df)
    dz = z[1] - z[0]

    def expect(power: float) -> float:
        f = np.exp(power * (m + s * z))
        inside = float(np.sum(0.5 * (f[1:] * pdf[1:] + f[:-1] * pdf[:-1])) * dz)
        return inside + tail * (f[0] + f[-1])

    mean = expect(1.0)
    return mean, float(np.sqrt(max(expect(2.0) - mean * mean, 0.0)))


def student_params(mu: float, sigma: float, df: float) -> tuple[float, float]:
    """Location and scale of the log return so the simple return has mean ``mu`` and sd ``sigma``
    when shocks are (clipped, unit-variance) Student-t with ``df`` degrees of freedom."""
    if sigma == 0:
        return float(np.log1p(mu)), 0.0
    m0, s0 = lognormal_params(mu, sigma)

    def gap(x: Array) -> Array:
        mean, sd = _student_shock_moments(x[0], x[1], df)
        return np.array([mean - (1.0 + mu), sd - sigma])

    sol = optimize.root(gap, np.array([m0, s0]), tol=1e-12)
    return float(sol.x[0]), float(sol.x[1])


def sample_returns(
    assumptions: Assumptions,
    size: int | tuple[int, ...],
    seed: int,
    dist: Literal["lognormal", "student_t"] = "lognormal",
    df: float = 5.0,
    moment_match: bool = False,
) -> tuple[Array, Array]:
    """Draw gross annual returns for stocks and bonds.

    Args:
        assumptions: Means, volatilities and correlation.
        size: Output shape, e.g. ``M`` for DP scenarios or ``(N, T)`` for forward paths.
        seed: Random seed (same seed -> same underlying shocks for any assumptions).
        dist: ``"lognormal"`` or ``"student_t"`` (bivariate t log returns with ``df``
            degrees of freedom, matched to the same mean and volatility of simple
            returns, so only the tail shape changes).
        df: Degrees of freedom of the Student-t (lower = fatter tails; must be > 2).
        moment_match: Use antithetic pairs and rescale the underlying normal shocks
            so their sample mean is exactly 0, variance exactly 1 and cross-correlation
            exactly ``rho``. Meant for the DP scenario set (1-D ``size``): a plain
            sample of a few thousand draws misses the mean stock return by a few
            tenths of a percent, which moves the DP's success estimate by several
            points. Leave off for forward paths, where standard errors assume
            independent draws.

    Returns:
        ``(stock gross returns, bond gross returns)``, each of shape ``size``.
    """
    rng = np.random.default_rng(seed)
    if moment_match:
        if not isinstance(size, (int, np.integer)) or size % 2:
            raise ValueError("moment_match needs a one-dimensional sample of even size")
        half = int(size) // 2
        z1 = rng.standard_normal(half)
        z2 = rng.standard_normal(half)
        z1 = np.concatenate([z1, -z1])
        z2 = np.concatenate([z2, -z2])
        z1 /= z1.std()
        z2 -= (z1 @ z2) / (z1 @ z1) * z1
        z2 /= z2.std()
    else:
        z1 = rng.standard_normal(size)
        z2 = rng.standard_normal(size)
    rho = assumptions.rho
    zs, zb = z1, rho * z1 + np.sqrt(1.0 - rho * rho) * z2
    if dist == "lognormal":
        ms, ss = lognormal_params(assumptions.mu_stock, assumptions.sigma_stock)
        mb, sb = lognormal_params(assumptions.mu_bond, assumptions.sigma_bond)
    elif dist == "student_t":
        if df <= 2:
            raise ValueError("df must exceed 2 for the variance to exist")
        # A shared chi-square mixing variable makes both assets fat-tailed together.
        if moment_match:
            mix_half = np.sqrt((df - 2.0) / rng.chisquare(df, int(size) // 2))
            mix = np.concatenate([mix_half, mix_half])
        else:
            mix = np.sqrt((df - 2.0) / rng.chisquare(df, size))
        zs = np.clip(zs * mix, -T_CLIP, T_CLIP)
        zb = np.clip(zb * mix, -T_CLIP, T_CLIP)
        ms, ss = student_params(assumptions.mu_stock, assumptions.sigma_stock, df)
        mb, sb = student_params(assumptions.mu_bond, assumptions.sigma_bond, df)
    else:
        raise ValueError(f"unknown distribution {dist!r}")
    return np.exp(ms + ss * zs), np.exp(mb + sb * zb)


# --------------------------------------------------------------------------- #
# Historical record and block bootstrap
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class History:
    """Annual gross returns of stocks and bonds, in calendar order."""

    years: Array
    stock: Array
    bond: Array

    def window(self, start: int | None = None, end: int | None = None) -> "History":
        """Sub-period ``start..end`` (inclusive)."""
        keep = np.ones(self.years.size, dtype=bool)
        if start is not None:
            keep &= self.years >= start
        if end is not None:
            keep &= self.years <= end
        return History(self.years[keep], self.stock[keep], self.bond[keep])

    def recentred(self, target: Assumptions) -> "History":
        """Same sequence and shape of returns, rescaled to the target means and volatilities.

        Log returns are standardised and rescaled (location and scale found
        numerically) so that the simple returns have exactly the target mean and
        standard deviation. This separates *shape* (tails, skew, year-to-year
        dependence) from *level*, which is set by forward-looking assumptions.
        """

        def rescale(gross: Array, mu: float, sigma: float) -> Array:
            z = np.log(gross)
            z = (z - z.mean()) / z.std()

            def gap(x: Array) -> Array:
                r = np.exp(x[0] + x[1] * z)
                return np.array([r.mean() - (1.0 + mu), r.std() - sigma])

            m0, s0 = lognormal_params(mu, sigma)
            sol = optimize.root(gap, np.array([m0, s0]), tol=1e-13)
            return np.exp(sol.x[0] + sol.x[1] * z)

        return History(
            self.years,
            rescale(self.stock, target.mu_stock, target.sigma_stock),
            rescale(self.bond, target.mu_bond, target.sigma_bond),
        )


def load_history(path: str | Path = DATA_FILE) -> History:
    """Load annual S&P 500 and 10-year Treasury total returns (percent) as gross returns."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    return History(raw[:, 0].astype(int), 1.0 + raw[:, 1] / 100.0, 1.0 + raw[:, 3] / 100.0)


def block_bootstrap(
    history: History, n_paths: int, horizon: int, seed: int, block: int = 5
) -> tuple[Array, Array]:
    """Resample the historical record in blocks of consecutive years.

    Each path is built from blocks of ``block`` consecutive years starting at
    random points (wrapping around the end of the sample), so stock and bond
    returns stay paired and short-run dependence across years is kept.
    ``block=1`` is the ordinary bootstrap (independent years).

    Returns:
        ``(stock gross returns, bond gross returns)``, each of shape ``(n_paths, horizon)``.
    """
    rng = np.random.default_rng(seed)
    n = history.years.size
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(n_paths, n_blocks * block)[:, :horizon]
    return history.stock[idx], history.bond[idx]
