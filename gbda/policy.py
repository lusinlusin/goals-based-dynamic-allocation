"""Policy tables: solve once offline, look up at run time.

Solving is slow: backward induction evaluates every action at every year and
wealth level (about 3 seconds for the base problem). Using the result is fast:
the weight for a given (year, wealth) is one index into an array. So the two are
separate functions::

    table = policy.solve_policy(investor, world)       # offline
    weight = policy.lookup(table, t=7, wealth=180_000)  # at run time
    policy.save(table, "base.npz"); table = policy.load("base.npz")

A table records the inputs that produced it (investor, return assumptions,
objective, solver settings, package version). Any change in those inputs means
the table must be solved again; :mod:`gbda.resolve` does that part-way through
a plan, which is why a table also carries the calendar year of its first row.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from . import __version__, analysis, dp, simulate
from .analysis import Investor, World

Array = np.ndarray

#: Bumped whenever the saved layout changes; :func:`load` refuses other versions.
FORMAT_VERSION = 1


@dataclass(frozen=True)
class PolicyTable:
    """A solved policy and the inputs behind it.

    Attributes:
        solution: Grid, weights and value function from :func:`gbda.dp.solve`.
        start_year: Calendar year of row 0. A plan solved at the outset has 0; a
            table re-solved in year 10 of a 20-year plan has 10 and 10 rows.
        inputs: Everything needed to say where the table came from (JSON-serialisable).
    """

    solution: dp.DPSolution
    start_year: int
    inputs: dict[str, Any]

    @property
    def end_year(self) -> int:
        """First calendar year the table no longer covers (the goal date)."""
        return self.start_year + self.solution.horizon

    def row(self, t: int) -> int:
        """Row of the table for calendar year ``t``."""
        if not self.start_year <= t < self.end_year:
            raise ValueError(f"year {t} is outside this table ({self.start_year}..{self.end_year - 1})")
        return t - self.start_year

    def success_estimate(self, t: int, wealth: Array) -> Array:
        """The table's own estimate of the chance of reaching the goal from (year ``t``, ``wealth``)."""
        return np.interp(np.asarray(wealth, dtype=float), self.solution.grid, self.solution.value[self.row(t)])


def solve_policy(
    investor: Investor,
    world: World,
    objective: str = "goal",
    start_year: int = 0,
    max_stock: float = 1.0,
    m: int = analysis.DP_SCENARIOS,
    seed: int = analysis.DP_SEED,
    n_grid: int = analysis.GRID_POINTS,
    moment_match: bool = True,
) -> PolicyTable:
    """Solve the DP for ``investor`` in ``world`` and wrap it with its inputs.

    ``investor.horizon`` is the number of years the table covers, counted from
    ``start_year``. To re-solve in year ``t`` of a plan, pass the same investor
    with ``horizon`` reduced by ``t`` and ``start_year=t``.
    """
    solution = analysis.solve_policy(investor, world, objective, max_stock, m, seed, n_grid, moment_match)
    inputs = {
        "investor": asdict(investor),
        "assumptions": asdict(world.assumptions),
        "dist": world.dist,
        "df": world.df,
        "block": world.block,
        "objective": objective,
        "max_stock": max_stock,
        "dp_scenarios": m,
        "dp_seed": seed,
        "grid_points": n_grid,
        "moment_match": moment_match,
        "package_version": __version__,
        "format_version": FORMAT_VERSION,
    }
    return PolicyTable(solution, start_year, inputs)


def lookup(table: PolicyTable, t: int, wealth: Array, method: Literal["nearest", "linear"] = "nearest") -> Array:
    """Stock weight for calendar year ``t`` and the given wealth level(s).

    ``"nearest"`` (default) uses the grid point closest in log wealth, the same
    rule the forward simulations use, so the weight is always one of the allowed
    steps (0%, 10%, ...). ``"linear"`` interpolates the weight between the two
    neighbouring grid points in log wealth; it smooths the jumps where the policy
    switches but can return weights between the allowed steps.
    """
    row = table.row(t)
    if method == "nearest":
        return table.solution.action(row, wealth)
    w = np.log(np.asarray(wealth, dtype=float))
    return np.interp(w, np.log(table.solution.grid), table.solution.policy[row])


def as_policy(tables: PolicyTable | Sequence[PolicyTable]) -> simulate.Policy:
    """A forward-simulation policy that, in each year, uses the latest table solved by then.

    With one table this is just a lookup. With several (an original plan plus
    re-solves), the table with the largest ``start_year <= t`` is used in year ``t``.
    """
    seq = [tables] if isinstance(tables, PolicyTable) else sorted(tables, key=lambda tb: tb.start_year)
    if seq[0].start_year != 0:
        raise ValueError("the first table must start in year 0")
    starts = np.array([tb.start_year for tb in seq])

    def policy(t: int, wealth: Array) -> Array:
        table = seq[int(np.searchsorted(starts, t, side="right")) - 1]
        return lookup(table, t, wealth)

    return policy


# --------------------------------------------------------------------------- #
# Saving and loading
# --------------------------------------------------------------------------- #
def save(table: PolicyTable, path: str | Path) -> Path:
    """Write the table and its inputs to a compressed ``.npz`` file."""
    path = Path(path)
    sol = table.solution
    np.savez_compressed(
        path,
        grid=sol.grid,
        actions=sol.actions,
        policy=sol.policy,
        value=sol.value,
        horizon=sol.horizon,
        contribution=sol.contribution,
        lookup=sol.lookup,
        start_year=table.start_year,
        inputs=json.dumps(table.inputs),
    )
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def load(path: str | Path) -> PolicyTable:
    """Read a table written by :func:`save`."""
    with np.load(path, allow_pickle=False) as f:
        inputs = json.loads(str(f["inputs"]))
        if inputs.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: format version {inputs.get('format_version')}, expected {FORMAT_VERSION}")
        investor = replace(Investor(), **inputs["investor"])
        solution = dp.DPSolution(
            grid=f["grid"],
            actions=f["actions"],
            policy=f["policy"],
            value=f["value"],
            objective=analysis.make_objective(inputs["objective"], investor),
            horizon=int(f["horizon"]),
            contribution=float(f["contribution"]),
            lookup=str(f["lookup"]),  # type: ignore[arg-type]
        )
        return PolicyTable(solution, int(f["start_year"]), inputs)
