"""Shared chart style and the figure types used across the project.

One palette, thin marks, recessive axes. Colour is assigned by job: fixed hues for
strategy identity, a single-hue ramp for the stock weight in policy heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

Array = np.ndarray

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
GREY = "#898781"

SEQ_BLUE = ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
WEIGHT_CMAP = LinearSegmentedColormap.from_list("weight", SEQ_BLUE)


def apply_style() -> None:
    """Set the matplotlib defaults used by every figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 110,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, path: str | Path) -> Path:
    """Save a figure, creating the folder if needed, and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def dollars(x: float, _pos: int | None = None) -> str:
    """Compact dollar tick label: ``$250K``, ``$1.2M``."""
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M".replace(".0M", "M")
    return f"${x / 1e3:.0f}K"


def policy_heatmap(
    ax: plt.Axes,
    grid: Array,
    policy: Array,
    title: str,
    goal: float | None = None,
    w_range: tuple[float, float] | None = None,
    reach: tuple[Array, Array] | None = None,
    weight_label: str = "Stock weight",
) -> matplotlib.cm.ScalarMappable:
    """Draw a policy table: time on x, wealth (log scale) on y, weight as colour.

    Args:
        ax: Target axes.
        grid: Wealth grid, shape ``(n,)``.
        policy: Weights, shape ``(T, n)``.
        title: Panel title.
        goal: If given, draw the goal level.
        w_range: Wealth range to show.
        reach: Optional ``(low, high)`` wealth by year (length ``T``) to outline
            the region that simulated paths actually visit.
        weight_label: Colour-bar label (returned mappable is used by the caller).
    """
    horizon = policy.shape[0]
    levels = np.linspace(-0.05, 1.05, 12)
    norm = BoundaryNorm(levels, WEIGHT_CMAP.N)
    edges_w = np.concatenate(
        [[grid[0] ** 1.5 / grid[1] ** 0.5], np.sqrt(grid[:-1] * grid[1:]), [grid[-1] ** 1.5 / grid[-2] ** 0.5]]
    )
    mesh = ax.pcolormesh(
        np.arange(horizon + 1), edges_w, policy.T, cmap=WEIGHT_CMAP, norm=norm, shading="flat", rasterized=True
    )
    ax.set_yscale("log")
    if w_range:
        ax.set_ylim(*w_range)
    if goal is not None:
        ax.axhline(goal, color=INK, lw=1.0, ls=(0, (4, 3)))
        ax.text(0.15, goal * 1.06, "goal", color=INK, fontsize=8.5, va="bottom")
    if reach is not None:
        years = np.arange(horizon) + 0.5
        for edge in reach:
            ax.plot(years, edge, color=ORANGE, lw=1.4)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.grid(False)
    ax.set_xlim(0, horizon)
    mesh.set_label(weight_label)
    return mesh


def line_panel(
    ax: plt.Axes,
    x: Array,
    series: Mapping[str, Array],
    colors: Mapping[str, str],
    styles: Mapping[str, str] | None = None,
    end_labels: bool = False,
) -> None:
    """Plot several named lines with fixed colours (identity follows the name)."""
    for name, y in series.items():
        ax.plot(x, y, color=colors[name], ls=(styles or {}).get(name, "-"), label=name)
        if end_labels:
            ax.text(x[-1], y[-1], f"  {name}", color=INK_2, fontsize=8.5, va="center")


def hbar(
    ax: plt.Axes,
    labels: Sequence[str],
    values: Sequence[float],
    color: str | Sequence[str] = BLUE,
    fmt: str = "{:.1%}",
    errors: Sequence[float] | None = None,
) -> None:
    """Horizontal bars with the value written at the tip."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.55, color=color, xerr=errors, error_kw={"ecolor": INK_2, "lw": 1})
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", color=INK, fontsize=9)
