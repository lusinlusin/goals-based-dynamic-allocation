"""Run every experiment for the stock/bond goal problem.

    python run.py            # everything (a few minutes on a laptop)
    python run.py --only compare sensitivity
    python run.py --quick    # smaller samples, for a smoke test

Writes tables to ``results/`` (Markdown + one JSON with every number) and figures
to ``figures/``. Sections:

    compare      all strategies side by side, policy heatmaps, wealth distributions
    assumptions  four ways of setting expected returns -> success and required saving
    sensitivity  one assumption moved at a time, plus fat tails
    dynamic      where a dynamic policy beats static ones (goal distance, horizon, funding)
    cap          cost of a ceiling on the stock weight
    precision    Monte Carlo error: forward paths and DP scenario sets
    saving       success probability as a function of yearly saving
    resolve      assumptions change in year 10: keep the table, re-solve, or hindsight
    uncertainty  the expected stock return is uncertain; a policy that learns it
    regime       good and bad market states in the DP state
    hidden       the same states when they have to be inferred from returns
    review       saving reviewed every year, against fixed saving
    shortfall    penalise the size of a miss; against a cap, power utility, fixed mixes
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from gbda import analysis as an
from gbda import market, metrics, param_uncertainty, plots, regime, resolve, saving, shortfall, simulate
from gbda.analysis import Investor, World

RESULTS = Path("results")
FIGURES = Path("figures")
INVESTOR = Investor()
BASE = World()
B = market.BASE


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x: float, digits: int = 1) -> str:
    """Format a share as a percentage."""
    return "n/a" if x != x else f"{100 * x:.{digits}f}%"


def pp(x: float) -> str:
    """Format a difference of shares in percentage points."""
    return f"{100 * x:+.1f} pp"


def usd(x: float) -> str:
    """Format dollars to the nearest thousand."""
    return "n/a" if x != x else f"${x / 1000:,.0f}K"


def write(name: str, title: str, table: str, notes: str = "") -> None:
    """Write one results table and echo it to the console."""
    RESULTS.mkdir(exist_ok=True)
    text = f"## {title}\n\n{table}\n" + (f"\n{notes}\n" if notes else "")
    (RESULTS / f"{name}.md").write_text(text)
    print("\n" + text)


def history_shaped_world(block: int = 5) -> World:
    """Bootstrap of 1928–2025, rescaled to the base-case means and volatilities.

    ``block=5`` keeps runs of five consecutive years together; ``block=1`` draws
    single years independently (same one-year distribution, no dependence across years).
    """
    shaped = market.load_history().recentred(B)
    return World(B.shifted("History-shaped returns"), dist="bootstrap", history=shaped, block=block)


# --------------------------------------------------------------------------- #
# compare: every strategy on the same paths
# --------------------------------------------------------------------------- #
def section_compare(n_paths: int) -> dict[str, Any]:
    """Section 7 of the spec: strategy table, policy heatmaps, wealth and weight figures."""
    paths = BASE.paths(n_paths, INVESTOR.horizon)
    objectives = {
        "DP · goal probability": "goal",
        "DP · log utility": "log",
        "DP · power utility γ=2": "power:2",
        "DP · power utility γ=5": "power:5",
        "DP · power utility γ=10": "power:10",
    }
    solutions = {name: an.solve_policy(INVESTOR, BASE, obj) for name, obj in objectives.items()}
    policies: dict[str, simulate.Policy] = {name: simulate.dp_policy(sol) for name, sol in solutions.items()}
    policies |= {
        "Fixed 60/40": simulate.fixed_mix(0.6),
        "Fixed 100% stock": simulate.fixed_mix(1.0),
        "Fixed 100% bond": simulate.fixed_mix(0.0),
        "Glide path 90→30": an.glide(INVESTOR),
    }
    sims = {name: an.run(policy, INVESTOR, paths) for name, policy in policies.items()}
    rows = {name: metrics.summarize(sim.terminal, INVESTOR.goal, sim.weights) for name, sim in sims.items()}
    for name, sol in solutions.items():
        rows[name]["dp_value"] = sol.value_at(0, INVESTOR.w0) if name.endswith("goal probability") else float("nan")

    table = md_table(
        ["Strategy", "Success", "± s.e.", "10th pct", "Median", "90th pct", "Avg. wealth if missed",
         "CE (γ=2)", "CE (γ=5)", "Avg. stock"],
        [[name, pct(r["success"]), pct(r["success_se"], 2), usd(r["p10"]), usd(r["median"]), usd(r["p90"]),
          usd(r["mean_wealth_if_missed"]), usd(r["ce_gamma2"]), usd(r["ce_gamma5"]), pct(r["avg_stock_weight"], 0)]
         for name, r in rows.items()],
    )
    goal_row = rows["DP · goal probability"]
    write(
        "strategy_comparison",
        f"Strategy comparison — base case, {n_paths:,} forward paths",
        table,
        f"Goal ${INVESTOR.goal:,.0f} in {INVESTOR.horizon} years from ${INVESTOR.w0:,.0f} plus "
        f"${INVESTOR.contribution:,.0f} a year (needs {INVESTOR.required_return():.1%} a year). "
        f"DP in-sample estimate for the goal policy: {pct(goal_row['dp_value'])}; forward simulation: "
        f"{pct(goal_row['success'])} ± {pct(goal_row['success_se'], 2)}. CE = certainty equivalent of terminal wealth.",
    )

    # ---- figure: policy heatmaps -------------------------------------------- #
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharey=True)
    titles = {
        "DP · goal probability": "Goal probability",
        "DP · log utility": "Log utility",
        "DP · power utility γ=2": "Power utility, γ = 2",
        "DP · power utility γ=5": "Power utility, γ = 5",
        "DP · power utility γ=10": "Power utility, γ = 10",
    }
    mesh = None
    for ax, (name, sol) in zip(axes.ravel(), solutions.items()):
        reach = tuple(np.quantile(sims[name].wealth[:, :-1], q, axis=0) for q in (0.05, 0.95))
        mesh = plots.policy_heatmap(ax, sol.grid, sol.policy, titles[name], goal=INVESTOR.goal,
                                    w_range=(2e4, 2e6), reach=reach)
        ax.yaxis.set_major_formatter(plots.dollars)
        ax.set_yticks([2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6])
        ax.minorticks_off()
        ax.set_xticks(range(0, INVESTOR.horizon + 1, 5))
    for ax in axes[:, 0]:
        ax.set_ylabel("Wealth")
    note = axes[1, 2]
    note.axis("off")
    cax = note.inset_axes([0.05, 0.62, 0.9, 0.07])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=np.linspace(0, 1, 6))
    cb.ax.set_xticklabels([f"{int(100 * t)}%" for t in np.linspace(0, 1, 6)])
    cb.outline.set_visible(False)
    note.text(0.05, 0.80, "Stock weight", fontsize=10, fontweight="bold", transform=note.transAxes)
    note.plot([0.05, 0.17], [0.42, 0.42], color=plots.ORANGE, lw=1.4, transform=note.transAxes)
    note.text(0.21, 0.42, "5th–95th percentile of wealth\nunder that policy", va="center", fontsize=9,
              color=plots.INK_2, transform=note.transAxes)
    note.plot([0.05, 0.17], [0.22, 0.22], color=plots.INK, lw=1.0, ls=(0, (4, 3)), transform=note.transAxes)
    note.text(0.21, 0.22, f"goal ({plots.dollars(INVESTOR.goal)})", va="center", fontsize=9,
              color=plots.INK_2, transform=note.transAxes)
    fig.suptitle("Optimal stock weight by year and wealth, for five objectives", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plots.save(fig, FIGURES / "policy_heatmaps.png")

    # ---- figure: terminal wealth -------------------------------------------- #
    shown = {
        "DP · goal probability": plots.BLUE,
        "DP · power utility γ=5": plots.AQUA,
        "Fixed 60/40": plots.ORANGE,
        "Fixed 100% stock": plots.VIOLET,
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    bins = np.geomspace(1e5, 3e6, 90)
    centres = np.sqrt(bins[:-1] * bins[1:])
    for name, color in shown.items():
        density, _ = np.histogram(sims[name].terminal, bins=bins)
        ax.plot(centres, density / n_paths / np.diff(np.log(bins)), color=color, label=name)
    ax.axvline(INVESTOR.goal, color=plots.INK, lw=1.0, ls=(0, (4, 3)))
    ax.text(INVESTOR.goal * 0.97, ax.get_ylim()[1] * 0.96, "goal", fontsize=9, va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xticks([1e5, 2e5, 3e5, 5e5, 1e6, 2e6, 3e6])
    ax.xaxis.set_major_formatter(plots.dollars)
    ax.minorticks_off()
    ax.set_xlabel("Wealth after 20 years (log scale)")
    ax.set_ylabel("Density (per unit of log wealth)")
    ax.set_title("Distribution of final wealth")
    ax.legend(loc="upper right")

    ax = axes[1]
    order = list(rows)[::-1]
    for i, name in enumerate(order):
        r = rows[name]
        color = shown.get(name, plots.GREY)
        ax.plot([r["p10"], r["p90"]], [i, i], color=color, lw=2.5, solid_capstyle="round")
        ax.plot(r["median"], i, "o", color=color, ms=7, mec=plots.SURFACE, mew=1.5)
        ax.text(1.75e6, i, pct(r["success"], 0), va="center", ha="right", fontsize=9, color=plots.INK)
    ax.axvline(INVESTOR.goal, color=plots.INK, lw=1.0, ls=(0, (4, 3)))
    ax.set_yticks(range(len(order)), order)
    ax.set_xscale("log")
    ax.set_xlim(1.5e5, 1.9e6)
    ax.set_xticks([2e5, 3e5, 5e5, 1e6])
    ax.xaxis.set_major_formatter(plots.dollars)
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Final wealth: 10th percentile – median – 90th percentile")
    ax.set_title("Range of outcomes (right column: chance of reaching the goal)")
    fig.tight_layout()
    plots.save(fig, FIGURES / "terminal_wealth.png")

    # ---- figure: stock weight over time --------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
    years = np.arange(INVESTOR.horizon)
    ax = axes[0]
    lines = {
        "DP · goal probability": (plots.BLUE, "-"),
        "DP · log utility": (plots.YELLOW, "-"),
        "DP · power utility γ=2": (plots.AQUA, "-"),
        "DP · power utility γ=5": (plots.MAGENTA, "-"),
        "DP · power utility γ=10": (plots.VIOLET, "-"),
        "Glide path 90→30": (plots.GREY, (0, (4, 3))),
        "Fixed 60/40": (plots.ORANGE, (0, (1, 2))),
    }
    for name, (color, style) in lines.items():
        ax.plot(years, sims[name].weights.mean(axis=0), color=color, ls=style, label=name)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.set_xticks(range(0, INVESTOR.horizon, 5))
    ax.set_xlabel("Year")
    ax.set_ylabel("Average stock weight across paths")
    ax.set_title("Average stock weight by year")
    ax.legend(loc="lower left", ncol=2)

    ax = axes[1]
    goal_sim = sims["DP · goal probability"]
    hit = goal_sim.terminal >= INVESTOR.goal
    ax.plot(years, goal_sim.weights[hit].mean(axis=0), color=plots.BLUE, label="Paths that reached the goal")
    ax.plot(years, goal_sim.weights[~hit].mean(axis=0), color=plots.ORANGE, label="Paths that missed")
    ax.set_xticks(range(0, INVESTOR.horizon, 5))
    ax.set_xlabel("Year")
    ax.set_title("Goal policy: de-risks when ahead, stays in stocks when behind")
    ax.legend(loc="lower left")
    fig.tight_layout()
    plots.save(fig, FIGURES / "stock_weight_paths.png")

    return {
        "rows": rows,
        "weights_by_year": {name: sim.weights.mean(axis=0).tolist() for name, sim in sims.items()},
        "success_flags": {name: sim.terminal >= INVESTOR.goal for name, sim in sims.items()},
    }


# --------------------------------------------------------------------------- #
# assumptions: four ways to set expected returns
# --------------------------------------------------------------------------- #
def assumption_sets() -> list[market.Assumptions]:
    """The expected-return views compared in the README."""
    history = market.load_history()
    return [
        B,
        market.historical_assumptions(history),
        market.historical_assumptions(history, 1928, 1975),
        market.historical_assumptions(history, 1976, 2025),
        market.historical_assumptions(history, 2000, 2025),
        # Dividend yield 1.08% (S&P 500, 10 Nov 2026) and 10-year Treasury yield 4.84–4.98%
        # (2–10 Nov 2026) are observed; earnings growth and valuation change are judgment inputs.
        market.building_block_assumptions(0.011, 0.050, -0.010, 0.049,
                                          label="Building blocks (1.1% + 5.0% − 1.0%; bond yield 4.9%)"),
        market.equilibrium_assumptions(label="Market equilibrium (60/40 market, cash 3%)"),
        # J.P. Morgan 2027 Long-Term Capital Market Assumptions (data as of Sep 2026): U.S. large cap
        # 7.5%; the only bond figure on the public page is global aggregate, 5.3%.
        market.compound_forecast_assumptions(0.075, 0.053, label="Published outlook A (7.5% / 5.3% compound)"),
        # Vanguard Capital Markets Model, 30 Sep 2026 run, 10-year ranges: U.S. equities 4.8–6.8%,
        # U.S. aggregate bonds 5.1–6.1%; midpoints used.
        market.compound_forecast_assumptions(0.058, 0.056, label="Published outlook B (5.8% / 5.6% compound)"),
    ]


def section_assumptions(n_paths: int) -> dict[str, Any]:
    """Success probability and required saving under each expected-return view."""
    sets = assumption_sets()
    worlds = [World(a) for a in sets]
    cases = an.run_parallel(an.goal_case, [(INVESTOR, w, None, 1.0, n_paths) for w in worlds])
    save_dp = an.run_parallel(an.required_saving, [(INVESTOR, w, 0.75, "dp") for w in worlds])
    save_6040 = an.run_parallel(an.required_saving, [(INVESTOR, w, 0.75, "60/40") for w in worlds])
    rows = []
    for a, c, s_dp, s_fix in zip(sets, cases, save_dp, save_6040):
        rows.append({
            "label": a.label, "mu_stock": a.mu_stock, "mu_bond": a.mu_bond, "sigma_stock": a.sigma_stock,
            "sigma_bond": a.sigma_bond, "rho": a.rho,
            "stock_compound": market.compound_from_arithmetic(a.mu_stock, a.sigma_stock),
            "bond_compound": market.compound_from_arithmetic(a.mu_bond, a.sigma_bond),
            "dp_success": c["success"], "fixed_60_40_success": c["fixed_60_40_success"],
            "avg_stock_weight": c["avg_stock_weight"], "saving_dp": s_dp, "saving_60_40": s_fix,
        })
    table = md_table(
        ["Expected-return view", "Stock mean / vol", "Bond mean / vol", "Stock / bond compound",
         "Dynamic policy", "Fixed 60/40", "Saving for 75%: dynamic", "Saving for 75%: 60/40"],
        [[r["label"], f"{pct(r['mu_stock'])} / {pct(r['sigma_stock'], 0)}", f"{pct(r['mu_bond'])} / {pct(r['sigma_bond'], 0)}",
          f"{pct(r['stock_compound'])} / {pct(r['bond_compound'])}", pct(r["dp_success"]), pct(r["fixed_60_40_success"]),
          f"${r['saving_dp']:,.0f}", f"${r['saving_60_40']:,.0f}"] for r in rows],
    )
    write("assumption_sets", "Same plan, different expected-return views", table,
          "Success = probability of reaching the goal with $10,000 saved a year. "
          "Saving for 75% = yearly saving at which the success probability reaches 75%. "
          "Means are arithmetic; historical rows use that window's own volatilities and correlation, the others use the base-case ones.")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    order = np.argsort([r["dp_success"] for r in rows])
    for i, j in enumerate(order):
        r = rows[j]
        ax.plot([r["fixed_60_40_success"], r["dp_success"]], [i, i], color=plots.AXIS, lw=2)
        ax.plot(r["fixed_60_40_success"], i, "o", color=plots.ORANGE, ms=8, mec=plots.SURFACE, mew=1.5,
                label="Fixed 60/40" if i == 0 else None)
        ax.plot(r["dp_success"], i, "o", color=plots.BLUE, ms=8, mec=plots.SURFACE, mew=1.5,
                label="Dynamic policy (goal probability)" if i == 0 else None)
        ax.text(r["dp_success"] + 0.02, i, pct(r["dp_success"], 0), va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [rows[j]["label"] for j in order])
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Probability of reaching the goal")
    ax.set_title("The answer depends mostly on which expected returns you believe")
    ax.legend(loc="lower right")
    plots.save(fig, FIGURES / "assumption_sets.png")
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# sensitivity: one assumption at a time
# --------------------------------------------------------------------------- #
def section_sensitivity(n_paths: int) -> dict[str, Any]:
    """Move one input at a time; report re-optimised and 'planned on the base case' results."""
    cases: list[tuple[str, str, str, World]] = [
        ("Stock expected return", "−1 pp (6%)", "low", World(B.shifted("s-", mu_stock=B.mu_stock - 0.01))),
        ("Stock expected return", "+1 pp (8%)", "high", World(B.shifted("s+", mu_stock=B.mu_stock + 0.01))),
        ("Bond expected return", "−1 pp (2%)", "low", World(B.shifted("b-", mu_bond=B.mu_bond - 0.01))),
        ("Bond expected return", "+1 pp (4%)", "high", World(B.shifted("b+", mu_bond=B.mu_bond + 0.01))),
        ("Stock volatility", "−20% (12.8%)", "low", World(B.shifted("sv-", sigma_stock=B.sigma_stock * 0.8))),
        ("Stock volatility", "+20% (19.2%)", "high", World(B.shifted("sv+", sigma_stock=B.sigma_stock * 1.2))),
        ("Bond volatility", "−20% (4%)", "low", World(B.shifted("bv-", sigma_bond=B.sigma_bond * 0.8))),
        ("Bond volatility", "+20% (6%)", "high", World(B.shifted("bv+", sigma_bond=B.sigma_bond * 1.2))),
        ("Stock–bond correlation", "−0.2", "low", World(B.shifted("r-", rho=-0.2))),
        ("Stock–bond correlation", "+0.6", "high", World(B.shifted("r+", rho=0.6))),
        ("Tail shape", "Student-t, 5 d.f.", "shape", World(B, dist="student_t", df=5)),
        ("Tail shape", "Student-t, 3 d.f.", "shape", World(B, dist="student_t", df=3)),
        ("Tail shape", "Historical years, drawn independently", "shape", history_shaped_world(block=1)),
        ("Tail shape", "Historical years, in 5-year blocks", "shape", history_shaped_world(block=5)),
    ]
    base = an.goal_case(INVESTOR, BASE, None, 1.0, n_paths)
    reopt = an.run_parallel(an.goal_case, [(INVESTOR, w, None, 1.0, n_paths) for *_, w in cases])
    planned = an.run_parallel(an.goal_case, [(INVESTOR, w, BASE, 1.0, n_paths) for *_, w in cases])
    rows = []
    for (group, change, side, _), r, p in zip(cases, reopt, planned):
        rows.append({
            "group": group, "change": change, "side": side,
            "dp_success": r["success"], "dp_delta": r["success"] - base["success"],
            "planned_on_base_success": p["success"], "cost_of_wrong_plan": r["success"] - p["success"],
            "fixed_60_40_success": r["fixed_60_40_success"],
            "fixed_60_40_delta": r["fixed_60_40_success"] - base["fixed_60_40_success"],
            "avg_stock_weight": r["avg_stock_weight"], "avg_stock_delta": r["avg_stock_weight"] - base["avg_stock_weight"],
            "p05": r["p05"], "p10": r["p10"],
        })
    table = md_table(
        ["Input", "Change", "Dynamic policy", "Δ vs base", "Base-case policy in this world", "Fixed 60/40", "Δ vs base",
         "Avg. stock weight (Δ)", "5th pct wealth"],
        [["Base case", "—", pct(base["success"]), "—", "—", pct(base["fixed_60_40_success"]), "—",
          pct(base["avg_stock_weight"], 0), usd(base["p05"])]]
        + [[r["group"], r["change"], pct(r["dp_success"]), pp(r["dp_delta"]), pct(r["planned_on_base_success"]),
            pct(r["fixed_60_40_success"]), pp(r["fixed_60_40_delta"]),
            f"{pct(r['avg_stock_weight'], 0)} ({100 * r['avg_stock_delta']:+.0f})", usd(r["p05"])] for r in rows],
    )
    se = np.sqrt(2 * base["success"] * (1 - base["success"]) / n_paths)
    write("sensitivity", "Sensitivity of the success probability to each assumption", table,
          "Dynamic policy = re-optimised for the changed assumption. 'Base-case policy in this world' = the policy "
          "optimised for the base case, scored when the changed assumption is the truth. "
          "5th pct wealth is for the dynamic policy. "
          f"One standard error of a difference between rows is at most {100 * se:.2f} pp "
          "(less for the parametric rows, which share random shocks).")

    # tornado
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    param_groups = [g for g in groups if g != "Tail shape"]
    param_groups.sort(key=lambda g: max(abs(r["dp_delta"]) for r in groups[g]))
    bars: list[tuple[str, float, str]] = []
    for r in sorted(groups["Tail shape"], key=lambda r: abs(r["dp_delta"])):
        bars.append((f"Tails: {r['change']}", r["dp_delta"], "shape"))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = 0
    ticks, labels = [], []
    colors = {"low": plots.ORANGE, "high": plots.BLUE, "shape": plots.VIOLET}
    for label, delta, side in bars:
        ax.barh(y, delta * 100, height=0.5, color=colors[side])
        ax.text(delta * 100 + (0.4 if delta >= 0 else -0.4), y, pp(delta), va="center",
                ha="left" if delta >= 0 else "right", fontsize=9)
        ticks.append(y); labels.append(label); y += 1
    for g in param_groups:
        for r in groups[g]:
            ax.barh(y, r["dp_delta"] * 100, height=0.5, color=colors[r["side"]])
            d = r["dp_delta"]
            ax.text(d * 100 + (0.4 if d >= 0 else -0.4), y, f"{r['change']}:  {pp(d)}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=9)
        ticks.append(y); labels.append(g); y += 1
    ax.axvline(0, color=plots.AXIS, lw=1)
    ax.set_yticks(ticks, labels)
    ax.grid(axis="y", visible=False)
    lim = max(abs(r["dp_delta"]) for r in rows) * 100 * 1.75
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(f"Change in success probability of the dynamic policy, percentage points (base case {pct(base['success'])})")
    ax.set_title("Which assumption moves the answer most")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("low", "high", "shape")]
    ax.legend(handles, ["Input lower", "Input higher", "Fatter tails, same mean and volatility"], loc="lower right")
    plots.save(fig, FIGURES / "sensitivity_tornado.png")
    return {"base": base, "rows": rows}


# --------------------------------------------------------------------------- #
# dynamic: where does a dynamic policy help most?
# --------------------------------------------------------------------------- #
def section_dynamic(n_paths: int) -> dict[str, Any]:
    """Dynamic vs static by distance to goal, horizon and funding pattern."""
    r_req = INVESTOR.required_return()
    growth = lambda t: (1 + r_req) ** t  # noqa: E731
    annuity = lambda t: (growth(t) - 1) / r_req  # noqa: E731

    goals = list(range(300_000, 800_001, 50_000))
    goal_jobs = [(replace(INVESTOR, goal=float(g)), BASE, None, 1.0, n_paths) for g in goals]
    horizons = [10, 20, 30]
    horizon_investors = [
        replace(INVESTOR, horizon=t, goal=round(INVESTOR.w0 * growth(t) + INVESTOR.contribution * annuity(t), -3))
        for t in horizons
    ]
    t = INVESTOR.horizon
    funding = {
        "Lump sum only": replace(INVESTOR, w0=INVESTOR.goal / growth(t), contribution=0.0),
        "Base plan ($50K + $10K a year)": INVESTOR,
        "Savings only": replace(INVESTOR, w0=0.0, contribution=INVESTOR.goal / annuity(t)),
    }
    jobs = goal_jobs + [(inv, BASE, None, 1.0, n_paths) for inv in horizon_investors] \
        + [(inv, BASE, None, 1.0, n_paths) for inv in funding.values()]
    res = an.run_parallel(an.goal_case, jobs)
    by_goal = res[: len(goals)]
    by_horizon = res[len(goals): len(goals) + len(horizons)]
    by_funding = res[len(goals) + len(horizons):]

    def row(label: str, inv: Investor, r: dict[str, float]) -> list[str]:
        return [label, f"{inv.required_return():.1%}", pct(r["success"]),
                f"{pct(r['best_fixed_success'])} ({int(round(100 * r['best_fixed_weight']))}% stock)",
                pct(r["glide_success"]), pct(r["fixed_60_40_success"]), pp(r["success"] - r["best_fixed_success"]),
                pct(r["avg_stock_weight"], 0)]

    headers = ["Case", "Return the plan needs", "Dynamic policy", "Best fixed mix", "Glide path 90→30", "Fixed 60/40",
               "Dynamic − best fixed", "Dynamic avg. stock"]
    rows = [row(f"Goal {plots.dollars(g)}", job[0], r) for g, job, r in zip(goals, goal_jobs, by_goal)]
    rows += [row(f"Horizon {inv.horizon} years, goal {plots.dollars(inv.goal)}", inv, r)
             for inv, r in zip(horizon_investors, by_horizon)]
    rows += [row(f"{name} (W0 {plots.dollars(inv.w0)}, {plots.dollars(inv.contribution)} a year)", inv, r)
             for (name, inv), r in zip(funding.items(), by_funding)]
    write("dynamic_vs_static", "Where the dynamic policy adds most", md_table(headers, rows),
          "Horizon and funding rows hold the required return at the base plan's level, so only the shape of the plan changes.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    x = np.array(goals)
    series = {
        "Dynamic policy": (np.array([r["success"] for r in by_goal]), plots.BLUE, "-"),
        "Best fixed mix": (np.array([r["best_fixed_success"] for r in by_goal]), plots.ORANGE, "-"),
        "Glide path 90→30": (np.array([r["glide_success"] for r in by_goal]), plots.AQUA, "-"),
        "Fixed 60/40": (np.array([r["fixed_60_40_success"] for r in by_goal]), plots.GREY, (0, (4, 3))),
    }
    for name, (yv, color, style) in series.items():
        ax.plot(x, yv, color=color, ls=style, marker="o", ms=4, label=name)
    ax.xaxis.set_major_formatter(plots.dollars)
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Goal (20 years, \$50K start, \$10K a year)")
    ax.set_ylabel("Probability of reaching the goal")
    ax.set_title("Success probability by goal")
    ax.legend(loc="upper right")
    ax = axes[1]
    adv = (series["Dynamic policy"][0] - series["Best fixed mix"][0]) * 100
    ax.bar(x, adv, width=22_000, color=plots.BLUE)
    for xi, a in zip(x, adv):
        ax.text(xi, a + 0.2, f"{a:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, adv.max() * 1.12)
    ax.xaxis.set_major_formatter(plots.dollars)
    ax.set_xlabel("Goal")
    ax.set_ylabel("Percentage points")
    ax.grid(axis="x", visible=False)
    ax.set_title("Dynamic policy minus best fixed mix")
    fig.tight_layout()
    plots.save(fig, FIGURES / "dynamic_vs_static.png")
    return {"goals": goals, "by_goal": by_goal, "by_horizon": by_horizon, "by_funding": by_funding,
            "horizon_goals": [inv.goal for inv in horizon_investors],
            "funding": {k: {"w0": v.w0, "contribution": v.contribution} for k, v in funding.items()}}


# --------------------------------------------------------------------------- #
# cap: cost of a ceiling on the stock weight
# --------------------------------------------------------------------------- #
def section_cap(n_paths: int) -> dict[str, Any]:
    """Success probability and downside under a maximum stock weight."""
    caps = [0.2, 0.4, 0.6, 0.8, 1.0]
    res = an.run_parallel(an.goal_case, [(INVESTOR, BASE, None, c, n_paths) for c in caps])
    free = res[-1]
    table = md_table(
        ["Max stock weight", "Dynamic policy", "Cost of the cap", "Best fixed mix under the cap", "5th pct wealth",
         "10th pct wealth", "Median wealth", "Avg. wealth if missed", "Avg. stock"],
        [[pct(c, 0), pct(r["success"]), pp(r["success"] - free["success"]), pct(r["best_fixed_success"]),
          usd(r["p05"]), usd(r["p10"]), usd(r["median"]), usd(r["mean_wealth_if_missed"]), pct(r["avg_stock_weight"], 0)]
         for c, r in zip(caps, res)],
    )
    write("stock_cap", "Cost of a ceiling on the stock weight (goal-probability policy)", table)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    x = np.array(caps) * 100
    ax = axes[0]
    ax.plot(x, [r["success"] for r in res], color=plots.BLUE, marker="o", label="Dynamic policy")
    ax.plot(x, [r["best_fixed_success"] for r in res], color=plots.ORANGE, marker="o", label="Best fixed mix")
    for xi, r in zip(x, res):
        ax.text(xi, r["success"] + 0.03, pct(r["success"], 0), ha="center", fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.set_xticks(x, [f"{int(v)}%" for v in x])
    ax.set_xlabel("Maximum stock weight")
    ax.set_title("Probability of reaching the goal")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(x, [r["p05"] for r in res], color=plots.BLUE, marker="o", label="5th percentile")
    ax.plot(x, [r["mean_wealth_if_missed"] for r in res], color=plots.AQUA, marker="o", label="Average when the goal is missed")
    ax.yaxis.set_major_formatter(plots.dollars)
    ax.set_xticks(x, [f"{int(v)}%" for v in x])
    ax.set_xlabel("Maximum stock weight")
    ax.set_title("Final wealth in bad outcomes (dynamic policy)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    plots.save(fig, FIGURES / "stock_cap.png")
    return {"caps": caps, "rows": res}


# --------------------------------------------------------------------------- #
# precision: how much Monte Carlo is enough
# --------------------------------------------------------------------------- #
def section_precision(n_paths: int, flags: dict[str, np.ndarray] | None) -> dict[str, Any]:
    """Forward-path error (paired vs independent) and DP scenario-set error."""
    paths = BASE.paths(n_paths, INVESTOR.horizon)
    if flags is None:
        goal_sol = an.solve_policy(INVESTOR, BASE, "goal")
        g5_sol = an.solve_policy(INVESTOR, BASE, "power:5")
        flags = {
            "DP · goal probability": an.run(simulate.dp_policy(goal_sol), INVESTOR, paths).terminal >= INVESTOR.goal,
            "DP · power utility γ=5": an.run(simulate.dp_policy(g5_sol), INVESTOR, paths).terminal >= INVESTOR.goal,
            "Fixed 60/40": an.run(simulate.fixed_mix(0.6), INVESTOR, paths).terminal >= INVESTOR.goal,
            "Fixed 100% stock": an.run(simulate.fixed_mix(1.0), INVESTOR, paths).terminal >= INVESTOR.goal,
            "Glide path 90→30": an.run(an.glide(INVESTOR), INVESTOR, paths).terminal >= INVESTOR.goal,
        }
    capped = an.solve_policy(INVESTOR, BASE, "goal", max_stock=0.8)
    flags = dict(flags) | {
        "DP · goal, stock ≤ 80%": an.run(simulate.dp_policy(capped), INVESTOR, paths).terminal >= INVESTOR.goal,
        "Fixed 70/30": an.run(simulate.fixed_mix(0.7), INVESTOR, paths).terminal >= INVESTOR.goal,
    }
    pairs = [
        ("DP · goal probability", "Fixed 60/40"),
        ("DP · goal probability", "Fixed 100% stock"),
        ("DP · goal probability", "DP · goal, stock ≤ 80%"),
        ("Fixed 70/30", "Glide path 90→30"),
        ("Fixed 60/40", "Glide path 90→30"),
        ("DP · power utility γ=5", "Glide path 90→30"),
    ]
    pair_rows = []
    for a, b in pairs:
        diff, se_p, se_i = metrics.paired_difference(flags[a], flags[b])
        n = flags[a].size
        pair_rows.append({
            "a": a, "b": b, "difference": diff, "se_paired": se_p, "se_independent": se_i,
            "paths_needed_paired": metrics.paths_needed(diff, se_p * np.sqrt(n)),
            "paths_needed_independent": metrics.paths_needed(diff, se_i * np.sqrt(n)),
            "paths_for_half_pp_paired": metrics.paths_needed(0.005, se_p * np.sqrt(n)),
            "paths_for_half_pp_independent": metrics.paths_needed(0.005, se_i * np.sqrt(n)),
        })
    table = md_table(
        ["Comparison", "Difference in success", "s.e. (same paths)", "s.e. (separate paths)",
         "Paths to tell apart: same / separate", "Paths for ±0.5 pp: same / separate"],
        [[f"{r['a']} vs {r['b']}", pp(r["difference"]), f"{100 * r['se_paired']:.2f} pp",
          f"{100 * r['se_independent']:.2f} pp",
          f"{r['paths_needed_paired']:,} / {r['paths_needed_independent']:,}",
          f"{r['paths_for_half_pp_paired']:,} / {r['paths_for_half_pp_independent']:,}"]
         for r in pair_rows],
    )
    p0 = float(np.mean(flags["DP · goal probability"]))
    write("mc_precision_paths", f"Forward-simulation precision ({n_paths:,} paths)", table,
          "'Paths to tell apart' = paths at which the difference is 1.96 standard errors from zero; "
          "'paths for ±0.5 pp' = paths at which the 95% interval on the difference is ±0.5 pp. "
          f"A single success probability near {pct(p0, 0)} has a standard error of "
          f"{100 * np.sqrt(p0 * (1 - p0) / 10_000):.2f} pp at 10,000 paths and "
          f"{100 * np.sqrt(p0 * (1 - p0) / n_paths):.2f} pp at {n_paths:,}.")

    # DP scenario-set error
    sizes, seeds = [500, 2000, 8000], list(range(11, 19))
    jobs = [(INVESTOR, BASE, m, s, mm) for mm in (False, True) for m in sizes for s in seeds]
    draws = an.run_parallel(an.dp_sampling_case, jobs)
    dp_rows = []
    for mm in (False, True):
        for m in sizes:
            sel = [d for d in draws if d["m"] == m and bool(d["moment_match"]) == mm]
            est = np.array([d["dp_value"] for d in sel])
            real = np.array([d["success"] for d in sel])
            dp_rows.append({"m": m, "moment_match": mm, "dp_mean": est.mean(), "dp_std": est.std(ddof=1),
                            "dp_min": est.min(), "dp_max": est.max(), "forward_mean": real.mean(),
                            "forward_min": real.min(), "forward_max": real.max()})
    table = md_table(
        ["Scenario set", "Scenarios", "DP's own estimate: range over 8 draws", "Std. dev.",
         "Same policies, forward-simulated: range"],
        [["Moment-matched" if r["moment_match"] else "Plain random", f"{r['m']:,}",
          f"{pct(r['dp_min'])} – {pct(r['dp_max'])}", f"{100 * r['dp_std']:.1f} pp",
          f"{pct(r['forward_min'], 2)} – {pct(r['forward_max'], 2)}"] for r in dp_rows],
    )
    write("mc_precision_dp", "How much the DP scenario set matters", table,
          "Each row: 8 different scenario draws, each giving a policy and the DP's own estimate of its success "
          "probability; every policy is then scored on the same 100,000 forward paths.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    a, b = "Fixed 70/30", "Glide path 90→30"
    other = BASE.paths(n_paths, INVESTOR.horizon, seed=an.SIM_SEED + 1)
    b_separate = an.run(an.glide(INVESTOR), INVESTOR, other).terminal >= INVESTOR.goal
    ns = np.unique(np.geomspace(500, n_paths, 60).astype(int))
    for label, y_b, color in (("Separate paths for each strategy", b_separate, plots.ORANGE),
                              ("Same paths for both", flags[b], plots.BLUE)):
        d = flags[a].astype(float) - y_b.astype(float)
        mean = np.array([d[:n].mean() for n in ns]) * 100
        se = np.array([d[:n].std(ddof=1) / np.sqrt(n) for n in ns]) * 100
        ax.fill_between(ns, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.18, lw=0)
        ax.plot(ns, mean, color=color, label=label)
    ax.axhline(0, color=plots.AXIS, lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 18)
    ax.set_xticks([1e3, 1e4, 1e5], ["1,000", "10,000", "100,000"])
    ax.minorticks_off()
    ax.set_xlabel("Number of forward paths")
    ax.set_ylabel("Estimated difference in success (pp), with 95% band")
    ax.set_title(f"{a} minus {b}: estimate as paths are added")
    ax.legend(loc="lower right")

    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, m in enumerate(sizes):
        for mm, color, off in ((False, plots.ORANGE, -0.2), (True, plots.BLUE, 0.0)):
            sel = [d for d in draws if d["m"] == m and bool(d["moment_match"]) == mm]
            ax.scatter(i + off + rng.uniform(-0.05, 0.05, len(sel)), [100 * d["dp_value"] for d in sel], s=36,
                       color=color, edgecolor=plots.SURFACE, lw=1,
                       label=("DP estimate, moment-matched scenarios" if mm else "DP estimate, plain random scenarios") if i == 0 else None)
        sel = [d for d in draws if d["m"] == m]
        ax.scatter(i + 0.2 + rng.uniform(-0.05, 0.05, len(sel)), [100 * d["success"] for d in sel], s=36,
                   color=plots.AQUA, edgecolor=plots.SURFACE, lw=1,
                   label="Same policies on 100,000 fresh paths" if i == 0 else None)
    ax.set_xticks(range(len(sizes)), [f"{m:,} scenarios" for m in sizes])
    ax.set_ylabel("Success probability (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("The DP's estimate is noisy; the policy it produces is not")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plots.save(fig, FIGURES / "mc_precision.png")
    return {"pairs": pair_rows, "dp_sampling": dp_rows}


# --------------------------------------------------------------------------- #
# saving: success as a function of yearly saving
# --------------------------------------------------------------------------- #
def section_saving(n_paths: int) -> dict[str, Any]:
    """Success probability vs yearly saving, base case and stock return 1 pp lower."""
    savings = list(range(4_000, 22_001, 2_000))
    low_world = World(B.shifted("Stock return 6%", mu_stock=B.mu_stock - 0.01))
    jobs = [(replace(INVESTOR, contribution=float(c)), w, None, 1.0, n_paths) for w in (BASE, low_world) for c in savings]
    res = an.run_parallel(an.goal_case, jobs)
    base, low = res[: len(savings)], res[len(savings):]
    table = md_table(
        ["Yearly saving", "Dynamic, base case", "Fixed 60/40, base case", "Dynamic, stock return 6%", "Fixed 60/40, stock return 6%"],
        [[f"${c:,}", pct(b["success"]), pct(b["fixed_60_40_success"]), pct(l["success"]), pct(l["fixed_60_40_success"])]
         for c, b, l in zip(savings, base, low)],
    )
    write("saving_curve", "Success probability by yearly saving", table)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.array(savings)
    ax.plot(x, [r["success"] for r in base], color=plots.BLUE, marker="o", ms=4, label="Dynamic policy, base case")
    ax.plot(x, [r["success"] for r in low], color=plots.BLUE, ls=(0, (4, 3)), marker="o", ms=4, label="Dynamic policy, stock return 1 pp lower")
    ax.plot(x, [r["fixed_60_40_success"] for r in base], color=plots.ORANGE, marker="o", ms=4, label="Fixed 60/40, base case")
    ax.plot(x, [r["fixed_60_40_success"] for r in low], color=plots.ORANGE, ls=(0, (4, 3)), marker="o", ms=4, label="Fixed 60/40, stock return 1 pp lower")
    ax.axhline(0.75, color=plots.AXIS, lw=1)
    ax.text(x[0], 0.76, "75%", fontsize=9, color=plots.INK_2, va="bottom")
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(plots.dollars)
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Yearly saving")
    ax.set_ylabel("Probability of reaching $500K in 20 years")
    ax.set_title("How much saving buys how much certainty")
    ax.legend(loc="lower right")
    plots.save(fig, FIGURES / "saving_curve.png")
    return {"savings": savings, "base": base, "stock_minus_1pp": low}


# --------------------------------------------------------------------------- #
# When the assumptions change: re-solve, parameter uncertainty, regimes
# --------------------------------------------------------------------------- #
SHOCKS = [
    resolve.Shock("Stock return 7% → 5% from year 10", 10, {"mu_stock": 0.05}, (0, 3)),
    resolve.Shock("Stock return 7% → 3% from year 10", 10, {"mu_stock": 0.03}, (0, 3)),
    resolve.Shock("Stock return 7% → 5% from year 5", 5, {"mu_stock": 0.05}, (0, 3)),
    resolve.Shock("Stock return 7% → 9% from year 10", 10, {"mu_stock": 0.09}, (0, 3)),
    resolve.Shock("Stock volatility 16% → 24% from year 10", 10, {"sigma_stock": 0.24}, (0, 3)),
]
KEEP, RESOLVE, LATE, HINDSIGHT = ("Keep the original table", "Re-solve every year",
                                  "Re-solve, change noticed 3 years late", "Hindsight: changes known from the start")


def section_resolve(n_paths: int) -> dict[str, Any]:
    """3.1: re-solve when the assumptions change, against keeping the table and against hindsight."""
    # The first shock re-solves literally every year; the others only when the belief changes,
    # which gives identical tables (optimality principle; checked here and in the tests).
    runs = [resolve.experiment(INVESTOR, B, shock, n_paths, annual=(i == 0)) for i, shock in enumerate(SHOCKS)]
    assert all(all(r["tail_checks"].values()) for r in runs), "a re-solve with unchanged assumptions changed the table"
    promised = an.solve_policy(INVESTOR, BASE).value_at(0, INVESTOR.w0)
    catch_up = resolve.catch_up_saving(INVESTOR, B, SHOCKS[0], promised)

    main = runs[0]["rows"]
    order = [KEEP, RESOLVE, LATE, HINDSIGHT, "Fixed 100% stock", "Fixed 60/40"]
    table = md_table(
        ["Approach", "Success", "± s.e.", "vs keeping the table", "Avg. stock weight, years 10–19",
         "Table's own forecast when it acts"],
        [[name, pct(main[name]["success"]), pct(main[name]["success_se"], 2),
          "—" if name == KEEP else f"{pp(main[name]['vs_original'])} ± {100 * main[name]['vs_original_se']:.2f}",
          pct(main[name]["avg_stock_after_change"], 0), pct(main[name]["forecast_at_change"])]
         for name in order],
    )
    summary = md_table(
        ["Change", "Keep the table", "Re-solve at once", "Re-solve 3 years late", "Hindsight",
         "Original table's forecast at the change", "Re-solved forecast"],
        [[r["shock"]["label"], pct(r["rows"][KEEP]["success"]),
          f"{pct(r['rows'][RESOLVE]['success'])} ({pp(r['rows'][RESOLVE]['vs_original'])})",
          f"{pct(r['rows'][LATE]['success'])} ({pp(r['rows'][LATE]['vs_original'])})",
          f"{pct(r['rows'][HINDSIGHT]['success'])} ({pp(r['rows'][HINDSIGHT]['vs_original'])})",
          pct(r["rows"][KEEP]["forecast_at_change"]), pct(r["rows"][RESOLVE]["forecast_at_change"])]
         for r in runs],
    )
    write(
        "resolve",
        f"Re-solving when the assumptions change — {n_paths:,} forward paths",
        f"### {SHOCKS[0].label}\n\n{table}\n\n### Every change tested\n\n{summary}",
        "All approaches run on the same paths, which follow the true (changing) assumptions. "
        "Differences against keeping the table are paired (same paths) with their standard error. "
        "\"Forecast\" is the success probability the table itself reports, averaged over paths, in the year the "
        "change is acted on (year 10, or 13 when noticed late); compare it with the success that then happens. "
        f"The first change was re-solved every year; re-solves in years before the change reproduced the original "
        f"table exactly. To get back to the original plan's {pct(promised)} after the first change, saving has to rise "
        f"from ${INVESTOR.contribution:,.0f} to ${round(catch_up, -2):,.0f} a year for years 10–19 (re-solved with the new saving).",
    )

    # ---- figure ---------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    pairs = [("Original table", main[KEEP]), ("Re-solved in year 10", main[RESOLVE])]
    for i, (label, r) in enumerate(pairs[::-1]):
        f, s = r["forecast_at_change"], r["success"]
        ax.plot([s, f], [i, i], color=plots.AXIS, lw=2.5, zorder=1)
        ax.plot(f, i, "o", ms=9, mfc=plots.SURFACE, mec=plots.BLUE, mew=2, zorder=2)
        ax.plot(s, i, "o", ms=9, color=plots.BLUE, mec=plots.SURFACE, mew=1.5, zorder=3)
        ax.text(f, i + 0.17, f"says {pct(f, 0)}", ha="center", fontsize=9, color=plots.INK_2)
        ax.text(s, i - 0.3, f"gets {pct(s, 0)}", ha="center", fontsize=9, color=plots.INK)
    ax.set_yticks([0, 1], [p[0] for p in pairs[::-1]])
    ax.set_ylim(-0.7, 1.6)
    ax.set_xlim(0.45, 0.67)
    ax.set_xticks([0.45, 0.50, 0.55, 0.60, 0.65])
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Chance of reaching the goal")
    ax.set_title("Stock return drops to 5% in year 10:\nre-solving fixes the forecast")

    ax = axes[1]
    marks = [(RESOLVE, "Re-solve at once", plots.AQUA, "o"), (LATE, "Re-solve 3 years late", plots.YELLOW, "s"),
             (HINDSIGHT, "Hindsight (upper bound)", plots.MAGENTA, "D")]
    labels = [r["shock"]["label"] for r in runs][::-1]
    for j, (key, label, color, marker) in enumerate(marks):
        y = np.arange(len(runs))[::-1] + (1 - j) * 0.22
        x = np.array([100 * r["rows"][key]["vs_original"] for r in runs])
        err = np.array([196 * r["rows"][key]["vs_original_se"] for r in runs])
        ax.errorbar(x, y, xerr=err, fmt=marker, color=color, ms=7, mec=plots.SURFACE, mew=1, elinewidth=1.2,
                    capsize=0, label=label)
    ax.axvline(0, color=plots.INK, lw=1)
    ax.set_yticks(range(len(runs)), labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Gain in success probability over keeping the original table (pp, ±95%)")
    ax.set_title("…but barely changes the outcome")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plots.save(fig, FIGURES / "resolve.png")
    return {"runs": [{k: v for k, v in r.items() if k != "weights_by_year"} for r in runs],
            "promised": promised, "catch_up_saving": catch_up}


S_VALUES = (0.0, 0.01, 0.02, 0.03)


def section_uncertainty(n_paths: int) -> dict[str, Any]:
    """3.3: each path has its own expected stock return; a policy that learns it vs one that ignores it."""
    jobs = [(INVESTOR, param_uncertainty.UncertainStock(B, s), n_paths) for s in S_VALUES]
    res = an.run_parallel(param_uncertainty.case, jobs)
    learn, ignore = "Learns the expected return", "Ignores the uncertainty"
    table = md_table(
        ["Uncertainty s", "Dynamic, learns μ", "Dynamic, ignores it", "Difference", "Fixed 100% stock", "Fixed 60/40",
         "DP's own estimate (learns / ignores)", "Dynamic success, lowest / highest μ decile", "Avg. stock (learns / ignores)"],
        [[f"{100 * r['s']:.0f}%", pct(r["rows"][learn]["success"]), pct(r["rows"][ignore]["success"]),
          f"{100 * r['rows'][learn]['vs_ignore']:+.2f} ± {100 * r['rows'][learn]['vs_ignore_se']:.2f} pp",
          pct(r["rows"]["Fixed 100% stock"]["success"]), pct(r["rows"]["Fixed 60/40"]["success"]),
          f"{pct(r['rows'][learn]['dp_value'])} / {pct(r['rows'][ignore]['dp_value'])}",
          f"{pct(r['rows'][learn]['success_lowest_mu_decile'], 0)} / {pct(r['rows'][learn]['success_highest_mu_decile'], 0)}",
          f"{pct(r['rows'][learn]['avg_stock_weight'], 0)} / {pct(r['rows'][ignore]['avg_stock_weight'], 0)}"]
         for r in res],
    )
    horizons = (1, 5, 10, 20, 30)
    shares = {s: param_uncertainty.variance_share(s, B.sigma_stock, horizons, B.mu_stock) for s in S_VALUES[1:]}
    share_table = md_table(["s"] + [f"{t} years" for t in horizons],
                           [[f"{100 * s:.0f}%"] + [pct(v[t], 0) for t in horizons] for s, v in shares.items()])
    write(
        "param_uncertainty",
        f"Parameter uncertainty in the expected stock return — {n_paths:,} forward paths",
        f"{table}\n\n### Share of the variance of the cumulative stock return due to not knowing μ\n\n{share_table}",
        "Each path draws its own expected stock return, approximately N(7%, s²), and keeps it for 20 years. "
        "\"Learns μ\" updates a normal prior from each year's stock return and solves the DP on (year, wealth, belief); "
        "\"ignores it\" is the same solver with s = 0 (the base-case policy, 2,000 DP scenarios). Differences are paired. "
        "Deciles: the success probability among the 10% of paths with the lowest and the highest drawn μ.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    xs = 100 * np.array(S_VALUES)
    for name, label, color in ((learn, "Dynamic policy", plots.BLUE), ("Fixed 60/40", "Fixed 60/40", plots.ORANGE)):
        y = np.array([r["rows"][name]["success"] for r in res])
        lo = np.array([r["rows"][name]["success_lowest_mu_decile"] for r in res])
        hi = np.array([r["rows"][name]["success_highest_mu_decile"] for r in res])
        if name == learn:  # one filled band only: two overlapping fills mix into a third colour
            ax.fill_between(xs, lo, hi, color=color, alpha=0.10, lw=0)
        ax.plot(xs, y, color=color, marker="o", ms=6, mec=plots.SURFACE, mew=1.2, label=f"{label}, all paths")
        ax.plot(xs, lo, color=color, lw=1, ls=(0, (2, 2)))
        ax.plot(xs, hi, color=color, lw=1, ls=(0, (2, 2)))
        ax.text(xs[-1] + 0.08, y[-1], pct(y[-1], 0), va="center", fontsize=9, color=plots.INK)
    ax.text(xs[-1] + 0.08, res[-1]["rows"][learn]["success_highest_mu_decile"], "best 10% of μ draws",
            va="center", fontsize=8.5, color=plots.INK_2)
    ax.text(xs[-1] + 0.08, res[-1]["rows"][learn]["success_lowest_mu_decile"], "worst 10%",
            va="center", fontsize=8.5, color=plots.INK_2)
    ax.set_xticks(xs, [f"{x:.0f}%" for x in xs])
    ax.set_xlim(-0.15, 3.9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.set_xlabel("Uncertainty about the expected stock return (s.d. s)")
    ax.set_ylabel("Chance of reaching the goal")
    ax.set_title("The average barely moves; the range explodes")
    ax.plot([], [], color=plots.INK_2, lw=1, ls=(0, (2, 2)), label="worst / best 10% of μ draws")
    ax.legend(loc="lower left")

    ax = axes[1]
    t = np.arange(1, 41)
    for s, color in zip(S_VALUES[1:], (plots.SEQ_BLUE[3], plots.SEQ_BLUE[5], plots.SEQ_BLUE[7])):
        share = param_uncertainty.variance_share(s, B.sigma_stock, tuple(int(x) for x in t), B.mu_stock)
        y = np.array([share[int(x)] for x in t])
        ax.plot(t, y, color=color)
        ax.text(t[-1] + 0.5, y[-1], f"s = {100 * s:.0f}%", va="center", fontsize=9, color=plots.INK_2)
    ax.axvline(INVESTOR.horizon, color=plots.AXIS, lw=1)
    ax.text(INVESTOR.horizon + 0.4, 0.02, "this plan", fontsize=8.5, color=plots.INK_2)
    ax.set_xlim(0, 44)
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Share of variance")
    ax.set_title("Not knowing μ: small for one year, large over decades")
    fig.tight_layout()
    plots.save(fig, FIGURES / "param_uncertainty.png")
    return {"by_s": [{k: v for k, v in r.items() if k not in ("weights_by_year", "policy_year10")} for r in res],
            "variance_share": {str(s): v for s, v in shares.items()}}


def section_regime(n_paths: int) -> dict[str, Any]:
    """3.2: two market regimes in the DP state; regime-blind policy; wrong persistence."""
    model = regime.calibrated(B)
    res = regime.experiment(INVESTOR, model, n_paths=n_paths)
    starts = ["good", "bad", "long-run mix"]
    names = ["Regime-aware DP", "Regime-blind DP (base case)", "Fixed 100% stock", "Fixed 60/40"]

    def cell(start: str, name: str) -> str:
        r = res["by_start"][start][name]
        out = pct(r["success"])
        if name != "Regime-aware DP":
            out += f" ({pp(r['vs_aware'])})"
        return out

    table = md_table(["Policy"] + [f"Starts in {s} state" if s != "long-run mix" else "Start drawn from long-run mix"
                                   for s in starts],
                     [[name] + [cell(s, name) for s in starts] for name in names])
    wrong = [k for k in res["by_start"]["long-run mix"] if k.startswith("Regime-aware, planned")]
    mix = res["by_start"]["long-run mix"]
    wrong_table = md_table(
        ["Planned with stay probability", "Success (true stay = 0.9)", "Cost vs the right probability", "Avg. stock weight"],
        [["0.9 (right)", pct(mix["Regime-aware DP"]["success"]), "—", pct(mix["Regime-aware DP"]["avg_stock_weight"], 0)]]
        + [[k.split("= ")[1], pct(mix[k]["success"]), f"{100 * mix[k]['vs_aware']:+.2f} ± {100 * mix[k]['vs_aware_se']:.2f} pp",
            pct(mix[k]["avg_stock_weight"], 0)] for k in sorted(wrong, key=lambda k: float(k.split("= ")[1]))],
    )
    g, b = res["good"], res["bad"]
    write(
        "regime",
        f"Two market regimes — {n_paths:,} forward paths",
        f"{table}\n\n### Planning with the wrong persistence\n\n{wrong_table}",
        f"Good state: stocks {pct(g['mu_stock'])} expected, {pct(g['sigma_stock'])} volatility; bad state: "
        f"{pct(b['mu_stock'])} and {pct(b['sigma_stock'])}; bonds 3% and 5% in both. Each state persists with "
        f"probability {res['stay']:g} a year, and the 50/50 long-run mix has the base-case 7% and 16%. "
        "The investor sees the current state. The regime-blind policy is the base-case DP. Brackets: paired "
        "difference against the regime-aware policy. The DP's own estimate for the regime-aware policy is "
        f"{pct(res['by_start']['good']['Regime-aware DP']['dp_value'])} starting good and "
        f"{pct(res['by_start']['bad']['Regime-aware DP']['dp_value'])} starting bad.",
    )

    pol = res["policy"]
    grid = np.array(pol["grid"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 0.28]})
    mesh = None
    panels = [("good", f"Good state (stocks {pct(g['mu_stock'], 0)}, vol {pct(g['sigma_stock'], 0)})"),
              ("bad", f"Bad state (stocks {pct(b['mu_stock'], 0)}, vol {pct(b['sigma_stock'], 0)})"),
              ("blind", "Regime-blind (base case 7%, 16%)")]
    for ax, (key, title) in zip(axes[:3], panels):
        mesh = plots.policy_heatmap(ax, grid, np.array(pol[key]), title, goal=INVESTOR.goal, w_range=(2e4, 2e6))
        ax.yaxis.set_major_formatter(plots.dollars)
        ax.set_yticks([2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6])
        ax.minorticks_off()
        ax.set_xticks(range(0, INVESTOR.horizon + 1, 5))
    for ax in axes[1:3]:
        ax.set_yticklabels([])
    axes[0].set_ylabel("Wealth")
    cax = axes[3]
    cax.axis("off")
    bar_ax = cax.inset_axes([0.15, 0.1, 0.18, 0.8])
    cb = fig.colorbar(mesh, cax=bar_ax, ticks=np.linspace(0, 1, 6))
    cb.ax.set_yticklabels([f"{int(100 * x)}%" for x in np.linspace(0, 1, 6)])
    cb.outline.set_visible(False)
    cax.set_title("Stock weight", fontsize=10, loc="left")
    fig.suptitle("Optimal stock weight when the investor knows the market state", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plots.save(fig, FIGURES / "regime_policy.png")
    res.pop("policy")
    return res


def section_hidden(n_paths: int) -> dict[str, Any]:
    """3.2b: the market state has to be inferred from stock returns."""
    model = regime.calibrated(B)
    res = regime.hidden_experiment(INVESTOR, model, n_paths)
    seen, inferred, ignored = "Sees the state", "Infers the state from returns", "Ignores the state (base case)"
    starts = ["good", "bad", "long-run mix"]
    names = [seen, inferred, ignored, "Fixed 100% stock", "Fixed 60/40"]

    def cell(start: str, name: str) -> str:
        r = res["by_start"][start][name]
        return pct(r["success"]) + ("" if name == ignored else f" ({pp(r['vs_ignore'])})")

    headers = ["Policy", "First state good (known)", "First state bad (known)", "First state unknown (long-run mix)"]
    table = md_table(headers, [[name] + [cell(s, name) for s in starts] for name in names])
    mix = res["by_start"]["long-run mix"]
    tr = res["tracking"]
    write(
        "regime_hidden",
        f"Regimes that have to be inferred — {n_paths:,} forward paths",
        table,
        "Same two states as results/regime.md (stay probability 0.9). \"Infers the state\" updates P(bad state) "
        "from each year's stock return by Bayes' rule and solves the DP on (year, wealth, P(bad)) with "
        f"{res['n_beliefs']} belief points; \"sees the state\" is told the state every year; \"ignores the state\" "
        "is the base-case policy. Brackets: paired difference against ignoring the state (s.e. "
        f"{100 * mix[inferred]['vs_ignore_se']:.2f} pp for inferring, {100 * mix[seen]['vs_ignore_se']:.2f} pp for "
        f"seeing). In the first two columns the investor knows the first state; later states are hidden. "
        f"The filter calls the state right in {pct(tr['share_of_years_called_right'], 0)} of years; its average "
        f"P(bad) is {tr['mean_belief_when_bad']:.2f} in bad years and {tr['mean_belief_when_good']:.2f} in good ones. "
        f"DP's own estimate, first state unknown: inferring {pct(mix[inferred]['dp_value'])} (actual "
        f"{pct(mix[inferred]['success'])}); ignoring {pct(mix[ignored]['dp_value'])} (actual {pct(mix[ignored]['success'])}).",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    years = np.arange(INVESTOR.horizon)
    reg = np.array(tr["example_regimes"])
    for t in years[reg == regime.BAD]:
        ax.axvspan(t - 0.5, t + 0.5, color=plots.GRID, lw=0, zorder=0)
    ax.plot(years, tr["example_beliefs"], color=plots.BLUE, marker="o", ms=5, mec=plots.SURFACE, mew=1, zorder=3)
    ax.axhline(0.5, color=plots.AXIS, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(-0.5, INVESTOR.horizon - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, INVESTOR.horizon, 5))
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.set_xlabel("Year")
    ax.set_ylabel("Investor's P(bad state)")
    ax.set_title("One path: shaded years are bad, the line is what the investor believes")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=plots.GRID), plt.Line2D([], [], color=plots.BLUE, marker="o")],
              labels=["Bad state (not seen by the investor)", "P(bad) inferred from stock returns"], loc="upper left")

    ax = axes[1]
    labels = {"good": "First state good (known)", "bad": "First state bad (known)", "long-run mix": "First state unknown"}
    for j, (name, label, color, marker) in enumerate(((seen, "Sees the state", plots.AQUA, "o"),
                                                      (inferred, "Infers it from returns", plots.BLUE, "s"))):
        y = np.arange(len(starts))[::-1] + (0.5 - j) * 0.3
        x = np.array([100 * res["by_start"][s][name]["vs_ignore"] for s in starts])
        err = np.array([196 * res["by_start"][s][name]["vs_ignore_se"] for s in starts])
        ax.errorbar(x, y, xerr=err, fmt=marker, color=color, ms=8, mec=plots.SURFACE, mew=1, elinewidth=1.2,
                    capsize=0, label=label)
        for xi, ei, yi in zip(x, err, y):
            ax.text(xi + ei + 0.06, yi, f"+{xi:.1f}", va="center", fontsize=9, color=plots.INK)
    ax.axvline(0, color=plots.INK, lw=1)
    ax.set_yticks(range(len(starts)), [labels[s] for s in starts][::-1])
    ax.set_xlim(-0.2, 3.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Gain in success probability over ignoring the state (pp, ±95%)")
    ax.set_title("Seeing the state is worth 2.6 pp; inferring it, 0.4")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plots.save(fig, FIGURES / "regime_hidden.png")
    res.pop("policy_year5")
    return res


def section_review(n_paths: int) -> dict[str, Any]:
    """Section 7: review the saving every year, not just the allocation."""
    res = saving.experiment(INVESTOR, n_paths)

    def usd(x: float) -> str:  # nearest $100: the differences here are a few hundred dollars
        return f"${round(x, -2):,.0f}"
    rows = res["base"]
    names = [("Fixed 60/40", "Fixed 60/40, fixed saving", "Fixed 60/40, saving reviewed every year"),
             ("Dynamic allocation", "Dynamic allocation, fixed saving", "Dynamic allocation, saving reviewed every year")]
    table = md_table(
        ["Allocation", "Fixed saving", "Saving reviewed every year: average", "10th–90th pct across paths",
         "Same in present value (3%)", "Success (fixed / reviewed)", "Years at the $40K cap", "Avg. stock (fixed / reviewed)"],
        [[label, usd(rows[f]["avg_saving"]), usd(rows[r]["avg_saving"]),
          f"{usd(rows[r]['saving_p10'])}–{usd(rows[r]['saving_p90'])}",
          f"{usd(rows[f]['pv_level_saving'])} / {usd(rows[r]['pv_level_saving'])}",
          f"{pct(rows[f]['success'])} / {pct(rows[r]['success'])}", pct(rows[r]["share_years_at_cap"]),
          f"{pct(rows[f]['avg_stock_weight'], 0)} / {pct(rows[r]['avg_stock_weight'], 0)}"]
         for label, f, r in names],
    )
    sh = res["shock"]["rows"]
    shock_table = md_table(
        ["Saving plan", "Success", "± s.e.", "Avg. saving, years 0–9", "Avg. saving, years 10–19", "90th pct of avg. saving"],
        [[name, pct(r["success"]), pct(r["success_se"], 2), usd(r["avg_saving_before"]), usd(r["avg_saving_after"]),
          usd(r["saving_p90"])] for name, r in sh.items()],
    )
    write(
        "saving_review",
        f"Saving reviewed every year — {n_paths:,} forward paths",
        f"### Average yearly saving needed for a {pct(res['target'], 0)} chance of reaching the goal (base case)\n\n"
        f"{table}\n\n### {res['shock']['label']}, noticed at once\n\n{shock_table}",
        f"Review rule: each year save the smallest amount (between $0 and ${res['cap']:,.0f}) whose reported success "
        f"reaches a target, with the allocation re-optimised for that amount. The target is set so that the realised "
        f"success is {pct(res['target'], 0)}: {pct(res['review_target_dp'])} for the dynamic allocation, "
        f"{pct(res['review_target_60_40'])} for 60/40 (lower than {pct(res['target'], 0)} because the rule steps in "
        "again in later years). Fixed saving amounts are the bisection results of the saving curve. Present value: "
        "the level yearly amount with the same value discounted at 3%. After the change, all plans re-solve the "
        "allocation; the review rule also re-solves the saving.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for i, ((label, f, r), color) in enumerate(zip(names[::-1], (plots.BLUE, plots.ORANGE))):
        x_f, x_r = rows[f]["avg_saving"], rows[r]["avg_saving"]
        ax.plot([x_r, x_f], [i, i], color=plots.AXIS, lw=2.5, zorder=1)
        ax.plot(x_f, i, "o", ms=10, mfc=plots.SURFACE, mec=color, mew=2, zorder=2)
        ax.plot(x_r, i, "o", ms=10, color=color, mec=plots.SURFACE, mew=1.5, zorder=3)
        ax.text(x_f, i + 0.2, f"fixed {usd(x_f)}", ha="center", fontsize=9, color=plots.INK_2)
        ax.text(x_r, i - 0.32, f"reviewed {usd(x_r)}", ha="center", fontsize=9, color=plots.INK)
    ax.set_yticks([0, 1], [names[1][0], names[0][0]])
    ax.set_ylim(-0.7, 1.6)
    ax.set_xlim(9_000, 15_500)
    ax.xaxis.set_major_formatter(plots.dollars)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Average yearly saving for a {pct(res['target'], 0)} chance of reaching $500K")
    ax.set_title("Two levers of about the same size")

    ax = axes[1]
    years = np.arange(INVESTOR.horizon)
    r = rows["Dynamic allocation, saving reviewed every year"]
    ax.fill_between(years, r["saving_by_year_p10"], r["saving_by_year_p90"], color=plots.BLUE, alpha=0.12, lw=0)
    ax.plot(years, r["saving_by_year_median"], color=plots.BLUE, label="Reviewed, base case (median, 10th–90th pct)")
    s = sh["Reviewed every year"]
    ax.plot(years, s["saving_by_year_median"], color=plots.AQUA, label=f"Reviewed, {res['shock']['label'].lower()} (median)")
    ax.axhline(res["fixed_saving_dp"], color=plots.GREY, lw=1.2, ls=(0, (4, 3)),
               label=f"Fixed plan for {pct(res['target'], 0)}: {usd(res['fixed_saving_dp'])}")
    ax.axvline(res["shock"]["year"] - 0.5, color=plots.AXIS, lw=1)
    ax.text(res["shock"]["year"] - 0.35, 30_500, "stock return drops\n(second line only)", fontsize=8.5,
            color=plots.INK_2, va="bottom")
    ax.axhline(res["cap"], color=plots.AXIS, lw=1)
    ax.text(0.0, res["cap"] + 500, f"cap {usd(res['cap'])}", fontsize=8.5, color=plots.INK_2, va="bottom")
    ax.set_xticks(range(0, INVESTOR.horizon, 5))
    ax.set_xlim(-0.3, INVESTOR.horizon - 0.7)
    ax.set_ylim(0, res["cap"] + 3_000)
    ax.yaxis.set_major_formatter(plots.dollars)
    ax.set_xlabel("Year")
    ax.set_ylabel("Saving that year")
    ax.set_title("The review saves more when behind, less (or nothing) when ahead")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout()
    plots.save(fig, FIGURES / "saving_review.png")
    return res


def section_shortfall(n_paths: int) -> dict[str, Any]:
    """Section 8: penalise the size of a miss; compare with a cap, power utility and fixed mixes."""
    res = shortfall.experiment(INVESTOR, n_paths)
    fam = res["families"]

    def k(x: float) -> str:
        return f"${x / 1000:,.0f}K"

    def row(label: str, r: dict[str, Any]) -> list[str]:
        return [label, pct(r["success"]), k(r["p05"]), k(r["p10"]), k(r["median"]), k(r["mean_wealth_if_missed"]),
                k(r["expected_shortfall"]), k(r["rms_shortfall"]), pct(r["avg_stock_weight"], 0),
                pct(r["avg_stock_last5_if_missed"], 0)]

    headers = ["Strategy", "Success", "5th pct", "10th pct", "Median", "Avg. wealth if missed", "Expected shortfall",
               "RMS shortfall", "Avg. stock", "Stock in last 5 years, paths that miss"]
    rows = [row(f"Linear penalty, η = {r['eta']:g}", r) for r in fam["Linear shortfall penalty"]]
    rows += [row(f"Squared penalty, η = {r['eta']:g}", r) for r in fam["Squared shortfall penalty"][1:]]
    rows += [row(f"Goal, stock cap {r['cap']:.0%}", r) for r in fam["Goal with a stock cap"][1:]]
    rows += [row(f"Power utility, γ = {r['gamma']:g}", r) for r in fam["Power utility"]]
    rows += [row(f"Fixed {r['weight']:.0%} stock", r) for r in fam["Fixed mix"] if r["weight"] >= 0.3]
    levels = (305_000.0, 310_000.0, 315_000.0, 320_000.0)
    names = ["Squared shortfall penalty", "Linear shortfall penalty", "Goal with a stock cap", "Power utility", "Fixed mix"]
    matched = {name: [shortfall.success_at(fam[name], lv) for lv in levels] for name in names}
    matched_table = md_table(["Strategy family"] + [f"10th pct ≥ {k(lv)}" for lv in levels],
                             [[name] + [pct(v) if v == v else "not reached" for v in vals] for name, vals in matched.items()])
    write(
        "shortfall",
        f"Penalising the size of a miss — {n_paths:,} forward paths",
        f"### Chance of reaching the goal when the 10th-percentile outcome must be at least …\n\n{matched_table}\n\n"
        f"### Every strategy\n\n{md_table(headers, rows)}",
        "Objective: maximise P(W ≥ goal) − η · E[((goal − W)⁺ / goal)^p], p = 1 (linear) or 2 (squared); η = 0 is "
        "the goal objective. Expected shortfall = E[(goal − W)⁺] over all paths; RMS shortfall = its root mean square. "
        "Matched table: success read off each family's efficient side, by linear interpolation between its points.",
    )

    # ---- figure: the trade-off -------------------------------------------------- #
    styles = {
        "Squared shortfall penalty": (plots.BLUE, "o", "-", "eta", "η = {:g}"),
        "Linear shortfall penalty": (plots.AQUA, "s", "-", "eta", "η = {:g}"),
        "Goal with a stock cap": (plots.YELLOW, "^", "-", "cap", "cap {:.0%}"),
        "Power utility": (plots.MAGENTA, "D", "-", "gamma", "γ = {:g}"),
        "Fixed mix": (plots.GREY, ".", (0, (3, 2)), "weight", "{:.0%}"),
    }
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for name, (color, marker, ls, key, fmt) in styles.items():
        # In parameter order (eta up, cap down, gamma up, weight up), so each line traces its family.
        pts = [r for r in fam[name] if name != "Fixed mix" or r["weight"] >= 0.3]
        x = np.array([r["p10"] for r in pts]) / 1000
        y = np.array([r["success"] for r in pts])
        ax.plot(x, y, color=color, ls=ls, marker=marker, ms=6 if marker != "." else 8, mec=plots.SURFACE, mew=1,
                label=name, zorder=3 if name.startswith("Squared") else 2)
    for name, pick in (("Squared shortfall penalty", (16.0, 32.0, 64.0)), ("Goal with a stock cap", (0.8, 0.6)),
                       ("Power utility", (3.0, 5.0, 10.0)), ("Linear shortfall penalty", (32.0,)),
                       ("Fixed mix", (0.3, 0.6, 1.0))):
        color, _, _, key, fmt = styles[name]
        for r in fam[name]:
            if r[key] in pick:
                ax.annotate(fmt.format(r[key]), (r["p10"] / 1000, r["success"]), xytext=(6, 4),
                            textcoords="offset points", fontsize=8.5, color=plots.INK_2)
    goal_row = fam["Squared shortfall penalty"][0]
    ax.annotate("goal objective\n(η = 0, no cap)", (goal_row["p10"] / 1000, goal_row["success"]), xytext=(-10, 10),
                textcoords="offset points", ha="right", fontsize=8.5, color=plots.INK)
    ax.set_xlabel("10th percentile of final wealth ($K) — how bad a bad outcome is")
    ax.set_ylabel("Chance of reaching $500K")
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(round(100 * v))}%")
    ax.set_ylim(0.1, 0.68)
    ax.set_title("Buying a better bad case: the squared shortfall penalty gives up the least success")
    ax.legend(loc="lower left")
    fig.tight_layout()
    plots.save(fig, FIGURES / "shortfall_frontier.png")

    # ---- figure: what the policies do --------------------------------------------- #
    shown = {"goal": "Goal objective", "shortfall:1:8": "Linear penalty, η = 8", "shortfall:2:32": "Squared penalty, η = 32",
             "power:5": "Power utility, γ = 5"}
    sols = an.run_parallel(an.solve_policy, [(INVESTOR, BASE, obj) for obj in shown])
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.22]})
    mesh = None
    for ax, sol, title in zip(axes[:4], sols, shown.values()):
        mesh = plots.policy_heatmap(ax, sol.grid, sol.policy, title, goal=INVESTOR.goal, w_range=(5e4, 2e6))
        ax.yaxis.set_major_formatter(plots.dollars)
        ax.set_yticks([5e4, 1e5, 2e5, 5e5, 1e6, 2e6])
        ax.minorticks_off()
        ax.set_xticks(range(0, INVESTOR.horizon + 1, 5))
    for ax in axes[1:4]:
        ax.set_yticklabels([])
    axes[0].set_ylabel("Wealth")
    axes[4].axis("off")
    cax = axes[4].inset_axes([0.1, 0.1, 0.25, 0.8])
    cb = fig.colorbar(mesh, cax=cax, ticks=np.linspace(0, 1, 6))
    cb.ax.set_yticklabels([f"{int(100 * x)}%" for x in np.linspace(0, 1, 6)])
    cb.outline.set_visible(False)
    axes[4].set_title("Stock", fontsize=10, loc="left")
    fig.suptitle("Optimal stock weight: near the date, the squared penalty cuts risk for an investor who is short of the goal",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plots.save(fig, FIGURES / "shortfall_policies.png")
    res["matched"] = {"levels": list(levels), "success": matched}
    return res


# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    """Convert numpy types and drop large arrays for the JSON dump."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k != "success_flags"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def main() -> None:
    """Run the selected sections and write tables, figures and ``results/results.json``."""
    sections = ["compare", "assumptions", "sensitivity", "dynamic", "cap", "precision", "saving",
                "resolve", "uncertainty", "regime", "hidden", "review", "shortfall"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sections, default=sections)
    parser.add_argument("--quick", action="store_true", help="20,000 forward paths instead of 100,000")
    args = parser.parse_args()
    n_paths = 20_000 if args.quick else an.SIM_PATHS
    plots.apply_style()
    RESULTS.mkdir(exist_ok=True)
    out_file = RESULTS / "results.json"
    results: dict[str, Any] = json.loads(out_file.read_text()) if out_file.exists() else {}
    flags = None
    for name in sections:
        if name not in args.only:
            continue
        start = time.time()
        if name == "compare":
            res = section_compare(n_paths)
            flags = res["success_flags"]
        elif name == "assumptions":
            res = section_assumptions(n_paths)
        elif name == "sensitivity":
            res = section_sensitivity(n_paths)
        elif name == "dynamic":
            res = section_dynamic(n_paths)
        elif name == "cap":
            res = section_cap(n_paths)
        elif name == "precision":
            res = section_precision(n_paths, flags)
        elif name == "saving":
            res = section_saving(n_paths)
        elif name == "resolve":
            res = section_resolve(n_paths)
        elif name == "uncertainty":
            res = section_uncertainty(n_paths)
        elif name == "regime":
            res = section_regime(n_paths)
        elif name == "hidden":
            res = section_hidden(n_paths)
        elif name == "review":
            res = section_review(n_paths)
        else:
            res = section_shortfall(n_paths)
        results[name] = _jsonable(res)
        out_file.write_text(json.dumps(results, indent=1))
        print(f"[{name}] done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
