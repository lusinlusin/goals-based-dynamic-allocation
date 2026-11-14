"""Policy tables, and the three ways of handling assumptions that change.

Each method gets the check named in the plan: re-solving with unchanged
assumptions reproduces the original table; a regime model whose two states are
identical is the base model; parameter uncertainty of zero is the base model.
"""

import pickle
from dataclasses import replace

import numpy as np
import pytest

from gbda import analysis as an
from gbda import dp, market, param_uncertainty as pu, policy, regime, resolve, simulate

INVESTOR = an.Investor()
SHORT = replace(INVESTOR, horizon=6, w0=250_000.0)  # a quick problem for the slower checks
B = market.BASE


@pytest.fixture(scope="module")
def table() -> policy.PolicyTable:
    return policy.solve_policy(INVESTOR, an.World())


def base_dp(investor: an.Investor, m: int) -> dp.DPSolution:
    return dp.solve(an.World().scenarios(m), an.ACTIONS, an.wealth_grid(investor), dp.goal_objective(investor.goal),
                    investor.horizon, investor.contribution, zero_tol=an.GOAL_ZERO_TOL)


# ---- Policy tables: offline solve, run-time lookup, save and load ----------------- #
def test_saved_table_loads_back_identically(table: policy.PolicyTable, tmp_path) -> None:
    path = policy.save(table, tmp_path / "base.npz")
    loaded = policy.load(path)
    for name in ("grid", "actions", "policy", "value"):
        assert np.array_equal(getattr(loaded.solution, name), getattr(table.solution, name))
    assert loaded.inputs == table.inputs
    assert loaded.inputs["assumptions"]["mu_stock"] == B.mu_stock
    paths = an.World().paths(2000, INVESTOR.horizon)
    a = an.run(policy.as_policy(table), INVESTOR, paths)
    b = an.run(policy.as_policy(loaded), INVESTOR, paths)
    assert np.array_equal(a.wealth, b.wealth)


def test_lookup(table: policy.PolicyTable) -> None:
    sol = table.solution
    wealth = np.geomspace(3e4, 1e6, 50)
    assert np.array_equal(policy.lookup(table, 7, wealth), sol.action(7, wealth))
    # Linear lookup agrees on grid points and stays between the neighbouring weights elsewhere.
    assert np.allclose(policy.lookup(table, 7, sol.grid, "linear"), sol.policy[7])
    mid = np.sqrt(sol.grid[:-1] * sol.grid[1:])
    lin = policy.lookup(table, 7, mid, "linear")
    lo = np.minimum(sol.policy[7][:-1], sol.policy[7][1:])
    hi = np.maximum(sol.policy[7][:-1], sol.policy[7][1:])
    assert np.all((lin >= lo - 1e-12) & (lin <= hi + 1e-12))
    with pytest.raises(ValueError):
        policy.lookup(table, INVESTOR.horizon, 1e5)


def test_solved_tables_can_be_sent_between_processes(table: policy.PolicyTable) -> None:
    clone = pickle.loads(pickle.dumps(table))
    assert np.array_equal(clone.solution.objective.terminal(np.array([4e5, 6e5])), [0.0, 1.0])
    for obj in (dp.power_objective(5.0, 5e5), dp.log_objective(5e5)):
        assert pickle.loads(pickle.dumps(obj)).to_ce(np.array([-0.3]))[0] == obj.to_ce(np.array([-0.3]))[0]


# ---- Engine extensions ----------------------------------------------------------- #
def test_one_scenario_set_per_year_equals_one_set_for_all_years() -> None:
    sc = an.World().scenarios(1000)
    args = (an.ACTIONS, an.wealth_grid(INVESTOR), dp.goal_objective(INVESTOR.goal), INVESTOR.horizon,
            INVESTOR.contribution)
    a = dp.solve(sc, *args)
    b = dp.solve([sc] * INVESTOR.horizon, *args)
    assert np.array_equal(a.policy, b.policy) and np.array_equal(a.value, b.value)


def test_extra_state_with_one_state_is_the_base_solver() -> None:
    sc = an.World().scenarios(1000)
    m = sc.r_a.size
    stay = dp.StateStep(sc, np.zeros(m, int), np.zeros(m, int), np.zeros(m))
    one = dp.solve_with_state(lambda t, s: stay, 1, an.ACTIONS, an.wealth_grid(SHORT), SHORT.goal, SHORT.horizon,
                              SHORT.contribution, zero_tol=an.GOAL_ZERO_TOL)
    base = base_dp(SHORT, 1000)
    assert np.array_equal(one.policy[:, 0], base.policy)
    assert np.allclose(one.value[:, 0], base.value, atol=1e-12)


# ---- 3.1 Re-solving ------------------------------------------------------------ #
def test_resolving_with_unchanged_assumptions_gives_back_the_rest_of_the_table(table: policy.PolicyTable) -> None:
    """Optimality principle: the tail of an optimal policy is optimal for the tail problem."""
    for year in (1, 10, 19):
        again = policy.solve_policy(replace(INVESTOR, horizon=INVESTOR.horizon - year), an.World(), start_year=year)
        assert resolve.tail_matches(table, again)


def test_known_change_solved_at_the_outset_matches_a_resolve_at_the_change() -> None:
    """With the change known in advance, the rows after it are the problem solved fresh at that date."""
    path = resolve.changing(B, SHORT.horizon, 3, mu_stock=0.05)
    whole = resolve.hindsight(SHORT, path)
    later = policy.solve_policy(replace(SHORT, horizon=SHORT.horizon - 3), an.World(path[3]), start_year=3)
    assert np.array_equal(whole.policy[3:], later.solution.policy)
    assert np.array_equal(resolve.hindsight(SHORT, [B] * SHORT.horizon).policy,
                          policy.solve_policy(SHORT, an.World()).solution.policy)


def test_resolving_every_year_equals_resolving_only_when_the_belief_changes() -> None:
    beliefs = resolve.changing(B, SHORT.horizon, 3, mu_stock=0.05)
    every = resolve.rolling_tables(SHORT, beliefs, parallel=False)
    some = resolve.rolling_tables(SHORT, beliefs, resolve.change_years(beliefs), parallel=False)
    assert [tb.start_year for tb in some] == [0, 3]
    paths = resolve.paths(beliefs, 5000)
    a = simulate.simulate(policy.as_policy(every), *paths, SHORT.w0, SHORT.contribution)
    b = simulate.simulate(policy.as_policy(some), *paths, SHORT.w0, SHORT.contribution)
    assert np.array_equal(a.weights, b.weights)


def test_changing_return_paths_splice_the_two_worlds() -> None:
    path = resolve.changing(B, 8, 5, mu_stock=0.05)
    r_s, r_b = resolve.paths(path, 3000)
    before = an.World(B).paths(3000, 8)
    after = an.World(path[-1]).paths(3000, 8)
    assert np.array_equal(r_s[:, :5], before[0][:, :5]) and np.array_equal(r_b[:, 5:], after[1][:, 5:])
    assert np.array_equal(resolve.paths([B] * 8, 3000)[0], before[0])


# ---- 3.3 Parameter uncertainty ---------------------------------------------------- #
def test_zero_parameter_uncertainty_is_the_base_model() -> None:
    model = pu.UncertainStock(B, 0.0)
    r_s, r_b, mu = model.paths(2000, 10)
    base = an.World().paths(2000, 10)
    assert np.array_equal(r_s, base[0]) and np.array_equal(r_b, base[1])
    assert np.all(mu == mu[0])
    sol = model.solve(SHORT, m=1000)
    assert np.array_equal(sol.policy[:, 0], base_dp(SHORT, 1000).policy)


def test_parameter_draws_have_the_intended_spread() -> None:
    _, _, mu = pu.UncertainStock(B, 0.02).paths(200_000, 1)
    assert mu.mean() == pytest.approx(B.mu_stock, abs=0.0005)
    assert mu.std() == pytest.approx(0.02, rel=0.03)


def test_belief_is_the_bayesian_posterior_mean() -> None:
    """The year-by-year update equals the closed-form precision-weighted average."""
    model = pu.UncertainStock(B, 0.02)
    r_s, _, _ = model.paths(500, 12)
    beliefs = model.beliefs(r_s)
    m_hat, sigma = model.stock_log
    t = 12 - 1
    x = np.log(r_s[:, :t]).sum(axis=1)
    closed = (m_hat / model.tau**2 + x / sigma**2) / (1 / model.tau**2 + t / sigma**2)
    assert np.allclose(beliefs[:, t], closed)
    assert model.posterior_var(t) == pytest.approx(1 / (1 / model.tau**2 + t / sigma**2))


def test_expected_next_belief_is_todays_belief() -> None:
    """A posterior mean is a martingale; the DP's belief transitions must keep that (interior grid points)."""
    model = pu.UncertainStock(B, 0.02)
    steps = model.stepper(2000, an.DP_SEED)
    grid = model.belief_grid()
    for t in (0, 10):
        for b in (3, 6, 9):
            st = steps(t, b)
            assert np.all((st.w_hi >= 0) & (st.w_hi <= 1))
            nxt = (1 - st.w_hi) * grid[st.lo] + st.w_hi * grid[st.hi]
            assert nxt @ st.scenarios.probs == pytest.approx(grid[b], abs=1e-12)


def test_parameter_uncertainty_matters_more_over_longer_horizons() -> None:
    share = pu.variance_share(0.02, 0.16, (1, 10, 20, 40))
    values = list(share.values())
    assert all(0 < v < 1 for v in values) and values == sorted(values)


# ---- 3.2 Regimes ------------------------------------------------------------------ #
def test_two_identical_regimes_are_the_base_model() -> None:
    model = regime.RegimeModel(B, B, 0.9, 0.9)
    sol = model.solve(SHORT, m=1000)
    base = base_dp(SHORT, 1000)
    for s in (regime.GOOD, regime.BAD):
        assert np.array_equal(sol.policy[:, s], base.policy)
        assert np.allclose(sol.value[:, s], base.value, atol=1e-12)


def test_calibrated_regimes_average_to_the_base_case() -> None:
    model = regime.calibrated(B)
    pi = model.stationary
    mean = pi[0] * model.good.mu_stock + pi[1] * model.bad.mu_stock
    second = pi[0] * (model.good.sigma_stock**2 + model.good.mu_stock**2) + pi[1] * (model.bad.sigma_stock**2 + model.bad.mu_stock**2)
    assert mean == pytest.approx(B.mu_stock)
    assert np.sqrt(second - mean**2) == pytest.approx(B.sigma_stock)


def test_regime_chains_switch_at_the_stated_rate() -> None:
    model = regime.RegimeModel(B, B, 0.9, 0.7)
    assert model.stationary == pytest.approx([0.75, 0.25])
    reg = model.regimes(50_000, 30)
    assert (reg == regime.BAD).mean() == pytest.approx(0.25, abs=0.01)
    stay_good = (reg[:, 1:] == regime.GOOD)[reg[:, :-1] == regime.GOOD].mean()
    assert stay_good == pytest.approx(0.9, abs=0.005)
    paths = model.paths(reg[:2000, :8])
    assert np.array_equal(paths[0], an.World(B).paths(2000, 8)[0])  # identical regimes: base paths


def test_starting_in_the_good_regime_is_better() -> None:
    sol = regime.calibrated(B).solve(SHORT, m=1000)
    assert sol.value_at(0, SHORT.w0, regime.GOOD) > sol.value_at(0, SHORT.w0, regime.BAD)


# ---- 3.2b Regimes that have to be inferred ---------------------------------------- #
def test_belief_update_end_points() -> None:
    model = regime.calibrated(B)
    x = np.log(np.array([0.7, 1.0, 1.3]))
    assert np.allclose(regime.update_belief(model, 0.0, x), 1 - model.stay_good)  # sure it was good
    assert np.allclose(regime.update_belief(model, 1.0, x), model.stay_bad)       # sure it was bad
    up, down = regime.update_belief(model, 0.5, np.log(np.array([1.30, 0.80])))
    assert up < 0.5 < down  # a good year points to the good state, a bad year to the bad one


def test_filtered_belief_is_calibrated() -> None:
    """Among path-years where the filter says P(bad) is about p, the market is bad about p of the time."""
    model = regime.calibrated(B)
    reg = model.regimes(40_000, 20)
    r_s, _ = model.paths(reg)
    beliefs = regime.filter_beliefs(model, r_s, 0.5)[:, 1:].ravel()
    bad = (reg[:, 1:] == regime.BAD).ravel()
    edges = np.quantile(beliefs, np.linspace(0, 1, 6))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (beliefs >= lo) & (beliefs < hi)
        assert bad[sel].mean() == pytest.approx(beliefs[sel].mean(), abs=0.015)


def test_expected_next_belief_follows_the_chain() -> None:
    """Averaged over next year's returns, the updated belief is the prior moved one step along the chain."""
    model = regime.calibrated(B)
    steps = regime.HiddenSteps(model, an.World(model.good).scenarios(4000), an.World(model.bad).scenarios(4000),
                               regime.belief_grid(11))
    for b in (2, 5, 8):
        p = steps.grid[b]
        st = steps(0, b)
        nxt = (1 - st.w_hi) * steps.grid[st.lo] + st.w_hi * steps.grid[st.hi]
        assert nxt @ st.scenarios.probs == pytest.approx(p * model.stay_bad + (1 - p) * (1 - model.stay_good), abs=0.01)


def test_hidden_state_with_identical_regimes_is_the_base_model() -> None:
    same = regime.RegimeModel(B, B, 0.9, 0.9)
    sol = regime.solve_hidden(same, SHORT, 5, 1000)
    base = base_dp(SHORT, 1000)
    for b in range(5):
        assert np.array_equal(sol.policy[:, b], base.policy)
        assert np.allclose(sol.value[:, b], base.value, atol=1e-12)
