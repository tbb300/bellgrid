"""Tests for ``GoldenSearch`` action grid spec.

Three things to verify:

1. **Equivalence.** Golden search on a smooth concave problem (log-utility
   Merton) hits the same precision as a fine ``RegularGrid`` at a small
   fraction of the Bellman evals.
2. **Multi-continuous coordinate descent.** Two correlated continuous
   actions (consume + risky-asset share) converge to a reference solved
   on a fine joint grid.
3. **Mixed continuous + discrete.** A continuous consume action paired
   with a discrete on/off decision recovers the same policy as the
   all-grid solve.

Plus a couple of construction-validation checks.
"""

import math

import pytest
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    Problem,
    solve,
)
from bellgrid.grids import GoldenSearch, RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_golden_search_construction_defaults():
    spec = GoldenSearch()
    assert spec.n_init == 4
    assert spec.n_iter == 20
    assert spec.n_coord == 2


def test_golden_search_n_init_too_small_raises():
    with pytest.raises(ValueError, match="n_init >= 2"):
        GoldenSearch(n_init=1)


def test_golden_search_n_iter_too_small_raises():
    with pytest.raises(ValueError, match="n_iter >= 1"):
        GoldenSearch(n_iter=0)


def test_golden_search_n_coord_too_small_raises():
    with pytest.raises(ValueError, match="n_coord >= 1"):
        GoldenSearch(n_coord=0)


# ---------------------------------------------------------------------------
# Helper to build a log-utility Merton problem for the equivalence tests.
# ---------------------------------------------------------------------------


def _merton_log_utility(beta=0.96, mu=0.04, sigma=0.15, T=10):
    B = 1.0 / (1.0 - beta)
    A = (
        math.log(1.0 - beta) / (1.0 - beta)
        + (beta / (1.0 - beta) ** 2) * (math.log(beta) + mu)
    )

    def transition(state, action, shock, _t):
        return {
            "wealth": (state["wealth"] - action["consume"])
            * torch.exp(mu + sigma * shock["z"])
        }

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    def terminal_reward(state):
        return A + B * torch.log(state["wealth"])

    return Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 50.0))],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, T),
        discount=beta,
        terminal_reward=terminal_reward,
    )


# ---------------------------------------------------------------------------
# 1. Equivalence on a single continuous action
# ---------------------------------------------------------------------------


def test_golden_matches_fine_grid_on_merton():
    """Golden search with n_init=4, n_iter=20 should saturate the same
    precision floor as a 500-point regular grid on log-utility Merton.

    Both solvers share the same wealth grid; on this problem the residual
    error is dominated by the wealth-axis V interpolation, not the
    consume axis (going from grid n=500 → n=5000 only shifts the rate
    error from 8.6e-5 → 2.7e-5 by lucky alignment, and golden lands
    right at the n=500 floor)."""
    problem = _merton_log_utility()
    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)

    policy_grid, value_grid = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
        device="cpu",
    )
    policy_gold, value_gold = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": GoldenSearch(n_init=4, n_iter=20)},
        solver=BackwardInduction(n_quad=7),
        device="cpu",
    )

    c_grid = policy_grid({"wealth": test_w}, t=5)["consume"]
    c_gold = policy_gold({"wealth": test_w}, t=5)["consume"]
    v_grid = value_grid({"wealth": test_w}, t=5)
    v_gold = value_gold({"wealth": test_w}, t=5)

    # Both should agree to ~1 part in 5e-3 (action-axis precision is
    # below the wealth-grid floor for both, so they disagree only by the
    # quantisation noise of the grid).
    assert torch.allclose(c_gold, c_grid, rtol=5e-3, atol=5e-3), (
        f"consume mismatch: grid={c_grid.tolist()} golden={c_gold.tolist()}"
    )
    assert torch.allclose(v_gold, v_grid, rtol=1e-4, atol=1e-3), (
        f"V mismatch: grid={v_grid.tolist()} golden={v_gold.tolist()}"
    )


def test_golden_recovers_closed_form_consumption_rate():
    """End-to-end: golden should recover the closed-form c/w = 1 - β
    rate as tightly as the existing grid-based test does."""
    beta = 0.96
    problem = _merton_log_utility(beta=beta)
    policy, _ = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": GoldenSearch(n_init=4, n_iter=20)},
        solver=BackwardInduction(n_quad=7),
        device="cpu",
    )
    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)
    actions = policy({"wealth": test_w}, t=5)
    rates = actions["consume"] / test_w
    expected = torch.full_like(rates, 1.0 - beta)
    assert torch.allclose(rates, expected, rtol=0.01, atol=0.005), (
        f"rates={rates.tolist()} expected≈{1.0-beta}"
    )


# ---------------------------------------------------------------------------
# 2. Multi-continuous coordinate descent
# ---------------------------------------------------------------------------


def test_golden_multi_continuous_matches_grid():
    """Two-asset Merton-style problem: jointly choose ``consume`` and
    ``share`` (the risky-asset weight). The optimal policy is concave in
    both. Golden coordinate descent should match a fine joint grid.

    Setup uses deterministic returns + a single Normal shock to keep the
    reference cheap; the structure exercises the multi-axis refinement."""
    beta, mu_r, sigma_r, rf = 0.96, 0.05, 0.20, 0.02

    def transition(state, action, shock, _t):
        gross_risky = math.exp(mu_r - 0.5 * sigma_r ** 2) * torch.exp(sigma_r * shock["z"])
        gross_rf = math.exp(rf)
        portfolio = action["share"] * gross_risky + (1.0 - action["share"]) * gross_rf
        return {"wealth": (state["wealth"] - action["consume"]) * portfolio}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    problem = Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 50.0))],
        actions=[
            ContinuousAction("consume", bounds=(1e-6, "wealth")),
            ContinuousAction("share", bounds=(0.0, 1.0)),
        ],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 8),
        discount=beta,
        terminal_reward=lambda s: torch.log(s["wealth"]),
    )

    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)

    policy_grid, value_grid = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=96)},
        action_grid={
            "consume": RegularGrid(n=200),
            "share": RegularGrid(n=80),
        },
        solver=BackwardInduction(n_quad=7),
        device="cpu",
    )
    policy_gold, value_gold = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=96)},
        action_grid={
            "consume": GoldenSearch(n_init=4, n_iter=20, n_coord=3),
            "share": GoldenSearch(n_init=4, n_iter=20, n_coord=3),
        },
        solver=BackwardInduction(n_quad=7),
        device="cpu",
    )

    a_grid = policy_grid({"wealth": test_w}, t=4)
    a_gold = policy_gold({"wealth": test_w}, t=4)
    v_grid = value_grid({"wealth": test_w}, t=4)
    v_gold = value_gold({"wealth": test_w}, t=4)

    # Value should agree to ~grid resolution (consume grid step ~ wealth/200).
    assert torch.allclose(v_gold, v_grid, rtol=1e-3, atol=1e-3), (
        f"V grid={v_grid.tolist()} gold={v_gold.tolist()}"
    )
    # Consume policy should agree to the same tolerance.
    assert torch.allclose(
        a_gold["consume"], a_grid["consume"], rtol=5e-3, atol=1e-3,
    ), (
        f"consume grid={a_grid['consume'].tolist()} "
        f"gold={a_gold['consume'].tolist()}"
    )
    # Share is on [0, 1] — looser tol because the optimal share is near
    # the upper boundary and grid quantises it.
    assert torch.allclose(
        a_gold["share"], a_grid["share"], rtol=5e-2, atol=2e-2,
    ), (
        f"share grid={a_grid['share'].tolist()} "
        f"gold={a_gold['share'].tolist()}"
    )


# ---------------------------------------------------------------------------
# 3. Mixed continuous (golden) + discrete (enumerated)
# ---------------------------------------------------------------------------


def test_golden_with_discrete_action_matches_grid():
    """Continuous ``consume`` (golden) + discrete ``invest`` (enumerated).
    The discrete choice is between a safe and a risky asset; the optimum
    is to always pick the risky one when its expected return dominates.
    Either way the joint policy from golden + discrete should match the
    grid-only reference."""
    beta, mu_r, sigma_r, rf = 0.96, 0.06, 0.15, 0.02

    def transition(state, action, shock, _t):
        use_risky = action["invest"].to(torch.float64)
        risky = math.exp(mu_r - 0.5 * sigma_r ** 2) * torch.exp(sigma_r * shock["z"])
        safe = math.exp(rf) * torch.ones_like(risky)
        gross = use_risky * risky + (1.0 - use_risky) * safe
        return {"wealth": (state["wealth"] - action["consume"]) * gross}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    problem = Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 30.0))],
        actions=[
            ContinuousAction("consume", bounds=(1e-6, "wealth")),
            DiscreteAction("invest", n=2),
        ],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 8),
        discount=beta,
        terminal_reward=lambda s: torch.log(s["wealth"]),
    )

    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)

    policy_grid, value_grid = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=96)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
        device="cpu",
    )
    policy_gold, value_gold = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=96)},
        action_grid={"consume": GoldenSearch(n_init=4, n_iter=20)},
        solver=BackwardInduction(n_quad=7),
        device="cpu",
    )

    a_grid = policy_grid({"wealth": test_w}, t=4)
    a_gold = policy_gold({"wealth": test_w}, t=4)
    v_grid = value_grid({"wealth": test_w}, t=4)
    v_gold = value_gold({"wealth": test_w}, t=4)

    assert torch.allclose(v_gold, v_grid, rtol=1e-3, atol=1e-3), (
        f"V grid={v_grid.tolist()} gold={v_gold.tolist()}"
    )
    assert torch.allclose(
        a_gold["consume"], a_grid["consume"], rtol=5e-3, atol=1e-3,
    ), (
        f"consume grid={a_grid['consume'].tolist()} "
        f"gold={a_gold['consume'].tolist()}"
    )
    # Discrete choice should agree exactly (it's a long index).
    assert torch.equal(a_gold["invest"], a_grid["invest"]), (
        f"invest grid={a_grid['invest'].tolist()} "
        f"gold={a_gold['invest'].tolist()}"
    )


# ---------------------------------------------------------------------------
# Cost / correctness micro-benchmark — golden converges with far fewer
# Bellman evals than a comparable RegularGrid.
# ---------------------------------------------------------------------------


def test_golden_print_speedup_vs_grid():
    """Diagnostic — not asserting wall-clock (CI noise), just printing
    the work / precision trade-off for the user to inspect."""
    import time

    problem = _merton_log_utility()
    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)

    beta = 0.96
    expected_rate = 1.0 - beta

    print()
    print("Merton consume policy: action-axis precision vs work")
    print(f"{'spec':35s} {'rate_err':>12} {'wall (s)':>10}")
    print("-" * 60)
    for label, grid in [
        ("grid n=50", RegularGrid(n=50)),
        ("grid n=500", RegularGrid(n=500)),
        ("grid n=5000", RegularGrid(n=5000)),
        ("golden n_init=4, n_iter=15", GoldenSearch(n_init=4, n_iter=15)),
        ("golden n_init=4, n_iter=20", GoldenSearch(n_init=4, n_iter=20)),
        ("golden n_init=8, n_iter=20", GoldenSearch(n_init=8, n_iter=20)),
    ]:
        t0 = time.perf_counter()
        policy, _ = solve(
            problem,
            state_grid={"wealth": WarpedGrid(n=128)},
            action_grid={"consume": grid},
            solver=BackwardInduction(n_quad=7),
            device="cpu",
        )
        wall = time.perf_counter() - t0
        rates = policy({"wealth": test_w}, t=5)["consume"] / test_w
        rate_err = (rates - expected_rate).abs().max().item()
        print(f"{label:35s} {rate_err:>12.2e} {wall:>10.3f}")
