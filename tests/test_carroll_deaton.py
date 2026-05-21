"""Carroll/Deaton lifecycle consumption-savings.

A finite-lived household with deterministic risk-free returns and a
stochastic labor income process, subject to a borrowing constraint
(``consume ≤ cash-on-hand``). The qualitative signature of this model is
the *kinked* consumption function: at low cash the constraint binds and
the household consumes ~all of its cash (MPC ≈ 1); above some threshold
it saves a fraction of marginal cash (MPC < 1).

There is no closed form to test against, so we validate the qualitative
shape and a printed table that lets us eyeball the kink.
"""

import math

import pytest
import torch

from bellgrid import ContinuousAction, ContinuousState, Problem, simulate, solve
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction


# --- problem builder -----------------------------------------------------


def _build_problem(
    *,
    gamma: float = 2.0,   # CRRA risk aversion (gamma=1 is log utility)
    R: float = 1.04,      # gross risk-free return
    beta: float = 0.94,   # discount factor (R*beta < 1 → buffer-stock saver)
    mu_y: float = 1.0,    # mean labor income
    sigma_y: float = 0.1, # income innovation std
    T: int = 25,
    cash_range: tuple[float, float] = (0.5, 20.0),
):
    def transition(state, action, shock, t):
        savings = state["cash"] - action["consume"]
        next_income = mu_y + sigma_y * shock["z"]
        return {"cash": R * savings + next_income}

    def reward(state, action, shock, t):
        c = action["consume"]
        if gamma == 1.0:
            return torch.log(c)
        return (c ** (1.0 - gamma)) / (1.0 - gamma)

    return Problem(
        states=[ContinuousState("cash", warp="asinh", range=cash_range)],
        actions=[ContinuousAction("consume", bounds=(1e-4, "cash"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, T),
        discount=beta,
    )


# --- qualitative tests ---------------------------------------------------


def test_borrowing_constraint_binds_at_low_cash():
    """At very low cash, the household consumes nearly all of it (MPC ≈ 1)."""
    problem = _build_problem(gamma=2.0, R=1.04, beta=0.94)
    policy, _ = solve(
        problem,
        state_grid={"cash": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    cash_low = torch.tensor([0.6, 0.8, 1.0], dtype=torch.float64)
    c = policy({"cash": cash_low}, t=5)["consume"]
    rates = (c / cash_low).tolist()
    # At low cash, c/cash should be very close to 1 (constraint binding)
    assert all(r > 0.85 for r in rates), f"c/cash at low cash = {rates}"


def test_household_saves_at_high_cash():
    """At high cash, the household consumes less than its cash (saves)."""
    problem = _build_problem(gamma=2.0, R=1.04, beta=0.94)
    policy, _ = solve(
        problem,
        state_grid={"cash": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    cash_high = torch.tensor([10.0, 15.0], dtype=torch.float64)
    c = policy({"cash": cash_high}, t=5)["consume"]
    rates = (c / cash_high).tolist()
    assert all(r < 0.5 for r in rates), f"c/cash at high cash = {rates}"


def test_consumption_is_monotonic_in_cash():
    """More cash → more consumption (the policy is increasing)."""
    problem = _build_problem(gamma=2.0, R=1.04, beta=0.94)
    policy, _ = solve(
        problem,
        state_grid={"cash": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    cash = torch.linspace(0.6, 15.0, 30, dtype=torch.float64)
    c = policy({"cash": cash}, t=5)["consume"]
    diffs = torch.diff(c)
    # Monotone increasing (allow tiny dips from grid discretization)
    assert (diffs >= -1e-6).all(), f"non-monotone at indices {(diffs < -1e-6).nonzero()}"


def test_mpc_decreases_from_unit_at_constraint():
    """The marginal propensity to consume drops from ~1 (constrained) to a
    smaller positive number (unconstrained). We check that MPC at high cash
    is materially below MPC at low cash."""
    problem = _build_problem(gamma=2.0, R=1.04, beta=0.94)
    policy, _ = solve(
        problem,
        state_grid={"cash": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    def mpc(cash_center: float, h: float = 0.1) -> float:
        cash = torch.tensor(
            [cash_center - h, cash_center + h], dtype=torch.float64
        )
        c = policy({"cash": cash}, t=5)["consume"]
        return ((c[1] - c[0]) / (2 * h)).item()

    mpc_low = mpc(0.8)
    mpc_high = mpc(10.0)
    assert mpc_low > 0.8, f"MPC at low cash should be near 1; got {mpc_low}"
    assert mpc_high < 0.5, f"MPC at high cash should be << 1; got {mpc_high}"


# --- printed diagnostic table -------------------------------------------


def test_carroll_deaton_print_consumption_function():
    """Print the consumption function across cash levels at mid-horizon."""
    problem = _build_problem(gamma=2.0, R=1.04, beta=0.94)
    policy, value = solve(
        problem,
        state_grid={"cash": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    cash = torch.tensor(
        [0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0],
        dtype=torch.float64,
    )

    print()
    print(f"Carroll/Deaton  |  gamma=2.0, R=1.04, beta=0.94, "
          f"mu_y=1.0, sigma_y=0.1, T=25, t=5")
    print()
    print(f"{'cash':>8} {'consume':>10} {'savings':>10} "
          f"{'c/cash':>10} {'V(cash)':>12}")
    print("-" * 56)

    c = policy({"cash": cash}, t=5)["consume"]
    v = value({"cash": cash}, t=5)

    for i, w in enumerate(cash.tolist()):
        ci = c[i].item()
        savings = w - ci
        rate = ci / w
        vi = v[i].item()
        print(f"{w:>8.2f} {ci:>10.4f} {savings:>+10.4f} "
              f"{rate:>10.4f} {vi:>12.4f}")
    print()


# --- simulator integration ----------------------------------------------


def test_simulated_wealth_settles_near_buffer_target():
    """Forward simulation: starting from low cash, the household builds up
    a buffer; the long-run mean cash settles into a sensible range."""
    problem = _build_problem(gamma=2.0, R=1.04, beta=0.94, T=50)
    policy, _ = solve(
        problem,
        state_grid={"cash": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    paths = simulate(
        policy=policy, problem=problem,
        n=2_000, initial_state={"cash": 0.8}, seed=0,
    )

    # Cash should grow and settle. Compare average cash early vs late.
    mean_early = paths["cash"][:, 1:6].mean().item()
    mean_late = paths["cash"][:, 30:50].mean().item()
    assert mean_late > mean_early, (
        f"cash should accumulate over time: early={mean_early:.3f}, "
        f"late={mean_late:.3f}"
    )
    # Buffer target for these params is typically a few units of income; check
    # the late mean is positive and finite (not blowing up or collapsing).
    assert 1.0 < mean_late < 20.0, f"unreasonable late mean cash = {mean_late}"
