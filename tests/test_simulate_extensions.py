"""Tests for the simulate() extensions: callable discount, infinite
horizon, and next-state-aware reward."""

import math

import pytest
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    Problem,
    simulate,
    solve,
)
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction, PolicyIteration


# --- callable discount ---------------------------------------------------


def test_simulate_with_constant_callable_discount_matches_scalar():
    """A callable discount that returns a constant must give the same
    per-path discounted total as the equivalent scalar discount."""
    beta = 0.9

    def transition(state, _a, _sh, _t):
        return {"x": state["x"] + 1.0}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(1.0, dtype=torch.float64)

    common = dict(
        states=[ContinuousState("x", range=(0.0, 100.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 5),
    )
    problem_scalar = Problem(discount=beta, **common)
    problem_callable = Problem(discount=lambda _s, _t: beta, **common)
    policy_s, _ = solve(problem_scalar,
                       state_grid={"x": RegularGrid(n=16)}, action_grid={},
                       solver=BackwardInduction(n_quad=1))
    policy_c, _ = solve(problem_callable,
                       state_grid={"x": RegularGrid(n=16)}, action_grid={},
                       solver=BackwardInduction(n_quad=1))

    paths_s = simulate(policy=policy_s, problem=problem_scalar, n=10,
                      initial_state={"x": 0.0}, seed=0)
    paths_c = simulate(policy=policy_c, problem=problem_callable, n=10,
                      initial_state={"x": 0.0}, seed=0)
    assert torch.allclose(
        paths_s["discounted_total"], paths_c["discounted_total"], atol=1e-12
    )


def test_simulate_callable_discount_state_dependent():
    """State-dependent discount accumulates per-path correctly.

    Setup: 4 periods, reward = 1 each period. Discount halves the
    factor when alive=1 each period; alive starts at 1 and stays 1.
    Per-period factor: 0.5 each period.
    Expected discounted total = 1 + 0.5 + 0.25 + 0.125 = 1.875.
    """

    def transition(state, _a, _sh, _t):
        return {"alive": state["alive"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(1.0, dtype=torch.float64)

    def discount(state, _t):
        return torch.where(
            state["alive"] == 1,
            torch.tensor(0.5, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )

    problem = Problem(
        states=[DiscreteState("alive", n=2)],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 4), discount=discount,
    )
    policy, _ = solve(problem,
                     state_grid={}, action_grid={},
                     solver=BackwardInduction(n_quad=1))
    paths = simulate(policy=policy, problem=problem, n=3,
                    initial_state={"alive": 1}, seed=0)
    expected = 1.0 + 0.5 + 0.25 + 0.125
    for v in paths["discounted_total"]:
        assert v.item() == pytest.approx(expected, abs=1e-12)


# --- infinite horizon ----------------------------------------------------


def test_simulate_infinite_horizon_requires_n_periods():
    """Calling simulate on an infinite-horizon Problem without n_periods
    is a clear ValueError."""

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=None, discount=0.9,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=8)}, action_grid={},
                     solver=PolicyIteration(n_quad=1, tol=1e-6))
    with pytest.raises(ValueError, match="n_periods"):
        simulate(policy=policy, problem=problem, n=5,
                initial_state={"x": 0.5}, seed=0)


def test_simulate_infinite_horizon_stationary_policy():
    """simulate over a PolicyIteration solution accumulates rewards under
    the stationary policy. Geometric-series cross-check: with constant
    reward 1 and beta=0.9 for n_periods=20, total = (1-0.9^20)/(1-0.9)."""

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(1.0, dtype=torch.float64)

    beta = 0.9
    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=None, discount=beta,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=8)}, action_grid={},
                     solver=PolicyIteration(n_quad=1, tol=1e-6))

    n_periods = 20
    paths = simulate(policy=policy, problem=problem, n=4,
                    initial_state={"x": 0.5}, n_periods=n_periods, seed=0)
    expected = (1.0 - beta ** n_periods) / (1.0 - beta)
    for v in paths["discounted_total"]:
        assert v.item() == pytest.approx(expected, abs=1e-12)


def test_simulate_finite_with_n_periods_mismatch_raises():
    """Passing n_periods that doesn't match a finite horizon is a clear error."""
    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 5), discount=0.9,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=8)}, action_grid={},
                     solver=BackwardInduction(n_quad=1))
    with pytest.raises(ValueError, match="does not match horizon length"):
        simulate(policy=policy, problem=problem, n=3,
                initial_state={"x": 0.5}, n_periods=10, seed=0)


# --- 5-arg reward in simulate -------------------------------------------


def test_simulate_with_next_state_aware_reward():
    """5-arg reward in simulate sees next_state and matches the value
    expected from solve()."""

    def transition(state, _a, _sh, _t):
        return {"x": state["x"] + 1.0}

    def reward(_s, _a, _sh, t, next_state):
        # Per-period reward = next_x. Over 3 periods starting from x=0:
        # period 0: next_x = 1
        # period 1: next_x = 2
        # period 2: next_x = 3
        # undiscounted total = 6
        return next_state["x"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 10.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 3), discount=1.0,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=16)}, action_grid={},
                     solver=BackwardInduction(n_quad=1))
    paths = simulate(policy=policy, problem=problem, n=5,
                    initial_state={"x": 0.0}, seed=0)
    for v in paths["discounted_total"]:
        assert v.item() == pytest.approx(1.0 + 2.0 + 3.0, abs=1e-12)
