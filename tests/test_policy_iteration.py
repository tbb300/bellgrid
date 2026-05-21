"""Tests for the PolicyIteration (infinite-horizon) solver."""

import math

import numpy as np
import pytest
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    Problem,
    solve,
)
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction, PolicyIteration


# --- API plumbing -------------------------------------------------------


def test_policy_iteration_requires_horizon_none():
    """Passing a finite horizon should be a clear error."""
    def transition(state, action, _sh, _t):
        return {"x": state["x"] - 0.1 * action["consume"]}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"] + 1e-6)

    problem = Problem(
        states=[ContinuousState("x", range=(0.1, 5.0))],
        actions=[ContinuousAction("consume", bounds=(0.01, 0.5))],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 10),   # finite — wrong for PolicyIteration
        discount=0.9,
    )
    with pytest.raises(ValueError, match="horizon=None"):
        solve(
            problem,
            state_grid={"x": RegularGrid(n=8)},
            action_grid={"consume": RegularGrid(n=10)},
            solver=PolicyIteration(tol=1e-4, max_iters=100),
        )


def test_backward_induction_requires_horizon_not_none():
    """And BackwardInduction must reject horizon=None."""
    def transition(state, action, _sh, _t):
        return {"x": state["x"] - 0.1 * action["consume"]}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"] + 1e-6)

    problem = Problem(
        states=[ContinuousState("x", range=(0.1, 5.0))],
        actions=[ContinuousAction("consume", bounds=(0.01, 0.5))],
        transition=transition, reward=reward, shocks=[],
        horizon=None,
        discount=0.9,
    )
    with pytest.raises(NotImplementedError, match="finite horizon"):
        solve(
            problem,
            state_grid={"x": RegularGrid(n=8)},
            action_grid={"consume": RegularGrid(n=10)},
            solver=BackwardInduction(),
        )


def test_policy_iteration_max_iters_overflow_raises():
    """If max_iters too low to converge, expect a clean RuntimeError."""
    beta, mu, sigma = 0.96, 0.04, 0.15

    def transition(state, action, shock, _t):
        return {"wealth": (state["wealth"] - action["consume"])
                * torch.exp(mu + sigma * shock["z"])}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    problem = Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 200.0))],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition, reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=None,
        discount=beta,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        solve(
            problem,
            state_grid={"wealth": WarpedGrid(n=32)},
            action_grid={"consume": RegularGrid(n=100)},
            solver=PolicyIteration(n_quad=5, tol=1e-10, max_iters=3),
        )


# --- Correctness: infinite-horizon Merton matches the closed form -------


def test_infinite_horizon_merton_consumption_rate():
    """Log-utility Merton, infinite horizon: c/w = 1 − β exactly."""
    beta, mu, sigma = 0.96, 0.04, 0.15

    def transition(state, action, shock, _t):
        return {"wealth": (state["wealth"] - action["consume"])
                * torch.exp(mu + sigma * shock["z"])}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    problem = Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 200.0))],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition, reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=None,
        discount=beta,
    )

    policy, value = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=PolicyIteration(n_quad=7, tol=1e-7),
    )

    w = torch.tensor([2.0, 10.0, 25.0, 50.0], dtype=torch.float64)
    rates = (policy({"wealth": w}, t=None)["consume"] / w).tolist()
    expected = 1.0 - beta
    for rate in rates:
        assert rate == pytest.approx(expected, abs=0.003)


def test_infinite_horizon_merton_value_matches_closed_form():
    """V(w) = A + B log w under log utility, infinite horizon."""
    beta, mu, sigma = 0.96, 0.04, 0.15
    B = 1.0 / (1.0 - beta)
    A = (
        math.log(1.0 - beta) / (1.0 - beta)
        + (beta / (1.0 - beta) ** 2) * (math.log(beta) + mu)
    )

    def transition(state, action, shock, _t):
        return {"wealth": (state["wealth"] - action["consume"])
                * torch.exp(mu + sigma * shock["z"])}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    problem = Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 200.0))],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition, reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=None,
        discount=beta,
    )

    _, value = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=PolicyIteration(n_quad=7, tol=1e-7),
    )

    # Closed form: V(w) = A + B log w. Compare at a few interior wealths.
    # Worst-case error at the grid edge is dominated by interpolation;
    # interior points should be within ~1% of A + B log w.
    for w_val in (2.0, 5.0, 10.0, 25.0, 50.0):
        v_bg = value(
            {"wealth": torch.tensor([w_val], dtype=torch.float64)}, t=None
        ).item()
        v_cf = A + B * math.log(w_val)
        assert v_bg == pytest.approx(v_cf, abs=0.5)


def test_infinite_matches_long_finite_horizon_via_closed_form():
    """Both BI(long finite, closed-form terminal) and PI should be close
    to the closed-form V* at mid-horizon / steady state. They won't match
    each other exactly — discretization errors push the two iterates to
    slightly different sides of V* — but each should be near V*."""
    beta, mu, sigma = 0.96, 0.04, 0.15
    B = 1.0 / (1.0 - beta)
    A = (
        math.log(1.0 - beta) / (1.0 - beta)
        + (beta / (1.0 - beta) ** 2) * (math.log(beta) + mu)
    )

    def transition(state, action, shock, _t):
        return {"wealth": (state["wealth"] - action["consume"])
                * torch.exp(mu + sigma * shock["z"])}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    T = 50
    problem_finite = Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 200.0))],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition, reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, T), discount=beta,
        terminal_reward=lambda s: A + B * torch.log(s["wealth"]),
    )
    problem_infinite = Problem(
        states=problem_finite.states,
        actions=problem_finite.actions,
        transition=transition, reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=None, discount=beta,
    )

    _, value_finite = solve(
        problem_finite,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )
    _, value_infinite = solve(
        problem_infinite,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=PolicyIteration(n_quad=7, tol=1e-7),
    )

    for w in (5.0, 15.0, 30.0):
        q = {"wealth": torch.tensor([w], dtype=torch.float64)}
        v_f = value_finite(q, t=T // 2).item()
        v_i = value_infinite(q, t=None).item()
        v_star = A + B * math.log(w)
        # Both BI and PI within ~0.2 of V* (discretization floor; the two
        # iterates can sit on opposite sides of V* and so diverge from
        # each other by up to 2× this).
        assert v_f == pytest.approx(v_star, abs=0.2), \
            f"BI at t={T//2}, w={w}: got {v_f}, expected V*={v_star}"
        assert v_i == pytest.approx(v_star, abs=0.2), \
            f"PI at w={w}: got {v_i}, expected V*={v_star}"
