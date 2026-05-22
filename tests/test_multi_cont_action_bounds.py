"""Tests for state-dependent action bounds with K_cont > 1."""

import pytest
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    Problem,
    solve,
)
from bellgrid.grids import RegularGrid
from bellgrid.solvers import BackwardInduction


def test_state_dep_bound_referencing_first_continuous_state_with_kcont_2():
    """Two continuous states (w, x); action bound c ≤ w. Verify the
    optimal c at the maximum action grid index equals w on a sweep of
    w values, regardless of x."""

    def transition(state, action, _sh, _t):
        return {"w": state["w"] - action["c"], "x": state["x"]}

    def reward(_s, action, _sh, _t):
        return action["c"]   # reward = c, max c is optimal

    problem = Problem(
        states=[
            ContinuousState("w", range=(0.0, 10.0)),
            ContinuousState("x", range=(-1.0, 1.0)),
        ],
        actions=[ContinuousAction("c", bounds=(0.0, "w"))],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 1),
        discount=0.9,
    )
    policy, value = solve(
        problem,
        state_grid={
            "w": RegularGrid(n=11),
            "x": RegularGrid(n=5),
        },
        action_grid={"c": RegularGrid(n=20)},
        solver=BackwardInduction(n_quad=1),
    )

    # The action grid linearly interpolates [0, w] across 20 points;
    # the optimal (reward = c) is the upper end, i.e. c = w.
    for w_test in (1.0, 5.0, 9.0):
        for x_test in (-0.5, 0.0, 0.5):
            a = policy({
                "w": torch.tensor([w_test], dtype=torch.float64),
                "x": torch.tensor([x_test], dtype=torch.float64),
            }, t=0)
            assert a["c"].item() == pytest.approx(w_test, abs=0.1), (
                f"optimal c at (w={w_test}, x={x_test}) = {a['c'].item()}, "
                f"expected ~{w_test}"
            )


def test_state_dep_bound_referencing_second_continuous_state():
    """Two continuous states; action bound references the SECOND continuous
    state. Verifies the axis-position logic correctly identifies which
    state to use for the bound (not just the first)."""

    def transition(state, action, _sh, _t):
        return {"w": state["w"], "x": state["x"]}

    def reward(_s, action, _sh, _t):
        return action["c"]   # max c is optimal

    problem = Problem(
        states=[
            ContinuousState("w", range=(0.0, 100.0)),
            ContinuousState("x", range=(0.0, 5.0)),
        ],
        actions=[ContinuousAction("c", bounds=(0.0, "x"))],  # bound on x (the SECOND state)
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 1),
        discount=0.9,
    )
    policy, _ = solve(
        problem,
        state_grid={
            "w": RegularGrid(n=8),
            "x": RegularGrid(n=11),
        },
        action_grid={"c": RegularGrid(n=20)},
        solver=BackwardInduction(n_quad=1),
    )
    # Optimal c = x (the upper bound), independent of w
    for w_test in (10.0, 50.0, 90.0):
        for x_test in (0.5, 2.5, 4.5):
            a = policy({
                "w": torch.tensor([w_test], dtype=torch.float64),
                "x": torch.tensor([x_test], dtype=torch.float64),
            }, t=0)
            assert a["c"].item() == pytest.approx(x_test, abs=0.1)


def test_state_dep_bound_with_discrete_and_continuous():
    """K_cont = 2 plus a DiscreteState. Action bound on one continuous;
    correct axis-position resolution across the mix."""

    def transition(state, action, _sh, _t):
        return {
            "w": state["w"] - action["c"],
            "x": state["x"],
            "phase": state["phase"],
        }

    def reward(_s, action, _sh, _t):
        return action["c"]

    from bellgrid import DiscreteState
    problem = Problem(
        states=[
            ContinuousState("w", range=(0.0, 10.0)),
            ContinuousState("x", range=(0.0, 1.0)),
            DiscreteState("phase", n=2),
        ],
        actions=[ContinuousAction("c", bounds=(0.0, "w"))],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 1),
        discount=0.9,
    )
    policy, _ = solve(
        problem,
        state_grid={
            "w": RegularGrid(n=11),
            "x": RegularGrid(n=5),
        },
        action_grid={"c": RegularGrid(n=20)},
        solver=BackwardInduction(n_quad=1),
    )
    for w_test in (2.0, 5.0, 9.0):
        for phase in (0, 1):
            a = policy({
                "w": torch.tensor([w_test], dtype=torch.float64),
                "x": torch.tensor([0.5], dtype=torch.float64),
                "phase": torch.tensor([phase], dtype=torch.long),
            }, t=0)
            assert a["c"].item() == pytest.approx(w_test, abs=0.1)
