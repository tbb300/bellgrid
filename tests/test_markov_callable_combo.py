"""Combo tests: MarkovChain + callable discount + next-state-aware reward.

These primitives all interact in the lifecycle example. Each is tested
in isolation elsewhere, but the combination — callable discount that
depends on the regime, plus a 5-arg reward that sees next_state with
non-MC keys — was untested. Critical for the rl-inv2 lifecycle pattern.
"""

import numpy as np
import pytest
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    MarkovChain,
    Problem,
    simulate,
    solve,
)
from bellgrid.grids import RegularGrid
from bellgrid.shocks import Categorical, Normal, Uniform
from bellgrid.solvers import BackwardInduction


def test_callable_discount_sees_current_markov_regime():
    """Build a problem where ``discount(state, t)`` depends on the
    MarkovChain regime. Compare V across two regimes — the higher-
    discount regime should give a strictly larger continuation value."""
    P = np.array([[0.8, 0.2], [0.2, 0.8]])

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(1.0, dtype=torch.float64)

    def terminal(state):
        return torch.zeros_like(state["x"], dtype=torch.float64)

    # Discount = 0.5 when regime=0, 0.95 when regime=1
    def discount(state, _t):
        return torch.where(
            state["regime"] == 0,
            torch.tensor(0.5, dtype=torch.float64),
            torch.tensor(0.95, dtype=torch.float64),
        )

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 5), discount=discount,
        terminal_reward=terminal,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=4)}, action_grid={},
        solver=BackwardInduction(n_quad=1),
    )

    # In regime 0 (lower discount), V is smaller; in regime 1, larger.
    v0 = value({
        "x": torch.tensor([0.5], dtype=torch.float64),
        "regime": torch.tensor([0], dtype=torch.long),
    }, t=0).item()
    v1 = value({
        "x": torch.tensor([0.5], dtype=torch.float64),
        "regime": torch.tensor([1], dtype=torch.long),
    }, t=0).item()
    assert v1 > v0, f"v1 ({v1}) should exceed v0 ({v0}) under higher discount"


def test_5arg_reward_with_markov_chain_sees_correct_next_state_dict():
    """A 5-arg reward must receive a next_state dict containing all
    non-markov state keys (and NO markov keys). Verify by inspecting
    the value function output. We construct a reward that's a function
    purely of next_state['x'] and a deterministic transition, so V at
    time 0 over 1 step is exactly the next_state contribution."""
    P = np.array([[0.6, 0.4], [0.3, 0.7]])
    captured_keys = {}

    def transition(state, _a, _sh, _t):
        return {"x": state["x"] + 1.0}

    def reward(_s, _a, _sh, _t, next_state):
        # Capture the keys of next_state once for inspection
        if not captured_keys:
            captured_keys["keys"] = set(next_state.keys())
        return next_state["x"]

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 20.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[], horizon=range(0, 1), discount=1.0,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)}, action_grid={},
        solver=BackwardInduction(n_quad=1),
    )

    # next_state dict should contain "x" but not "regime"
    assert captured_keys["keys"] == {"x"}, (
        f"next_state keys were {captured_keys['keys']}, expected {{'x'}}"
    )

    # V at (x=5, regime=*) for 1 step = next_x = 6 (since action is no-op
    # and transition adds 1 deterministically; reward returns next_x).
    for reg in (0, 1):
        v = value({
            "x": torch.tensor([5.0], dtype=torch.float64),
            "regime": torch.tensor([reg], dtype=torch.long),
        }, t=0).item()
        assert v == pytest.approx(6.0, abs=1e-10)


def test_callable_discount_plus_5arg_reward_plus_markov():
    """End-to-end: callable discount depending on regime, 5-arg reward
    depending on next_state, MarkovChain regime evolution. Verify the
    final V at a representative state has the right sign/magnitude."""
    P = np.array([[0.5, 0.5], [0.5, 0.5]])  # uniform 50/50

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t, next_state):
        return next_state["x"]   # constant per state

    def discount(state, _t):
        # 0 in regime 0 (annihilates continuation), 1 in regime 1
        return (state["regime"] == 1).to(torch.float64)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 10.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[], horizon=range(0, 3), discount=discount,
        terminal_reward=lambda s: torch.zeros_like(s["x"]),
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)}, action_grid={},
        solver=BackwardInduction(n_quad=1),
    )

    x_val = 5.0
    # In regime 0: V(s_0) = r(s_0) + 0 * E[V(s_1)] = x = 5
    # In regime 1: V(s_0) = r(s_0) + 1 * E[V(s_1)]
    #   where E[V(s_1)] under uniform P = 0.5*V(x, r=0)+0.5*V(x, r=1)
    #   V(x, r=0 at t=1) = x + 0*E[...] = x = 5
    #   V(x, r=1 at t=1) = x + 1*E[V(x at t=2)] = x + 0.5*x + 0.5*(x+0) = ...
    # Easier: at t=2 (last step), V = r = x in both regimes.
    # At t=1, V(r=0) = x + 0*… = x; V(r=1) = x + 1*0.5*(x+x) = x + x = 2x
    # At t=0, V(r=0) = x + 0*… = x = 5; V(r=1) = x + 0.5*(x + 2x) = x + 1.5x = 2.5x = 12.5
    v_r0 = value({"x": torch.tensor([x_val], dtype=torch.float64),
                  "regime": torch.tensor([0], dtype=torch.long)}, t=0).item()
    v_r1 = value({"x": torch.tensor([x_val], dtype=torch.float64),
                  "regime": torch.tensor([1], dtype=torch.long)}, t=0).item()
    assert v_r0 == pytest.approx(x_val, abs=1e-10)
    assert v_r1 == pytest.approx(2.5 * x_val, abs=1e-10)


def test_markov_with_categorical_shock_and_5arg_reward():
    """Categorical shock + MarkovChain + 5-arg reward. Verifies the
    next_state dict is correctly assembled and the reward sees the
    right shock realisations across the joint mesh."""
    P = np.array([[0.5, 0.5], [0.5, 0.5]])

    def transition(state, _a, shock, _t):
        return {"x": state["x"] + shock["jump"]}

    def reward(_s, _a, _sh, _t, next_state):
        return next_state["x"]   # constant per state

    problem = Problem(
        states=[
            ContinuousState("x", range=(-10.0, 30.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Categorical("jump", values=(-1.0, 0.0, 1.0),
                            probabilities=(0.25, 0.5, 0.25))],
        horizon=range(0, 1), discount=1.0,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=16)}, action_grid={},
        solver=BackwardInduction(n_quad=1),
    )

    # 1-step V at x=5 = E[x + jump] = 5 + (0.25*-1 + 0.5*0 + 0.25*1) = 5.0
    for reg in (0, 1):
        v = value({"x": torch.tensor([5.0], dtype=torch.float64),
                   "regime": torch.tensor([reg], dtype=torch.long)}, t=0).item()
        assert v == pytest.approx(5.0, abs=1e-10)


def test_markov_with_uniform_shock_and_5arg_reward():
    """Uniform shock + MarkovChain + 5-arg reward. Same template as
    above but with a continuous shock + Gauss-Legendre quadrature."""
    P = np.array([[0.5, 0.5], [0.5, 0.5]])

    def transition(state, _a, shock, _t):
        return {"x": state["x"] + shock["jump"]}

    def reward(_s, _a, _sh, _t, next_state):
        return next_state["x"]

    problem = Problem(
        states=[
            ContinuousState("x", range=(-10.0, 30.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Uniform("jump", low=-1.0, high=1.0)],
        horizon=range(0, 1), discount=1.0,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=16)}, action_grid={},
        solver=BackwardInduction(n_quad=5),
    )

    # E[U] = 0 → 1-step V at x=5 = 5.0
    for reg in (0, 1):
        v = value({"x": torch.tensor([5.0], dtype=torch.float64),
                   "regime": torch.tensor([reg], dtype=torch.long)}, t=0).item()
        assert v == pytest.approx(5.0, abs=1e-10)
