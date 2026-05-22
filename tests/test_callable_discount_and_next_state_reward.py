"""Tests for callable discount and next-state-aware reward.

Together these two API additions express mortality + bequest cleanly:

    V_t(s) = max_a  u(c) + β · E[ p_survive(t)·V_{t+1}(s') + (1-p_survive(t))·Bequest(s') ]

  callable-discount         next-state-aware reward
  → β · p_survive(t)      → u(c) + β·(1-p_survive(t))·Bequest(next_state)
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
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction


# --- callable discount: scalar-callable equivalent to constant -----------


def test_callable_discount_constant_matches_scalar():
    """discount returning a constant tensor must match passing the same
    constant as a float."""
    beta = 0.92

    def transition(state, action, _sh, _t):
        return {"x": state["x"] - 0.1 * action["consume"]}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"] + 1e-6)

    common = dict(
        states=[ContinuousState("x", range=(0.1, 5.0))],
        actions=[ContinuousAction("consume", bounds=(0.01, 0.5))],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 6),
        terminal_reward=lambda s: torch.log(s["x"]),
    )
    p_scalar = Problem(discount=beta, **common)
    p_callable = Problem(
        discount=lambda _s, _t: beta,  # callable returning a Python float
        **common,
    )

    state_grid = {"x": RegularGrid(n=16)}
    action_grid = {"consume": RegularGrid(n=20)}
    solver = BackwardInduction(n_quad=1)

    _, value_s = solve(p_scalar, state_grid=state_grid, action_grid=action_grid, solver=solver)
    _, value_c = solve(p_callable, state_grid=state_grid, action_grid=action_grid, solver=solver)

    q = {"x": torch.tensor([0.5, 1.0, 2.5], dtype=torch.float64)}
    v_s = value_s(q, t=2).numpy()
    v_c = value_c(q, t=2).numpy()
    for vs, vc in zip(v_s, v_c):
        assert vs == pytest.approx(vc, abs=1e-12)


def test_callable_discount_state_dependent():
    """discount depending on state changes V predictably.

    Toy: a 'frozen' state has discount=0 (no continuation value), an
    'active' state has discount=beta. The agent's V at the frozen state
    should equal the immediate reward alone."""
    beta = 0.9

    def transition(state, action, _sh, _t):
        return {"x": state["x"], "alive": state["alive"]}

    def reward(_s, action, _sh, _t):
        return action["consume"]

    def discount(state, _t):
        # alive=1 → beta, alive=0 → 0
        return torch.where(
            state["alive"] == 1,
            torch.tensor(beta, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        )

    from bellgrid import DiscreteState
    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            DiscreteState("alive", n=2),
        ],
        actions=[ContinuousAction("consume", bounds=(0.0, 1.0))],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 4),
        discount=discount,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={"consume": RegularGrid(n=4)},
        solver=BackwardInduction(n_quad=1),
    )

    # At alive=0 (discount=0), V = max_a r = max consume = 1.0 in every period.
    # At alive=1 (discount=beta), V_t = 1 + beta * V_{t+1} (geometric series).
    # With T=4 periods and V_T=0: V_3=1, V_2=1+0.9=1.9, V_1=1+0.9*1.9=2.71, V_0=1+0.9*2.71=3.439
    q_dead = {"x": torch.tensor([0.5], dtype=torch.float64),
              "alive": torch.tensor([0], dtype=torch.long)}
    q_alive = {"x": torch.tensor([0.5], dtype=torch.float64),
               "alive": torch.tensor([1], dtype=torch.long)}
    assert value(q_dead, t=0).item() == pytest.approx(1.0, abs=1e-12)
    assert value(q_alive, t=0).item() == pytest.approx(3.439, abs=1e-12)


# --- next-state-aware reward: 5-arg signature detected ------------------


def test_reward_with_next_state_is_detected_and_invoked():
    """A 5-arg reward signature is detected; the reward receives next_state."""
    captured = []

    def transition(state, action, _sh, _t):
        return {"x": state["x"] - 0.1 * action["consume"]}

    def reward(state, action, _sh, _t, next_state):
        captured.append({
            "x": state["x"].shape,
            "next_x": next_state["x"].shape,
        })
        return action["consume"] + 0.5 * next_state["x"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.1, 5.0))],
        actions=[ContinuousAction("consume", bounds=(0.0, 1.0))],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 2),
        discount=0.9,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={"consume": RegularGrid(n=4)},
        solver=BackwardInduction(n_quad=1),
    )
    assert len(captured) >= 1, "reward should have been called at least once"
    # next_state should have the same shape as state (full_shape during Bellman)
    for entry in captured:
        assert entry["x"] == entry["next_x"]
    v = value({"x": torch.tensor([1.0], dtype=torch.float64)}, t=0)
    assert torch.isfinite(v).all()


def test_reward_signature_rejection():
    """Reward with wrong number of args is rejected at solve time."""
    def transition(state, action, _sh, _t):
        return {"x": state["x"]}

    def bad_reward(state, action, shock):  # only 3 args
        return torch.tensor(0.0)

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[ContinuousAction("consume", bounds=(0.0, 1.0))],
        transition=transition, reward=bad_reward, shocks=[],
        horizon=range(0, 2), discount=0.9,
    )
    with pytest.raises(ValueError, match="reward must take 4 or 5"):
        solve(
            problem,
            state_grid={"x": RegularGrid(n=8)},
            action_grid={"consume": RegularGrid(n=4)},
            solver=BackwardInduction(n_quad=1),
        )


# --- end-to-end: mortality + bequest -------------------------------------


def test_mortality_bequest_lifecycle():
    """Exact-value test for a 3-period mortality + bequest problem.

    Setup: wealth state, single action c ∈ {0, w}, deterministic return R=1.
        - reward per period: u(c) = c
        - bequest: B(w) = w  (i.e. the agent values bequest at face value)
        - p_survive: 1.0 at t=0 (agent definitely survives period 0)
                     0.5 at t=1 (50/50 to survive period 1)
                     0.5 at t=2 (50/50 to survive period 2)
        - β = 1.0
        - terminal V_T(w) = 0  (no value after the horizon — Bequest is the
                                payoff during the period that death occurs)

    Closed-form V_0(w_0) under optimal policy (consume nothing, save all):
        V_2(w) = max_c c + β·(p_survive·V_3 + (1-p_survive)·B(w-c))
               = max_c c + 1·(0.5·0 + 0.5·(w-c))
               = max_c c + 0.5·(w-c)
               = max_c 0.5·c + 0.5·w   → optimum c=w
               = w
        V_1(w) = max_c c + 1·(0.5·V_2(w-c) + 0.5·B(w-c))
               = max_c c + 0.5·(w-c) + 0.5·(w-c)
               = max_c c + (w-c)
               = w  (c can be anything)
        V_0(w) = max_c c + 1·(1.0·V_1(w-c) + 0.0·B(w-c))
               = max_c c + (w-c) = w
    """
    p_survive_schedule = [1.0, 0.5, 0.5]
    beta = 1.0

    def transition(state, action, _sh, _t):
        return {"w": state["w"] - action["c"]}

    # Reward = u(c) + β · (1 - p_survive(t)) · Bequest(next_w)
    def reward(_s, action, _sh, t, next_state):
        ps = p_survive_schedule[t]
        return action["c"] + beta * (1.0 - ps) * next_state["w"]

    def discount(_state, t):
        return beta * p_survive_schedule[t]

    problem = Problem(
        states=[ContinuousState("w", range=(0.0, 10.0))],
        actions=[ContinuousAction("c", bounds=(0.0, "w"))],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 3),
        discount=discount,
    )
    _, value = solve(
        problem,
        state_grid={"w": RegularGrid(n=32)},
        action_grid={"c": RegularGrid(n=32)},
        solver=BackwardInduction(n_quad=1),
    )

    for w0 in (1.0, 3.0, 5.0, 8.0):
        v = value({"w": torch.tensor([w0], dtype=torch.float64)}, t=0).item()
        # Closed form: V_0(w) = w  (every dollar saved or consumed contributes 1 to V)
        assert v == pytest.approx(w0, abs=0.05), (
            f"V_0({w0}) = {v}, expected {w0}"
        )


def test_mortality_bequest_with_concave_utility():
    """Same mortality+bequest setup but with log utility: optimum becomes
    state-dependent and the closed-form match has more bite."""
    p_survive_schedule = [1.0, 0.5]
    beta = 1.0

    def transition(state, action, _sh, _t):
        return {"w": state["w"] - action["c"]}

    def reward(_s, action, _sh, t, next_state):
        ps = p_survive_schedule[t]
        return torch.log(action["c"] + 1e-9) + beta * (1.0 - ps) * torch.log(
            next_state["w"] + 1e-9
        )

    def discount(_state, t):
        return beta * p_survive_schedule[t]

    problem = Problem(
        states=[ContinuousState("w", range=(0.1, 10.0))],
        actions=[ContinuousAction("c", bounds=(1e-3, "w"))],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 2),
        discount=discount,
    )
    _, value = solve(
        problem,
        state_grid={"w": WarpedGrid(n=64, warp="log")},
        action_grid={"c": RegularGrid(n=200)},
        solver=BackwardInduction(n_quad=1),
    )

    # At t=1 (last working period): V_1(w) = max_c log(c) + 0.5·log(w-c).
    # FOC: 1/c = 0.5/(w-c) → w-c = 0.5c → c = 2w/3. V_1(w) = log(2w/3) + 0.5·log(w/3).
    for w in (0.5, 1.0, 2.0, 5.0):
        v = value({"w": torch.tensor([w], dtype=torch.float64)}, t=1).item()
        c_star = 2.0 * w / 3.0
        v_star = math.log(c_star) + 0.5 * math.log(w - c_star)
        assert v == pytest.approx(v_star, abs=0.05), (
            f"V_1({w}) = {v}, expected {v_star}"
        )
