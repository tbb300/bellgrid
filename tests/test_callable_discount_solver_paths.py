"""Regression tests: state-dependent callable discount on the solver paths
that evaluate the Bellman value at a non-N_a action axis.

A *state-dependent* callable discount (one that returns a tensor, e.g.
``torch.where(state["regime"] == 0, b0, b1)``) is evaluated on the joint
state mesh, so it arrives with size ``N_a`` on the action axis. Two solver
paths re-use the Bellman machinery at a *different* action-axis size:

  - ``PolicyIteration``'s Howard / modified-policy-iteration inner loop
    (``k_howard > 1``) evaluates ``T_σ`` at a single fixed action (M = 1);
  - ``GoldenSearch`` refinement evaluates candidates at M = N_enum.

Both used to broadcast the N_a-action discount against the M-action value
and crash. These tests pin the fix: the discount's action axis is collapsed
so it broadcasts, and the results match the plain-value-iteration / fine-grid
references.
"""

import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    Problem,
    solve,
)
from bellgrid.grids import GoldenSearch, RegularGrid, WarpedGrid
from bellgrid.solvers import BackwardInduction, PolicyIteration


def _b(x):
    return torch.tensor(x, dtype=torch.float64)


def test_callable_discount_policy_iteration_howard_matches_value_iteration():
    """State-dependent callable discount must work under modified policy
    iteration (default k_howard=10), not just plain value iteration
    (k_howard=1) — and both must converge to the same stationary V."""
    beta = 0.95

    def transition(state, action, _sh, _t):
        # consume c; the remainder mean-reverts back toward the interior.
        return {"x": 0.7 * (state["x"] - action["c"]) + 0.5,
                "regime": state["regime"]}

    def reward(_s, action, _sh, _t):
        return torch.log(action["c"] + 1e-9)

    def discount(state, _t):
        # state-dependent => returns a tensor with the action axis present
        return torch.where(state["regime"] == 0, _b(beta), _b(0.7 * beta))

    problem = Problem(
        states=[ContinuousState("x", range=(0.1, 5.0), warp="log"),
                DiscreteState("regime", n=2)],
        actions=[ContinuousAction("c", bounds=(1e-3, "x"))],
        transition=transition, reward=reward, shocks=[],
        horizon=None, discount=discount,
    )
    grids = dict(state_grid={"x": WarpedGrid(n=24, warp="log")},
                 action_grid={"c": RegularGrid(n=24)})

    _, v_vi = solve(
        problem, solver=PolicyIteration(n_quad=1, k_howard=1, boundary_check=False),
        **grids,
    )
    _, v_mpi = solve(
        problem, solver=PolicyIteration(n_quad=1, k_howard=10, boundary_check=False),
        **grids,
    )

    q = {"x": torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64),
         "regime": torch.tensor([0, 1, 0], dtype=torch.long)}
    v1 = v_vi(q, t=None)
    v10 = v_mpi(q, t=None)
    assert torch.isfinite(v10).all()
    # Same fixed point regardless of k_howard.
    assert torch.allclose(v1, v10, atol=1e-4), (v1, v10)


def test_callable_discount_with_golden_search_matches_fine_grid():
    """State-dependent callable discount must also work on the golden-search
    refinement path (which evaluates the Bellman value at M = N_enum != N_a),
    matching a fine RegularGrid solve of the same problem."""
    beta = 0.93

    def transition(state, action, _sh, _t):
        return {"w": state["w"] - action["c"], "regime": state["regime"]}

    def reward(_s, action, _sh, _t):
        return torch.log(action["c"] + 1e-9)

    def discount(state, _t):
        return torch.where(state["regime"] == 0, _b(beta), _b(0.5 * beta))

    common = dict(
        states=[ContinuousState("w", range=(0.1, 10.0), warp="log"),
                DiscreteState("regime", n=2)],
        actions=[ContinuousAction("c", bounds=(1e-3, "w"))],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 5), discount=discount,
    )
    problem = Problem(**common)

    _, v_golden = solve(
        problem,
        state_grid={"w": WarpedGrid(n=48, warp="log")},
        action_grid={"c": GoldenSearch(n_init=5, n_iter=30)},
        solver=BackwardInduction(n_quad=1, boundary_check=False),
    )
    _, v_grid = solve(
        problem,
        state_grid={"w": WarpedGrid(n=48, warp="log")},
        action_grid={"c": RegularGrid(n=400)},
        solver=BackwardInduction(n_quad=1, boundary_check=False),
    )

    q = {"w": torch.tensor([0.5, 1.0, 3.0, 7.0], dtype=torch.float64),
         "regime": torch.tensor([0, 1, 0, 1], dtype=torch.long)}
    vg = v_golden(q, t=0)
    vr = v_grid(q, t=0)
    assert torch.isfinite(vg).all()
    # Golden refinement is sharper than even a 400-point grid; they agree well.
    assert torch.allclose(vg, vr, atol=2e-3), (vg, vr)
