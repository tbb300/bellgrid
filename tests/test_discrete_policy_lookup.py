"""Discrete-action policy lookup must be nearest-neighbour, never interpolated.

A discrete action index is a label, not a quantity. Averaging the optimum at
two grid nodes -- index 13 at one and 52 at the other -- yields index 33, which
is generally optimal at neither and need not resemble either in behaviour.

This regressed once: the lookup interpolated-and-rounded whenever a problem had
more than one continuous state (the single-state path was already correct). It
was silent -- the value function stays right, only the *applied* action is
wrong -- and on a lifecycle model it made a few percent of simulated households
consume their entire balance in one year.
"""
import torch

from bellgrid import ContinuousState, DiscreteAction, Problem, solve
from bellgrid.grids import RegularGrid
from bellgrid.solvers import BackwardInduction


def _step_problem():
    """Optimal action jumps from 0 to 3 across x = 0.25, with 4 choices."""

    def desired(x):
        return torch.where(x < 0.25, 0, 3)

    def reward(state, action, shock, t, next_state):
        return (action["a"] == desired(state["x"])).to(torch.float64)

    return Problem(
        states=[ContinuousState("x", range=(0.0, 1.0)),
                ContinuousState("y", range=(0.0, 1.0))],
        actions=[DiscreteAction("a", n=4)],
        transition=lambda s, a, sh, t: {"x": s["x"], "y": s["y"]},
        reward=reward,
        shocks=[],
        horizon=range(0, 1),
        discount=lambda s, t: 0.0,
        terminal_reward=lambda s: torch.zeros_like(s["x"]),
    )


def _solve():
    return solve(
        _step_problem(),
        state_grid={"x": RegularGrid(n=3), "y": RegularGrid(n=2)},
        action_grid={},
        solver=BackwardInduction(n_quad=1),
    )[0]


def test_returns_only_actions_some_node_actually_wants():
    """Off-grid queries must never invent an action no grid node chose."""
    policy = _solve()
    xs = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.24, 0.26, 0.3, 0.4,
                       0.5, 0.6, 0.75, 0.9, 1.0], dtype=torch.float64)
    q = {"x": xs, "y": torch.full_like(xs, 0.5)}
    got = policy(q, 0)["a"]
    # Grid nodes are x in {0.0, 0.5, 1.0}; their optima are {0, 3, 3}.
    assert set(int(v) for v in got) <= {0, 3}, (
        f"policy produced actions no node chose: {sorted(set(int(v) for v in got))}"
    )


def test_matches_the_nearer_grid_node():
    """Each query resolves to the optimum of whichever node is closer."""
    policy = _solve()
    # x < 0.25 is nearer node 0.0 (wants 0); x > 0.25 is nearer node 0.5 (wants 3).
    for xq, want in ((0.0, 0), (0.10, 0), (0.24, 0), (0.26, 3), (0.5, 3), (1.0, 3)):
        q = {"x": torch.tensor([xq], dtype=torch.float64),
             "y": torch.tensor([0.5], dtype=torch.float64)}
        assert int(policy(q, 0)["a"][0]) == want, f"x={xq}"


def test_exact_on_grid_nodes():
    """Sanity: on a node the answer is that node's own optimum."""
    policy = _solve()
    for xq, want in ((0.0, 0), (0.5, 3), (1.0, 3)):
        q = {"x": torch.tensor([xq], dtype=torch.float64),
             "y": torch.tensor([0.25], dtype=torch.float64)}
        assert int(policy(q, 0)["a"][0]) == want, f"node x={xq}"
