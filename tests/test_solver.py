import pytest
import torch

from bellgrid import ContinuousAction, ContinuousState, Problem, solve
from bellgrid.grids import RegularGrid
from bellgrid.solvers import BackwardInduction


def test_one_step_consume_all():
    """1-period problem, linear reward in consumption, no future:
    optimal action is to consume everything at the upper bound.
    """

    def transition(state, action, shock, t):
        return {"wealth": state["wealth"] - action["consume"]}

    def reward(state, action, shock, t):
        return action["consume"]

    problem = Problem(
        states=[ContinuousState("wealth", range=(0.0, 10.0))],
        actions=[ContinuousAction("consume", bounds=(0.0, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 1),
        discount=1.0,
    )

    policy, value = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=50)},
        action_grid={"consume": RegularGrid(n=200)},
        solver=BackwardInduction(),
    )

    test_states = {"wealth": torch.tensor([1.0, 5.0, 9.0], dtype=torch.float64)}
    optimal = policy(test_states, t=0)
    v = value(test_states, t=0)

    # consume should pin to upper bound (= wealth) up to action grid resolution
    assert torch.allclose(
        optimal["consume"], test_states["wealth"], rtol=1.0 / 200
    )
    # V(wealth) = wealth (consume all gives reward = wealth)
    assert torch.allclose(v, test_states["wealth"], rtol=1.0 / 200)


def test_two_step_deterministic_lookahead():
    """2-period problem with no discount: V[0](w) = 2w (consume all over 2 periods)."""

    def transition(state, action, shock, t):
        return {"wealth": state["wealth"] - action["consume"]}

    def reward(state, action, shock, t):
        return action["consume"]

    problem = Problem(
        states=[ContinuousState("wealth", range=(0.0, 10.0))],
        actions=[ContinuousAction("consume", bounds=(0.0, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 2),
        discount=1.0,
    )

    policy, value = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=100)},
        action_grid={"consume": RegularGrid(n=200)},
        solver=BackwardInduction(),
    )

    test_states = {"wealth": torch.tensor([1.0, 5.0, 9.0], dtype=torch.float64)}
    v0 = value(test_states, t=0)
    # With linear reward and no discount, total payoff = w (any split works).
    # The solver finds *a* maximizer; V(0) should equal w to grid tolerance.
    assert torch.allclose(v0, test_states["wealth"], rtol=2.0 / 100)


def test_state_dependent_bounds_respected():
    """consume can never exceed wealth: optimal action at every wealth point is <= wealth."""

    def transition(state, action, shock, t):
        return {"wealth": state["wealth"] - action["consume"]}

    def reward(state, action, shock, t):
        return action["consume"]

    problem = Problem(
        states=[ContinuousState("wealth", range=(0.0, 10.0))],
        actions=[ContinuousAction("consume", bounds=(0.0, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 3),
        discount=0.9,
    )

    policy, value = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=50)},
        action_grid={"consume": RegularGrid(n=100)},
        solver=BackwardInduction(),
    )

    w = torch.tensor([0.5, 1.0, 5.0, 9.0], dtype=torch.float64)
    actions = policy({"wealth": w}, t=0)
    # consume <= wealth at every state
    assert (actions["consume"] <= w + 1e-9).all()
    assert (actions["consume"] >= 0).all()


def test_solver_runs_with_normal_shock():
    """Smoke: stochastic problem with a single Normal shock runs end-to-end."""
    from bellgrid.shocks import Normal

    def transition(state, action, shock, t):
        # next wealth grows with a shocked return, then consumption is subtracted
        ret = 1.0 + 0.05 * shock["z"]
        return {"wealth": (state["wealth"] - action["consume"]) * ret}

    def reward(state, action, shock, t):
        return action["consume"]

    problem = Problem(
        states=[ContinuousState("wealth", range=(0.0, 10.0))],
        actions=[ContinuousAction("consume", bounds=(0.0, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 3),
        discount=0.95,
    )

    policy, value = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=40)},
        action_grid={"consume": RegularGrid(n=60)},
        solver=BackwardInduction(n_quad=5),
    )

    w = torch.tensor([1.0, 5.0], dtype=torch.float64)
    v = value({"wealth": w}, t=0)
    a = policy({"wealth": w}, t=0)
    assert v.shape == w.shape
    assert a["consume"].shape == w.shape
    assert (a["consume"] >= 0).all()
    assert (a["consume"] <= w + 1e-9).all()
