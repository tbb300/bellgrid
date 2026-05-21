from types import SimpleNamespace

import pytest

from bellgrid import ContinuousAction, ContinuousState, Problem


def _noop_transition(state, action, shock, t):
    return state


def _noop_reward(state, action, shock, t):
    return 0.0


def _basic_problem(**overrides):
    kwargs = dict(
        states=[ContinuousState("wealth", range=(0.0, 100.0))],
        actions=[ContinuousAction("consumption", bounds=(0.0, "wealth"))],
        transition=_noop_transition,
        reward=_noop_reward,
        shocks=[],
        horizon=range(10),
        discount=0.96,
    )
    kwargs.update(overrides)
    return Problem(**kwargs)


def test_valid_construction():
    p = _basic_problem()
    assert p.states[0].name == "wealth"
    assert p.actions[0].bounds == (0.0, "wealth")
    assert p.discount == 0.96


def test_horizon_none_allowed():
    p = _basic_problem(horizon=None)
    assert p.horizon is None


def test_state_action_name_collision():
    with pytest.raises(ValueError, match="Name collision"):
        _basic_problem(
            states=[ContinuousState("x", range=(0, 1))],
            actions=[ContinuousAction("x", bounds=(0, 1))],
        )


def test_state_shock_name_collision():
    shock = SimpleNamespace(name="wealth")
    with pytest.raises(ValueError, match="Name collision"):
        _basic_problem(shocks=[shock])


def test_action_shock_name_collision():
    shock = SimpleNamespace(name="consumption")
    with pytest.raises(ValueError, match="Name collision"):
        _basic_problem(shocks=[shock])


def test_nameless_shock_does_not_collide():
    shock = SimpleNamespace(name=None)
    p = _basic_problem(shocks=[shock])
    assert p.shocks == [shock]


def test_bound_references_unknown_state():
    with pytest.raises(ValueError, match="undeclared state"):
        _basic_problem(
            actions=[ContinuousAction("draw", bounds=(0, "income"))],
        )


def test_static_bounds_ok():
    p = _basic_problem(
        actions=[ContinuousAction("equity_share", bounds=(0.0, 1.0))],
    )
    assert p.actions[0].bounds == (0.0, 1.0)


def test_problem_is_frozen():
    p = _basic_problem()
    with pytest.raises(AttributeError):
        p.discount = 0.5
