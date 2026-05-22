"""Tests for the Uniform shock (Gauss-Legendre quadrature)."""

import math

import pytest
import torch

from bellgrid import ContinuousState, DiscreteAction, Problem, solve
from bellgrid.grids import RegularGrid
from bellgrid.shocks import Uniform
from bellgrid.solvers import BackwardInduction


# --- construction + validation -------------------------------------------


def test_uniform_construction():
    u = Uniform("r", low=-1.0, high=2.0)
    assert u.name == "r"
    assert u.low == -1.0
    assert u.high == 2.0


def test_uniform_default_is_zero_to_one():
    u = Uniform("r")
    assert u.low == 0.0
    assert u.high == 1.0


def test_uniform_empty_name_raises():
    with pytest.raises(ValueError, match="requires a name"):
        Uniform("", low=0.0, high=1.0)


def test_uniform_high_equal_low_raises():
    with pytest.raises(ValueError, match="high must be"):
        Uniform("r", low=1.0, high=1.0)


def test_uniform_high_below_low_raises():
    with pytest.raises(ValueError, match="high must be"):
        Uniform("r", low=2.0, high=1.0)


def test_uniform_invalid_n_quad_raises():
    u = Uniform("r")
    with pytest.raises(ValueError, match="n_quad must be"):
        u.nodes_and_weights(0)


def test_uniform_is_frozen():
    u = Uniform("r")
    with pytest.raises(AttributeError):
        u.low = -1.0


# --- quadrature ----------------------------------------------------------


def test_uniform_weights_sum_to_one():
    u = Uniform("r", low=-1.0, high=3.0)
    for n_quad in (1, 3, 5, 7, 11):
        _, w = u.nodes_and_weights(n_quad)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_uniform_nodes_within_bounds():
    u = Uniform("r", low=-1.0, high=3.0)
    nodes, _ = u.nodes_and_weights(11)
    assert (nodes >= -1.0).all()
    assert (nodes <= 3.0).all()


def test_uniform_first_moment_matches_closed_form():
    """E[X] = (low+high)/2 for X ~ Uniform(low, high). Exact at any n_quad ≥ 1."""
    low, high = -1.0, 3.0
    u = Uniform("r", low=low, high=high)
    expected = (low + high) / 2.0
    for n_quad in (1, 3, 5, 7):
        nodes, weights = u.nodes_and_weights(n_quad)
        em = (weights * nodes).sum().item()
        assert em == pytest.approx(expected, abs=1e-12)


def test_uniform_second_moment_matches_closed_form():
    """E[X^2] = (high^3 − low^3) / (3·(high−low)). Exact at any n_quad ≥ 2."""
    low, high = -1.0, 3.0
    u = Uniform("r", low=low, high=high)
    expected = (high ** 3 - low ** 3) / (3.0 * (high - low))
    for n_quad in (2, 3, 5, 7):
        nodes, weights = u.nodes_and_weights(n_quad)
        em2 = (weights * nodes ** 2).sum().item()
        assert em2 == pytest.approx(expected, abs=1e-12)


def test_uniform_third_moment_via_quadrature():
    """E[X^3] = (high^4 − low^4) / (4·(high−low)). Exact at any n_quad ≥ 2."""
    low, high = -2.0, 4.0
    u = Uniform("r", low=low, high=high)
    expected = (high ** 4 - low ** 4) / (4.0 * (high - low))
    nodes, weights = u.nodes_and_weights(3)
    em3 = (weights * nodes ** 3).sum().item()
    assert em3 == pytest.approx(expected, abs=1e-12)


# --- sampling ------------------------------------------------------------


def test_uniform_sample_within_bounds():
    u = Uniform("r", low=-1.0, high=3.0)
    out = u.sample(10_000, generator=torch.Generator().manual_seed(0))
    assert (out >= -1.0).all()
    assert (out <= 3.0).all()


def test_uniform_sample_mean_matches_target():
    u = Uniform("r", low=-1.0, high=3.0)
    out = u.sample(50_000, generator=torch.Generator().manual_seed(0))
    em = out.mean().item()
    expected = (-1.0 + 3.0) / 2.0
    # Std of empirical mean of 50k draws of Uniform[-1, 3] ≈ sqrt(16/12/50000) ≈ 0.0052.
    assert abs(em - expected) < 0.05


# --- solver integration --------------------------------------------------


def test_uniform_in_solver_recovers_expected_reward():
    """Bellman one-step with reward = shock['r'] under Uniform[2, 4]
    should give V_0 = 3 (the mean)."""

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["r"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Uniform("r", low=2.0, high=4.0)],
        horizon=range(0, 1),
        discount=1.0,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={},
        solver=BackwardInduction(n_quad=7),
    )
    v = value({"x": torch.tensor([0.5], dtype=torch.float64)}, t=0)
    assert v.item() == pytest.approx(3.0, abs=1e-12)


def test_uniform_in_solver_second_moment():
    """reward = shock^2 with Uniform[-1, 3] → V_0 = E[X^2] = 7/3."""
    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["r"] ** 2

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Uniform("r", low=-1.0, high=3.0)],
        horizon=range(0, 1),
        discount=1.0,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={},
        solver=BackwardInduction(n_quad=7),
    )
    v = value({"x": torch.tensor([0.5], dtype=torch.float64)}, t=0)
    expected = (3.0 ** 3 - (-1.0) ** 3) / (3.0 * (3.0 - (-1.0)))  # = 7/3
    assert v.item() == pytest.approx(expected, abs=1e-12)
