"""Tests for the Categorical shock."""

import pytest
import torch

from bellgrid import ContinuousState, DiscreteAction, Problem, solve
from bellgrid.grids import RegularGrid
from bellgrid.shocks import Categorical, Normal
from bellgrid.solvers import BackwardInduction


# --- construction + validation -------------------------------------------


def test_categorical_construction():
    cat = Categorical("demand", values=[1.0, 2.0, 5.0], probabilities=[0.2, 0.5, 0.3])
    assert cat.name == "demand"
    assert cat.K == 3
    assert cat.values.tolist() == [1.0, 2.0, 5.0]
    assert cat.probabilities.tolist() == pytest.approx([0.2, 0.5, 0.3])


def test_categorical_empty_name_raises():
    with pytest.raises(ValueError, match="requires a name"):
        Categorical("", values=[1.0], probabilities=[1.0])


def test_categorical_missing_values_raises():
    with pytest.raises(ValueError, match="requires values"):
        Categorical("x", probabilities=[1.0])


def test_categorical_missing_probabilities_raises():
    with pytest.raises(ValueError, match="requires probabilities"):
        Categorical("x", values=[1.0])


def test_categorical_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        Categorical("x", values=[1.0, 2.0], probabilities=[1.0])


def test_categorical_negative_probability_raises():
    with pytest.raises(ValueError, match="non-negative"):
        Categorical("x", values=[1.0, 2.0], probabilities=[1.5, -0.5])


def test_categorical_probabilities_dont_sum_to_one_raises():
    with pytest.raises(ValueError, match="sum to 1"):
        Categorical("x", values=[1.0, 2.0], probabilities=[0.3, 0.3])


def test_categorical_multidim_values_raises():
    with pytest.raises(ValueError, match="must be 1-D"):
        Categorical("x", values=[[1.0], [2.0]], probabilities=[0.5, 0.5])


def test_categorical_empty_raises():
    with pytest.raises(ValueError, match="at least one value"):
        Categorical("x", values=[], probabilities=[])


def test_categorical_is_frozen():
    cat = Categorical("x", values=[1.0, 2.0], probabilities=[0.5, 0.5])
    with pytest.raises(AttributeError):
        cat.name = "y"


# --- quadrature ----------------------------------------------------------


def test_categorical_quadrature_exact():
    cat = Categorical("x", values=[1.0, 2.0, 5.0], probabilities=[0.2, 0.5, 0.3])
    nodes, weights = cat.nodes_and_weights(7)
    assert nodes.tolist() == [1.0, 2.0, 5.0]
    assert weights.tolist() == pytest.approx([0.2, 0.5, 0.3])
    assert weights.sum().item() == pytest.approx(1.0)


def test_categorical_n_quad_ignored():
    """n_quad has no effect — quadrature is always exact."""
    cat = Categorical("x", values=[1.0, 2.0], probabilities=[0.7, 0.3])
    for n_quad in (1, 3, 7, 11):
        nodes, weights = cat.nodes_and_weights(n_quad)
        assert nodes.numel() == 2
        assert weights.numel() == 2
        assert nodes.tolist() == [1.0, 2.0]


def test_categorical_first_moment():
    """E[X] = Σ p_i v_i exactly."""
    cat = Categorical("x", values=[1.0, 2.0, 5.0], probabilities=[0.2, 0.5, 0.3])
    nodes, weights = cat.nodes_and_weights(1)
    em = (weights * nodes).sum().item()
    expected = 0.2 * 1.0 + 0.5 * 2.0 + 0.3 * 5.0  # 2.7
    assert em == pytest.approx(expected, abs=1e-14)


def test_categorical_second_moment():
    """E[X^2] = Σ p_i v_i^2 exactly."""
    cat = Categorical("x", values=[1.0, 2.0, 5.0], probabilities=[0.2, 0.5, 0.3])
    nodes, weights = cat.nodes_and_weights(1)
    em2 = (weights * nodes ** 2).sum().item()
    expected = 0.2 * 1.0 + 0.5 * 4.0 + 0.3 * 25.0  # 9.7
    assert em2 == pytest.approx(expected, abs=1e-14)


# --- sampling ------------------------------------------------------------


def test_categorical_sample_fractions_match_probabilities():
    cat = Categorical("x", values=[0.0, 1.0, 2.0], probabilities=[0.5, 0.3, 0.2])
    out = cat.sample(50_000, generator=torch.Generator().manual_seed(0))
    f0 = (out == 0.0).float().mean().item()
    f1 = (out == 1.0).float().mean().item()
    f2 = (out == 2.0).float().mean().item()
    assert abs(f0 - 0.5) < 0.01
    assert abs(f1 - 0.3) < 0.01
    assert abs(f2 - 0.2) < 0.01


def test_categorical_sample_only_yields_declared_values():
    cat = Categorical("x", values=[1.5, 7.5], probabilities=[0.6, 0.4])
    out = cat.sample(10_000, generator=torch.Generator().manual_seed(0))
    unique = torch.unique(out).tolist()
    assert sorted(unique) == [1.5, 7.5]


# --- solver integration --------------------------------------------------


def test_categorical_in_solver_recovers_expected_reward():
    """Bellman one-step with reward = shock['payout'] under
    Categorical([0, 10], [0.5, 0.5]) should give V_0 = 5."""

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["payout"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Categorical("payout", values=[0.0, 10.0], probabilities=[0.5, 0.5])],
        horizon=range(0, 1),
        discount=1.0,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={},
        solver=BackwardInduction(n_quad=1),
    )
    v = value({"x": torch.tensor([0.5], dtype=torch.float64)}, t=0)
    assert v.item() == pytest.approx(5.0, abs=1e-12)


def test_categorical_in_solver_with_normal_multishock():
    """Combined Normal + Categorical: V_0 = E[Z^2] + E[payout] should
    equal sigma^2 + Σ p_i v_i."""

    sigma_z = 1.0
    cat_values = [0.0, 4.0]
    cat_probs = [0.7, 0.3]

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["z"] ** 2 + shock["payout"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[
            Normal("z", sigma=sigma_z),
            Categorical("payout", values=cat_values, probabilities=cat_probs),
        ],
        horizon=range(0, 1),
        discount=1.0,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={},
        solver=BackwardInduction(n_quad=5),
    )
    v = value({"x": torch.tensor([0.5], dtype=torch.float64)}, t=0)
    expected = sigma_z ** 2 + sum(p * v for p, v in zip(cat_probs, cat_values))
    assert v.item() == pytest.approx(expected, abs=1e-12)
