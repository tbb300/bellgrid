"""Tests that ``simulate()`` correctly samples and accumulates rewards
for Categorical and Uniform shocks. Both were added in 0.1.0a2 and
already have solver-side tests, but the simulate path was untested.
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
from bellgrid.shocks import Categorical, Uniform
from bellgrid.solvers import BackwardInduction


# --- Categorical shock in simulate ----------------------------------------


def test_simulate_categorical_shock_mean_reward_matches_quadrature():
    """A 1-period problem with reward = shock value should give a per-
    path total whose sample mean converges to the categorical mean."""
    values = (1.0, 5.0, 10.0)
    probs = (0.3, 0.5, 0.2)
    expected_mean = sum(v * p for v, p in zip(values, probs))  # = 1*0.3 + 5*0.5 + 10*0.2 = 4.8

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["s"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Categorical("s", values=values, probabilities=probs)],
        horizon=range(0, 1), discount=1.0,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=4)}, action_grid={},
                     solver=BackwardInduction(n_quad=1))

    paths = simulate(policy=policy, problem=problem, n=10_000,
                    initial_state={"x": 0.5}, seed=0)
    sample_mean = paths["discounted_total"].mean().item()
    # With 10k samples and finite-support {1, 5, 10}, std ≈ 3.3, so SE ≈ 0.033.
    # Allow 5σ slack: 0.17.
    assert sample_mean == pytest.approx(expected_mean, abs=0.17), (
        f"sample mean = {sample_mean}, expected {expected_mean}"
    )


def test_simulate_categorical_shock_value_frequencies_match_probabilities():
    """The empirical distribution of sampled categorical shocks must
    converge to the specified probabilities."""
    values = (10.0, 20.0, 30.0, 40.0)
    probs = (0.1, 0.4, 0.3, 0.2)

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["s"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Categorical("s", values=values, probabilities=probs)],
        horizon=range(0, 1), discount=1.0,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=4)}, action_grid={},
                     solver=BackwardInduction(n_quad=1))
    paths = simulate(policy=policy, problem=problem, n=20_000,
                    initial_state={"x": 0.5}, seed=42)

    # The reward path is the realised shock; count its values
    realised = paths["reward"][:, 0].cpu().numpy()
    for v, p in zip(values, probs):
        frac = (np.isclose(realised, v)).mean()
        # 20k draws → SE of fraction = sqrt(p(1-p)/n); allow 5σ
        se = np.sqrt(p * (1 - p) / 20_000)
        assert abs(frac - p) < 5 * se, (
            f"value {v}: observed fraction {frac:.4f}, expected {p:.4f} "
            f"(diff {abs(frac - p):.4f}, 5σ tol {5*se:.4f})"
        )


# --- Uniform shock in simulate --------------------------------------------


def test_simulate_uniform_shock_mean_reward_matches_midpoint():
    """A 1-period problem with reward = shock should give a per-path
    total whose sample mean is close to (low + high) / 2."""
    low, high = 5.0, 15.0
    expected_mean = (low + high) / 2.0  # = 10

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["u"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Uniform("u", low=low, high=high)],
        horizon=range(0, 1), discount=1.0,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=4)}, action_grid={},
                     solver=BackwardInduction(n_quad=5))

    paths = simulate(policy=policy, problem=problem, n=10_000,
                    initial_state={"x": 0.5}, seed=0)
    sample_mean = paths["discounted_total"].mean().item()
    # Uniform on [5, 15]: variance = (15-5)^2 / 12 = 8.33, std ≈ 2.89, SE ≈ 0.029
    # Allow 5σ: 0.145.
    assert sample_mean == pytest.approx(expected_mean, abs=0.15)


def test_simulate_uniform_shock_bounds_respected():
    """All sampled values must lie strictly inside [low, high]."""
    low, high = -3.0, 7.0

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["u"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Uniform("u", low=low, high=high)],
        horizon=range(0, 1), discount=1.0,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=4)}, action_grid={},
                     solver=BackwardInduction(n_quad=5))
    paths = simulate(policy=policy, problem=problem, n=5_000,
                    initial_state={"x": 0.5}, seed=42)
    realised = paths["reward"][:, 0]
    assert (realised >= low).all() and (realised <= high).all()


# --- Multi-period accumulation with new shocks ----------------------------


def test_simulate_categorical_multi_period_discounted_total():
    """Multi-period simulate with a Categorical shock; per-path discounted
    total = sum_t β^t * shock_t should match expectation."""
    values = (0.0, 2.0)
    probs = (0.5, 0.5)
    beta = 0.9
    T = 10
    expected_mean = sum(beta ** t * 1.0 for t in range(T))  # 1.0 = E[shock] per period

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["s"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Categorical("s", values=values, probabilities=probs)],
        horizon=range(0, T), discount=beta,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=4)}, action_grid={},
                     solver=BackwardInduction(n_quad=1))

    paths = simulate(policy=policy, problem=problem, n=10_000,
                    initial_state={"x": 0.5}, seed=0)
    sample_mean = paths["discounted_total"].mean().item()
    # Var per period: 1.0; sum of β^(2t) ≈ 1/(1-β^2) ≈ 5.26; std ≈ 2.29; SE ≈ 0.023
    assert sample_mean == pytest.approx(expected_mean, abs=0.12), (
        f"sample mean = {sample_mean}, expected {expected_mean}"
    )
