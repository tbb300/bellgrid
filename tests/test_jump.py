"""Tests for the Jump shock and multi-shock quadrature in the solver."""

import math

import numpy as np
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
from bellgrid.shocks import Jump, Normal
from bellgrid.solvers import BackwardInduction


# --- Jump class ----------------------------------------------------------


def test_jump_construction():
    j = Jump("rare", intensity=0.05, jump_mu=-0.1, jump_sigma=0.2)
    assert j.name == "rare"
    assert j.intensity == 0.05
    assert j.jump_mu == -0.1
    assert j.jump_sigma == 0.2


def test_jump_empty_name_raises():
    with pytest.raises(ValueError, match="requires a name"):
        Jump("", intensity=0.05)


def test_jump_negative_intensity_raises():
    with pytest.raises(ValueError, match="non-negative"):
        Jump("j", intensity=-0.1)


def test_jump_non_positive_sigma_raises():
    with pytest.raises(ValueError, match="must be positive"):
        Jump("j", intensity=0.05, jump_sigma=0.0)


def test_jump_p_jump_matches_formula():
    j = Jump("j", intensity=0.05)
    assert j.p_jump == pytest.approx(1.0 - math.exp(-0.05))


def test_jump_zero_intensity_means_no_jumps():
    """intensity=0 ⇒ p_jump=0; quadrature is just the no-jump node."""
    j = Jump("j", intensity=0.0, jump_mu=0.5, jump_sigma=1.0)
    nodes_dict, weights = j.nodes_and_weights(5)
    assert weights[0].item() == pytest.approx(1.0)
    assert weights[1:].sum().item() == pytest.approx(0.0, abs=1e-15)
    assert nodes_dict["j"][0].item() == 0.0  # no-jump branch


def test_jump_weights_sum_to_one():
    j = Jump("j", intensity=0.05, jump_mu=-0.1, jump_sigma=0.2)
    for n_quad in (3, 5, 7, 11):
        _, w = j.nodes_and_weights(n_quad)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_jump_node_count_is_one_plus_n_quad():
    j = Jump("j", intensity=0.1, jump_mu=0.0, jump_sigma=1.0)
    nodes, weights = j.nodes_and_weights(7)
    assert weights.numel() == 8
    assert nodes["j"].numel() == 8


def test_jump_first_moments_via_quadrature():
    """E[log_multiplier] = p_jump * jump_mu."""
    jump_mu, jump_sigma, intensity = 0.3, 0.4, 0.1
    j = Jump("j", intensity=intensity, jump_mu=jump_mu, jump_sigma=jump_sigma)
    nodes, w = j.nodes_and_weights(11)
    em = (w * nodes["j"]).sum().item()
    expected = (1.0 - math.exp(-intensity)) * jump_mu
    assert em == pytest.approx(expected, abs=1e-12)


def test_jump_second_moment_via_quadrature():
    """E[log_multiplier^2] = p_jump * (jump_mu^2 + jump_sigma^2)."""
    jump_mu, jump_sigma, intensity = 0.2, 0.5, 0.1
    j = Jump("j", intensity=intensity, jump_mu=jump_mu, jump_sigma=jump_sigma)
    nodes, w = j.nodes_and_weights(11)
    em2 = (w * nodes["j"] ** 2).sum().item()
    p = 1.0 - math.exp(-intensity)
    expected = p * (jump_mu ** 2 + jump_sigma ** 2)
    assert em2 == pytest.approx(expected, abs=1e-12)


def test_jump_sample_mean_matches_target():
    """50k samples → empirical mean ≈ E[X] = p_jump * jump_mu."""
    jump_mu = 0.2
    j = Jump("j", intensity=0.1, jump_mu=jump_mu, jump_sigma=0.3)
    out = j.sample(50_000, generator=torch.Generator().manual_seed(0))
    em = out["j"].mean().item()
    p = j.p_jump
    expected = p * jump_mu
    # Sample std of the mixture has variance p*(mu^2 + sigma^2) - (p*mu)^2 ≈ 0.0095
    # Std of empirical mean of 50k samples ≈ sqrt(0.0095/50000) ≈ 4.4e-4
    assert abs(em - expected) < 0.005


def test_jump_sample_fraction_jumped_matches_p_jump():
    j = Jump("j", intensity=0.2, jump_mu=10.0, jump_sigma=0.1)
    # With jump_mu=10 and jump_sigma=0.1, jumped samples are clearly ≠ 0.
    out = j.sample(20_000, generator=torch.Generator().manual_seed(0))
    fraction_nonzero = (out["j"].abs() > 1.0).float().mean().item()
    assert fraction_nonzero == pytest.approx(j.p_jump, abs=0.01)


def test_jump_is_frozen():
    j = Jump("j", intensity=0.1)
    with pytest.raises(AttributeError):
        j.intensity = 0.2


# --- multi-shock quadrature in the solver --------------------------------


def test_two_shock_quadrature_node_count():
    """A Normal + Jump together should produce N_q1 × N_q2 joint nodes."""

    def transition(state, _a, shock, _t):
        return {
            "x": state["x"] + shock["z"] + shock["jmp"],
        }

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    problem = Problem(
        states=[ContinuousState("x", range=(-3.0, 3.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition,
        reward=reward,
        shocks=[
            Normal("z", sigma=1.0),
            Jump("jmp", intensity=0.05, jump_mu=0.0, jump_sigma=0.2),
        ],
        horizon=range(0, 2),
        discount=0.95,
    )
    # Smoke test: solve runs end-to-end with two shocks.
    policy, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=32)},
        action_grid={},
        solver=BackwardInduction(n_quad=5),  # Normal: 5 nodes, Jump: 1+5=6 nodes → 30 joint
    )
    out = value({"x": torch.tensor([0.5], dtype=torch.float64)}, t=0)
    assert torch.isfinite(out).all()


def test_two_shock_quadrature_marginalizes_correctly():
    """A reward depending only on shock["z"] should give the same value
    whether Jump is also a shock or not (the Jump axis just marginalizes)."""
    sigma_z = 1.0

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward_z_only(_s, _a, shock, _t):
        return shock["z"] ** 2

    # One shock: just Normal
    p1 = Problem(
        states=[ContinuousState("x", range=(-1.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition,
        reward=reward_z_only,
        shocks=[Normal("z", sigma=sigma_z)],
        horizon=range(0, 1),
        discount=0.95,
    )
    # Two shocks: Normal + Jump
    p2 = Problem(
        states=[ContinuousState("x", range=(-1.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition,
        reward=reward_z_only,
        shocks=[
            Normal("z", sigma=sigma_z),
            Jump("jmp", intensity=0.05),
        ],
        horizon=range(0, 1),
        discount=0.95,
    )
    _, v1 = solve(p1, state_grid={"x": RegularGrid(n=8)}, action_grid={},
                  solver=BackwardInduction(n_quad=5))
    _, v2 = solve(p2, state_grid={"x": RegularGrid(n=8)}, action_grid={},
                  solver=BackwardInduction(n_quad=5))

    # V_0(x) = E[z^2] = sigma^2 = 1, identical in both
    q = {"x": torch.tensor([0.0], dtype=torch.float64)}
    assert v1(q, t=0).item() == pytest.approx(v2(q, t=0).item(), abs=1e-12)
    assert v1(q, t=0).item() == pytest.approx(sigma_z ** 2, abs=1e-12)


def test_two_shock_jump_contribution():
    """A reward = shock["jmp"]^2 should yield E[jmp^2] = p * (mu^2 + sigma^2)."""
    intensity, mu_j, sigma_j = 0.1, 0.2, 0.5

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, shock, _t):
        return shock["jmp"] ** 2

    problem = Problem(
        states=[ContinuousState("x", range=(-1.0, 1.0))],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition,
        reward=reward,
        shocks=[
            Normal("z", sigma=1.0),
            Jump("jmp", intensity=intensity, jump_mu=mu_j, jump_sigma=sigma_j),
        ],
        horizon=range(0, 1),
        discount=0.95,
    )
    _, value = solve(problem,
                     state_grid={"x": RegularGrid(n=8)}, action_grid={},
                     solver=BackwardInduction(n_quad=7))
    q = {"x": torch.tensor([0.0], dtype=torch.float64)}
    p = 1.0 - math.exp(-intensity)
    expected = p * (mu_j ** 2 + sigma_j ** 2)
    assert value(q, t=0).item() == pytest.approx(expected, abs=1e-12)
