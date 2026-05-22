"""Tests with 3 or more MarkovChains in a single problem.

Existing multi-MC tests use exactly 2 chains. This file stress-tests
the indexing, view_shape positioning, and reverse-contraction loop with
more chains.
"""

import numpy as np
import pytest
import torch

from bellgrid import (
    ContinuousState,
    DiscreteAction,
    MarkovChain,
    Problem,
    simulate,
    solve,
)
from bellgrid.grids import RegularGrid
from bellgrid.solvers import BackwardInduction


def test_three_independent_markov_chains_v_factorises():
    """Three independent MarkovChains with separable terminal reward give
    a factorisable V: V(r1, r2, r3) = E[f1(r1')] · E[f2(r2')] · E[f3(r3')]
    after one step. Verifies the reverse-order contraction handles
    arbitrary chain counts."""
    # Three 2-state chains with different transition matrices
    P1 = np.array([[0.8, 0.2], [0.3, 0.7]])
    P2 = np.array([[0.5, 0.5], [0.4, 0.6]])
    P3 = np.array([[0.9, 0.1], [0.6, 0.4]])

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    def terminal(state):
        # Separable: f(r1, r2, r3) = (1+r1) · (1+r2) · (1+r3)
        r1 = state["r1"].to(torch.float64)
        r2 = state["r2"].to(torch.float64)
        r3 = state["r3"].to(torch.float64)
        return (1.0 + r1) * (1.0 + r2) * (1.0 + r3)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("r1", matrix=P1),
            MarkovChain("r2", matrix=P2),
            MarkovChain("r3", matrix=P3),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[], horizon=range(0, 1), discount=1.0,
        terminal_reward=terminal,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=4)}, action_grid={},
        solver=BackwardInduction(n_quad=1),
    )

    # Under independence, E[f1·f2·f3] = E[f1] · E[f2] · E[f3]
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                # E[1 + r1' | r1=i] = 1 + P1[i, 1] (probability of being in state 1)
                e1 = 1.0 + P1[i, 1]
                e2 = 1.0 + P2[j, 1]
                e3 = 1.0 + P3[k, 1]
                expected = e1 * e2 * e3
                v = value({
                    "x": torch.tensor([0.5], dtype=torch.float64),
                    "r1": torch.tensor([i], dtype=torch.long),
                    "r2": torch.tensor([j], dtype=torch.long),
                    "r3": torch.tensor([k], dtype=torch.long),
                }, t=0).item()
                assert v == pytest.approx(expected, abs=1e-10), (
                    f"3-MC V at ({i},{j},{k}) = {v}, expected {expected}"
                )


def test_four_markov_chains_match_kronecker_product():
    """A problem with 4 MarkovChains should give bit-for-bit identical V
    to a problem with a single MarkovChain whose matrix is the Kronecker
    product of the four. The same equivalence as the existing two-chain
    test, but with four chains to stress the indexing."""
    P1 = np.array([[0.7, 0.3], [0.4, 0.6]])
    P2 = np.array([[0.8, 0.2], [0.5, 0.5]])
    P3 = np.array([[0.6, 0.4], [0.3, 0.7]])
    P4 = np.array([[0.9, 0.1], [0.2, 0.8]])
    P_joint = np.kron(np.kron(np.kron(P1, P2), P3), P4)  # 16x16
    assert P_joint.shape == (16, 16)
    np.testing.assert_allclose(P_joint.sum(axis=1), 1.0)

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    def transition_four(state, _a, _sh, _t):
        return {"x": state["x"]}

    # Terminal: r1·8 + r2·4 + r3·2 + r4 — i.e. the integer encoding of
    # (r1, r2, r3, r4) as a 4-bit binary number. This makes the terminal
    # match exactly the joint-state index in the product chain.
    def terminal_four(state):
        r1 = state["r1"].to(torch.float64)
        r2 = state["r2"].to(torch.float64)
        r3 = state["r3"].to(torch.float64)
        r4 = state["r4"].to(torch.float64)
        return r1 * 8.0 + r2 * 4.0 + r3 * 2.0 + r4

    problem_four = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("r1", matrix=P1),
            MarkovChain("r2", matrix=P2),
            MarkovChain("r3", matrix=P3),
            MarkovChain("r4", matrix=P4),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition_four, reward=reward, shocks=[],
        horizon=range(0, 2), discount=0.9,
        terminal_reward=terminal_four,
    )

    def transition_one(state, _a, _sh, _t):
        return {"x": state["x"]}

    def terminal_one(state):
        # joint = r1*8 + r2*4 + r3*2 + r4; recover and weight
        j = state["joint"].to(torch.float64)
        return j

    problem_one = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("joint", matrix=P_joint),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition_one, reward=reward, shocks=[],
        horizon=range(0, 2), discount=0.9,
        terminal_reward=terminal_one,
    )

    _, value_four = solve(problem_four, state_grid={"x": RegularGrid(n=4)},
                         action_grid={}, solver=BackwardInduction(n_quad=1))
    _, value_one = solve(problem_one, state_grid={"x": RegularGrid(n=4)},
                        action_grid={}, solver=BackwardInduction(n_quad=1))

    # Compare V at every joint state
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                for l in (0, 1):
                    joint_idx = i * 8 + j * 4 + k * 2 + l
                    v_four = value_four({
                        "x": torch.tensor([0.5], dtype=torch.float64),
                        "r1": torch.tensor([i], dtype=torch.long),
                        "r2": torch.tensor([j], dtype=torch.long),
                        "r3": torch.tensor([k], dtype=torch.long),
                        "r4": torch.tensor([l], dtype=torch.long),
                    }, t=0).item()
                    v_one = value_one({
                        "x": torch.tensor([0.5], dtype=torch.float64),
                        "joint": torch.tensor([joint_idx], dtype=torch.long),
                    }, t=0).item()
                    assert v_four == pytest.approx(v_one, abs=1e-9), (
                        f"4-MC V at {(i,j,k,l)} = {v_four}, "
                        f"joint V at {joint_idx} = {v_one}"
                    )


def test_three_markov_chains_simulate_each_evolves_independently():
    """In simulate, with 3 chains starting at category 0, each chain
    should reach both categories after enough periods (verifying each
    is actually being sampled, not getting stuck)."""
    P1 = np.array([[0.7, 0.3], [0.3, 0.7]])  # mean-reverting
    P2 = np.array([[0.6, 0.4], [0.5, 0.5]])
    P3 = np.array([[0.5, 0.5], [0.5, 0.5]])  # uniform

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("r1", matrix=P1),
            MarkovChain("r2", matrix=P2),
            MarkovChain("r3", matrix=P3),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 30), discount=0.95,
    )
    policy, _ = solve(problem,
                     state_grid={"x": RegularGrid(n=4)}, action_grid={},
                     solver=BackwardInduction(n_quad=1))

    paths = simulate(
        policy=policy, problem=problem, n=500,
        initial_state={"x": 0.5, "r1": 0, "r2": 0, "r3": 0},
        seed=0,
    )
    for name in ("r1", "r2", "r3"):
        finals = paths[name][:, -1].cpu().numpy()
        assert set(finals.tolist()) == {0, 1}, (
            f"chain {name} did not visit both states; finals = {set(finals.tolist())}"
        )
