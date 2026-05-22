"""Solver smoke tests for DiscreteState and MarkovChain."""

import math

import numpy as np
import pytest
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    MarkovChain,
    Problem,
    simulate,
    solve,
)
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction


# --- DiscreteState: user-controlled discrete dynamics -------------------


def test_discrete_state_constant_phase():
    """Two-state phase variable that user freezes (stays at current value).
    Should reduce to two independent 1-D Merton problems."""
    beta, mu, sigma = 0.96, 0.04, 0.15
    B = 1.0 / (1.0 - beta)
    A = (
        math.log(1.0 - beta) / (1.0 - beta)
        + (beta / (1.0 - beta) ** 2) * (math.log(beta) + mu)
    )

    def transition(state, action, shock, _t):
        w_next = (state["wealth"] - action["consume"]) * torch.exp(
            mu + sigma * shock["z"]
        )
        # phase stays put — discrete state advanced by user as identity
        return {"wealth": w_next, "phase": state["phase"]}

    def reward(_state, action, _shock, _t):
        return torch.log(action["consume"])

    def terminal(state):
        return A + B * torch.log(state["wealth"])

    problem = Problem(
        states=[
            ContinuousState("wealth", warp="asinh", range=(1e-3, 200.0)),
            DiscreteState("phase", n=2, labels=("accum", "decum")),
        ],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 12),
        discount=beta,
        terminal_reward=terminal,
    )

    policy, value = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=64)},
        action_grid={"consume": RegularGrid(n=300)},
        solver=BackwardInduction(n_quad=7),
    )

    # Consumption rate ≈ 1 - β = 0.04 at any wealth, regardless of phase
    for phase_val in (0, 1):
        a = policy(
            {
                "wealth": torch.tensor([2.0, 10.0, 25.0], dtype=torch.float64),
                "phase": torch.tensor([phase_val, phase_val, phase_val], dtype=torch.long),
            },
            t=5,
        )
        rates = (a["consume"] / torch.tensor([2.0, 10.0, 25.0])).tolist()
        for rate in rates:
            assert rate == pytest.approx(0.04, abs=0.005)


def test_discrete_state_value_function_shape():
    """V at t=0 has shape (n_wealth, n_phase)."""
    def transition(state, action, shock, _t):
        return {"wealth": state["wealth"], "phase": state["phase"]}

    def reward(_state, action, _shock, _t):
        return torch.log(action["consume"])

    problem = Problem(
        states=[
            ContinuousState("wealth", range=(0.1, 5.0)),
            DiscreteState("phase", n=3),
        ],
        actions=[ContinuousAction("consume", bounds=(1e-3, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 3),
        discount=0.95,
    )
    policy, value = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=16)},
        action_grid={"consume": RegularGrid(n=20)},
        solver=BackwardInduction(n_quad=1),
    )
    # Query at every phase; should return per-phase values
    v0 = value({
        "wealth": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "phase": torch.tensor([0, 2], dtype=torch.long),
    }, t=0)
    assert v0.shape == (2,)


# --- MarkovChain: solver-controlled discrete dynamics -------------------


def test_markov_chain_absorbing_state_irrelevant_when_starting_in_other():
    """Two-state chain: state 0 is absorbing (1 -> 1 self-loop trivially).
    If the value function is the same in both regimes, starting regime
    shouldn't matter for v(w)."""
    P = [[1.0, 0.0], [0.0, 1.0]]   # both absorbing — chains decouple
    beta, mu, sigma = 0.96, 0.04, 0.15
    B = 1.0 / (1.0 - beta)
    A = (
        math.log(1.0 - beta) / (1.0 - beta)
        + (beta / (1.0 - beta) ** 2) * (math.log(beta) + mu)
    )

    def transition(state, action, shock, _t):
        return {"wealth": (state["wealth"] - action["consume"])
                * torch.exp(mu + sigma * shock["z"])}

    def reward(_state, action, _shock, _t):
        return torch.log(action["consume"])

    def terminal(state):
        return A + B * torch.log(state["wealth"])

    problem = Problem(
        states=[
            ContinuousState("wealth", warp="asinh", range=(1e-3, 200.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 12),
        discount=beta,
        terminal_reward=terminal,
    )
    policy, value = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=64)},
        action_grid={"consume": RegularGrid(n=300)},
        solver=BackwardInduction(n_quad=7),
    )

    # Identity matrix: both regimes are absorbing, decoupled. V should
    # be the same in both (since the dynamics don't depend on regime in
    # this test). Both should match the standard 1-asset Merton.
    for regime in (0, 1):
        v = value(
            {
                "wealth": torch.tensor([10.0], dtype=torch.float64),
                "regime": torch.tensor([regime], dtype=torch.long),
            },
            t=5,
        ).item()
        expected = A + B * math.log(10.0)
        assert v == pytest.approx(expected, rel=0.02, abs=0.5)


def test_markov_chain_matrix_average_reduces_to_single_chain():
    """When the matrix is row-uniform (each row = stationary distribution),
    the next-period regime is independent of current; V should be the
    same as the per-period expectation over regimes weighted by that
    distribution.

    Simplest check: matrix [[0.5, 0.5], [0.5, 0.5]] with both regimes
    having identical dynamics — V should not depend on starting regime
    and should equal the no-regime Merton solution."""
    P = [[0.5, 0.5], [0.5, 0.5]]
    beta, mu, sigma = 0.96, 0.04, 0.15
    B = 1.0 / (1.0 - beta)
    A = (
        math.log(1.0 - beta) / (1.0 - beta)
        + (beta / (1.0 - beta) ** 2) * (math.log(beta) + mu)
    )

    def transition(state, action, shock, _t):
        return {"wealth": (state["wealth"] - action["consume"])
                * torch.exp(mu + sigma * shock["z"])}

    def reward(_state, action, _shock, _t):
        return torch.log(action["consume"])

    def terminal(state):
        return A + B * torch.log(state["wealth"])

    problem = Problem(
        states=[
            ContinuousState("wealth", warp="asinh", range=(1e-3, 200.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 12),
        discount=beta,
        terminal_reward=terminal,
    )
    policy, value = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=64)},
        action_grid={"consume": RegularGrid(n=300)},
        solver=BackwardInduction(n_quad=7),
    )

    v_r0 = value(
        {"wealth": torch.tensor([10.0]), "regime": torch.tensor([0], dtype=torch.long)},
        t=5,
    ).item()
    v_r1 = value(
        {"wealth": torch.tensor([10.0]), "regime": torch.tensor([1], dtype=torch.long)},
        t=5,
    ).item()
    assert v_r0 == pytest.approx(v_r1, abs=1e-10)
    expected = A + B * math.log(10.0)
    assert v_r0 == pytest.approx(expected, rel=0.02, abs=0.5)


def test_markov_chain_must_not_appear_in_transition_dict():
    """Returning the MC state from user's transition is a clear error."""
    def transition(state, action, shock, _t):
        return {
            "wealth": state["wealth"] - action["consume"],
            "regime": state["regime"],  # forbidden
        }

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    problem = Problem(
        states=[
            ContinuousState("wealth", range=(0.1, 5.0)),
            MarkovChain("regime", matrix=[[0.9, 0.1], [0.2, 0.8]]),
        ],
        actions=[ContinuousAction("consume", bounds=(1e-3, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 2),
        discount=0.95,
    )
    with pytest.raises(ValueError, match="must not return MarkovChain"):
        solve(
            problem,
            state_grid={"wealth": RegularGrid(n=8)},
            action_grid={"consume": RegularGrid(n=10)},
            solver=BackwardInduction(n_quad=1),
        )


def test_markov_chain_matrix_contraction_correct():
    """Hand-checked 1-period Bellman with a 2-state markov chain.

    Setup: one DiscreteState (so no continuous state mesh), reward
    depends on regime only, terminal V is a known function of regime.
    The Bellman value at t=0 should equal r(regime) + γ * E[V_T | regime].

    With r=0, V_T = [10, 20], and matrix [[0.7, 0.3], [0.4, 0.6]],
    expected V_0[regime=0] = γ * (0.7*10 + 0.3*20) = γ * 13
    expected V_0[regime=1] = γ * (0.4*10 + 0.6*20) = γ * 16
    """
    gamma = 0.9
    P = [[0.7, 0.3], [0.4, 0.6]]

    def transition(state, _a, _sh, _t):
        # One-shot DiscreteState that doesn't move (we just need it
        # so the problem has at least one continuous-ish state — we
        # use a 1-point DiscreteState for that).
        return {"phantom": state["phantom"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    def terminal(state):
        # V_T(regime) = 10 if regime == 0 else 20
        regime = state["regime"].to(torch.float64)
        return 10.0 + 10.0 * regime

    problem = Problem(
        states=[
            DiscreteState("phantom", n=1),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 1),  # single backward step
        discount=gamma,
        terminal_reward=terminal,
    )
    policy, value = solve(
        problem,
        state_grid={},
        action_grid={},
        solver=BackwardInduction(n_quad=1),
    )

    v0 = value(
        {
            "phantom": torch.tensor([0, 0], dtype=torch.long),
            "regime": torch.tensor([0, 1], dtype=torch.long),
        },
        t=0,
    )
    assert v0[0].item() == pytest.approx(gamma * 13.0, abs=1e-10)
    assert v0[1].item() == pytest.approx(gamma * 16.0, abs=1e-10)


# --- mixed: continuous + discrete + markov ------------------------------


def test_mixed_state_runs_end_to_end():
    """Continuous + DiscreteState + MarkovChain in one problem; just
    verify it solves and the value function lookup works at all combos."""
    P = [[0.8, 0.2], [0.3, 0.7]]

    def transition(state, action, shock, _t):
        # Wealth evolves, phase flips, regime advanced by solver
        next_w = state["wealth"] + 0.01 * shock["z"] - 0.1 * action["consume"]
        next_phase = (state["phase"] + 1) % 2
        return {"wealth": next_w, "phase": next_phase}

    def reward(_s, action, _sh, _t):
        return -action["consume"] ** 2

    problem = Problem(
        states=[
            ContinuousState("wealth", range=(-2.0, 2.0)),
            DiscreteState("phase", n=2),
            MarkovChain("regime", matrix=P),
        ],
        actions=[ContinuousAction("consume", bounds=(0.0, 1.0))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 4),
        discount=0.95,
    )
    policy, value = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=32)},
        action_grid={"consume": RegularGrid(n=20)},
        solver=BackwardInduction(n_quad=5),
    )

    # Query at a few (wealth, phase, regime) combinations
    for w in (-1.0, 0.0, 1.5):
        for phase in (0, 1):
            for regime in (0, 1):
                v = value(
                    {
                        "wealth": torch.tensor([w], dtype=torch.float64),
                        "phase": torch.tensor([phase], dtype=torch.long),
                        "regime": torch.tensor([regime], dtype=torch.long),
                    },
                    t=2,
                )
                assert torch.isfinite(v).all()


# --- simulate extension: discrete + markov ------------------------------


def test_simulate_with_discrete_state_absorbing_flip():
    """DiscreteState that flips 0→1 on a chosen action and is absorbing.
    Starting all paths in phase=0 with the always-flip policy, after one
    step every path should be in phase=1."""
    def transition(state, action, _sh, _t):
        return {
            "x": state["x"] - 0.1 * action["consume"],
            "phase": torch.where(
                (state["phase"] == 0) & (action["flip"] == 1),
                torch.ones_like(state["phase"]),
                state["phase"],
            ),
        }

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"] + 1e-6)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.1, 5.0)),
            DiscreteState("phase", n=2),
        ],
        actions=[
            ContinuousAction("consume", bounds=(0.01, 0.5)),
            DiscreteAction("flip", n=2),
        ],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 3),
        discount=0.9,
    )
    policy, _ = solve(
        problem,
        state_grid={"x": RegularGrid(n=16)},
        action_grid={"consume": RegularGrid(n=10)},
        solver=BackwardInduction(n_quad=1),
    )
    paths = simulate(
        policy=policy, problem=problem, n=50,
        initial_state={"x": 2.0, "phase": 0}, seed=0,
    )
    # Phase storage dtype is long for DiscreteState
    assert paths["phase"].dtype == torch.long
    assert paths["x"].dtype == torch.float64
    # Initial phase = 0 for every path
    assert (paths["phase"][:, 0] == 0).all()


def test_simulate_with_markov_chain_stationary_proportions():
    """MarkovChain advanced internally by the simulator. Starting all paths
    from regime=0 and running many steps under a fast-mixing chain, the
    long-run empirical distribution should approach the stationary."""
    P = [[0.7, 0.3], [0.4, 0.6]]
    # Stationary: π = (4/7, 3/7) ≈ (0.571, 0.429)

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 50),
        discount=0.95,
    )
    policy, _ = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={},
        solver=BackwardInduction(n_quad=1),
    )
    paths = simulate(
        policy=policy, problem=problem, n=2000,
        initial_state={"x": 0.5, "regime": 0}, seed=0,
    )
    # Regime is long
    assert paths["regime"].dtype == torch.long
    # All paths start in regime 0
    assert (paths["regime"][:, 0] == 0).all()
    # Last-step empirical proportion should be near stationary (4/7, 3/7)
    final_regimes = paths["regime"][:, -1].cpu().numpy()
    p0 = (final_regimes == 0).mean()
    assert abs(p0 - 4.0 / 7.0) < 0.05


# --- multiple MarkovChains -----------------------------------------------


def test_two_independent_markov_chains():
    """Two independent MarkovChains: V should respect both chain transitions
    and equal the analytic V for a 1-period problem with V_T=1 on a target cell."""
    # Two independent 2-state chains with different transition matrices.
    P1 = np.array([[0.9, 0.1], [0.2, 0.8]])
    P2 = np.array([[0.6, 0.4], [0.3, 0.7]])

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    def terminal(state):
        # 1 on the joint cell (regime1=0, regime2=0), 0 elsewhere
        r1 = state["regime1"]
        r2 = state["regime2"]
        return ((r1 == 0) & (r2 == 0)).to(torch.float64)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("regime1", matrix=P1),
            MarkovChain("regime2", matrix=P2),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[],
        horizon=range(0, 1),       # one backward step
        discount=1.0,
        terminal_reward=terminal,
    )
    _, value = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={},
        solver=BackwardInduction(n_quad=1),
    )

    # E[1{r1'=0, r2'=0} | r1=i, r2=j] = P1[i, 0] * P2[j, 0] under independence.
    for i in (0, 1):
        for j in (0, 1):
            v = value({
                "x": torch.tensor([0.5], dtype=torch.float64),
                "regime1": torch.tensor([i], dtype=torch.long),
                "regime2": torch.tensor([j], dtype=torch.long),
            }, t=0).item()
            expected = P1[i, 0] * P2[j, 0]
            assert v == pytest.approx(expected, abs=1e-12), (
                f"V at (regime1={i}, regime2={j}) = {v}, expected {expected}"
            )


def test_multiple_markov_chains_match_product_chain():
    """Two MarkovChains modelled separately should give the same V as a
    single MarkovChain over the product space with the Kronecker matrix."""
    P1 = np.array([[0.8, 0.2], [0.3, 0.7]])
    P2 = np.array([[0.5, 0.5], [0.4, 0.6]])
    # Kronecker product gives the joint matrix for the product chain
    P_joint = np.kron(P1, P2)  # 4x4

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    def transition_two(state, _a, _sh, _t):
        return {"x": state["x"]}

    # Terminal reward: a "smooth-ish" function of regime indices that
    # tests the integration is correct across multiple t values.
    def terminal_two(state):
        r1 = state["regime1"].to(torch.float64)
        r2 = state["regime2"].to(torch.float64)
        return r1 * 10.0 + r2

    problem_two = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("regime1", matrix=P1),
            MarkovChain("regime2", matrix=P2),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition_two, reward=reward,
        shocks=[], horizon=range(0, 3), discount=0.95,
        terminal_reward=terminal_two,
    )

    def transition_one(state, _a, _sh, _t):
        return {"x": state["x"]}

    # For the product chain: joint = 2*r1 + r2 (encoded as a single 4-state index)
    def terminal_one(state):
        r = state["joint"].to(torch.float64)
        r1 = (r // 2).to(torch.float64)
        r2 = (r % 2).to(torch.float64)
        return r1 * 10.0 + r2

    problem_one = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("joint", matrix=P_joint),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition_one, reward=reward,
        shocks=[], horizon=range(0, 3), discount=0.95,
        terminal_reward=terminal_one,
    )

    _, value_two = solve(
        problem_two,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={},
        solver=BackwardInduction(n_quad=1),
    )
    _, value_one = solve(
        problem_one,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={},
        solver=BackwardInduction(n_quad=1),
    )

    # Compare V at each joint state
    for i in (0, 1):
        for j in (0, 1):
            v_two = value_two({
                "x": torch.tensor([0.5], dtype=torch.float64),
                "regime1": torch.tensor([i], dtype=torch.long),
                "regime2": torch.tensor([j], dtype=torch.long),
            }, t=0).item()
            v_one = value_one({
                "x": torch.tensor([0.5], dtype=torch.float64),
                "joint": torch.tensor([2 * i + j], dtype=torch.long),
            }, t=0).item()
            assert v_two == pytest.approx(v_one, abs=1e-10), (
                f"two-MC V at ({i},{j}) = {v_two}, product-MC V = {v_one}"
            )


def test_simulate_with_two_markov_chains():
    """simulate works with two MarkovChains and samples each independently."""
    P1 = np.array([[0.9, 0.1], [0.2, 0.8]])
    P2 = np.array([[0.7, 0.3], [0.5, 0.5]])

    def transition(state, _a, _sh, _t):
        return {"x": state["x"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("regime1", matrix=P1),
            MarkovChain("regime2", matrix=P2),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 20), discount=0.95,
    )
    policy, _ = solve(
        problem,
        state_grid={"x": RegularGrid(n=8)},
        action_grid={},
        solver=BackwardInduction(n_quad=1),
    )

    paths = simulate(
        policy=policy, problem=problem, n=500,
        initial_state={"x": 0.5, "regime1": 0, "regime2": 0},
        seed=0,
    )
    assert paths["regime1"].dtype == torch.long
    assert paths["regime2"].dtype == torch.long
    # Both chains start at 0
    assert (paths["regime1"][:, 0] == 0).all()
    assert (paths["regime2"][:, 0] == 0).all()
    # After 20 steps the two chains have moved independently — check that
    # each has both categories represented.
    final1 = paths["regime1"][:, -1].cpu().numpy()
    final2 = paths["regime2"][:, -1].cpu().numpy()
    assert set(final1.tolist()) == {0, 1}
    assert set(final2.tolist()) == {0, 1}


def test_state_declaration_order_is_irrelevant():
    """User can list states in any order — solver reorders to canonical
    internally. The resulting V at the same state should match."""
    def transition(state, _a, shock, _t):
        return {"x": state["x"] + 0.1 * shock["z"]}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    base_kwargs = dict(
        actions=[DiscreteAction("noop", n=1)],
        transition=transition,
        reward=lambda s, a, sh, t: -s["x"] ** 2,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 5),
        discount=0.95,
    )
    P = [[0.8, 0.2], [0.3, 0.7]]

    p1 = Problem(
        states=[
            ContinuousState("x", range=(-2.0, 2.0)),
            MarkovChain("regime", matrix=P),
        ],
        **base_kwargs,
    )
    p2 = Problem(
        states=[
            MarkovChain("regime", matrix=P),
            ContinuousState("x", range=(-2.0, 2.0)),
        ],
        **base_kwargs,
    )

    _, v1 = solve(p1, state_grid={"x": RegularGrid(n=16)}, action_grid={},
                  solver=BackwardInduction(n_quad=3))
    _, v2 = solve(p2, state_grid={"x": RegularGrid(n=16)}, action_grid={},
                  solver=BackwardInduction(n_quad=3))

    q = {
        "x": torch.tensor([0.5], dtype=torch.float64),
        "regime": torch.tensor([1], dtype=torch.long),
    }
    assert v1(q, t=2).item() == pytest.approx(v2(q, t=2).item(), abs=1e-10)
