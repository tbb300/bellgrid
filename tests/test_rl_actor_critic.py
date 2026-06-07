"""Tests for the model-based actor-critic (neural) solver.

The headline test is the *correctness-checking* story that motivated the solver:
solve a problem small enough for an exact reference and confirm the neural
solution matches it. We use a 1-D linear-quadratic-Gaussian problem — the value
is quadratic and the policy linear, so a smooth net approximates both well, and
the scalar Riccati recursion gives the ground truth. (Merton's log-utility value
has a singularity at w→0 that nets approximate poorly near the boundary; it
tracks the grid in the bulk but is not a good *tight* unit-test target.)

The rest pin the scope guards (what v1 deliberately defers to the grid solver)
and the shared `(policy, value)` callable interface.
"""

import pytest
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    MarkovChain,
    Problem,
    solve,
)
from bellgrid.grids import RegularGrid
from bellgrid.shocks import Normal
from bellgrid.rl import ActorCritic
from bellgrid.solvers import BackwardInduction


# --- 1-D LQG: x' = a x + b u + c w, r = -(x^2 + R u^2) ------------------

A, B, C, R, GAMMA, T = 0.9, 0.5, 0.1, 0.1, 0.95, 4


def _scalar_riccati():
    P = [0.0] * (T + 1)
    K = [0.0] * T
    c = [0.0] * (T + 1)
    P[T] = 1.0
    for t in range(T - 1, -1, -1):
        Rg = R + GAMMA * B * B * P[t + 1]
        K[t] = GAMMA * B * P[t + 1] * A / Rg
        P[t] = 1.0 + GAMMA * A * A * P[t + 1] - GAMMA * A * P[t + 1] * B * K[t]
        c[t] = GAMMA * (C * C * P[t + 1] + c[t + 1])
    return P, K, c


def _lq_problem():
    def transition(s, act, sh, t):
        return {"x": A * s["x"] + B * act["u"] + C * sh["w"]}

    def reward(s, act, sh, t):
        return -(s["x"] ** 2 + R * act["u"] ** 2)

    return Problem(
        states=[ContinuousState("x", range=(-4.0, 4.0))],
        actions=[ContinuousAction("u", bounds=(-4.0, 4.0))],
        transition=transition, reward=reward, shocks=[Normal("w", sigma=1.0)],
        horizon=range(0, T), discount=GAMMA,
        terminal_reward=lambda s: -(s["x"] ** 2),
    )


def test_lq_matches_closed_form():
    """Neural V and π match the Riccati closed form on a smooth LQG problem."""
    problem = _lq_problem()
    P, K, c = _scalar_riccati()

    policy, value = solve(
        problem,
        solver=ActorCritic(
            n_quad=5, hidden=(64, 64), state_samples=768, steps=150,
            lr=3e-3, n_global=6, n_local=6, seed=0,
        ),
        device="cpu",
    )

    # Fit quality proxy is populated and small (value units).
    assert set(value.residual_by_t) == set(range(T))
    assert max(value.residual_by_t.values()) < 1.0

    xs = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float64)  # interior
    for t in (0, T - 1):
        vn = value({"x": xs}, t)
        un = policy({"x": xs}, t)["u"]
        v_closed = -(P[t] * xs ** 2 + c[t])
        u_closed = -(K[t] * xs)
        # Lenient but meaningful: the method lands ~0.3 here, tol guards seeds.
        assert (vn - v_closed).abs().max().item() < 1.5, (t, vn, v_closed)
        assert (un - u_closed).abs().max().item() < 1.0, (t, un, u_closed)
        # Policy slope (the economically meaningful quantity) within ~25%.
        k_neural = float(-(un / xs).mean())
        assert k_neural == pytest.approx(K[t], rel=0.25), (t, k_neural, K[t])


def test_value_policy_interface_and_device():
    """Returns the same callable interface as the grid solver; honours the
    query device and returns one action entry per declared action."""
    problem = _lq_problem()
    policy, value = solve(
        problem,
        solver=ActorCritic(steps=10, state_samples=128, n_quad=3, seed=1),
        device="cpu",
    )
    q = {"x": torch.tensor([0.0, 1.0], dtype=torch.float64)}
    v = value(q, 0)
    a = policy(q, 0)
    assert v.shape == (2,)
    assert set(a) == {"u"}
    assert a["u"].shape == (2,)


# --- scope guards: what v1 defers to the grid solver --------------------


def _minimal(states, actions, horizon=range(0, 2)):
    return Problem(
        states=states, actions=actions,
        transition=lambda s, a, sh, t: {n: s[n] for n in
                                        [x.name for x in states
                                         if not isinstance(x, MarkovChain)]},
        reward=lambda s, a, sh, t: torch.zeros(()),
        shocks=[], horizon=horizon, discount=0.95,
    )


def test_infinite_horizon_rejected():
    problem = _minimal([ContinuousState("x", range=(0.0, 1.0))],
                       [ContinuousAction("u", bounds=(0.0, 1.0))], horizon=None)
    with pytest.raises(NotImplementedError, match="finite-horizon"):
        solve(problem, solver=ActorCritic(), device="cpu")


def test_markov_chain_rejected():
    P = [[0.8, 0.2], [0.3, 0.7]]
    problem = _minimal(
        [ContinuousState("x", range=(0.0, 1.0)), MarkovChain("r", matrix=P)],
        [ContinuousAction("u", bounds=(0.0, 1.0))],
    )
    with pytest.raises(NotImplementedError, match="MarkovChain"):
        solve(problem, solver=ActorCritic(), device="cpu")


def test_mixed_actions_rejected():
    """v1 takes all-continuous or all-discrete actions, not a mix."""
    problem = _minimal(
        [ContinuousState("x", range=(0.0, 1.0))],
        [ContinuousAction("u", bounds=(0.0, 1.0)), DiscreteAction("d", n=3)],
    )
    with pytest.raises(NotImplementedError, match="all-continuous or all-discrete"):
        solve(problem, solver=ActorCritic(), device="cpu")


def test_discrete_action_matches_grid():
    """Discrete-action ActorCritic (fitted value iteration, argmax-Q policy) should
    reproduce the grid oracle on a small lost-sales inventory problem: value within
    a few percent and the discrete order policy matching almost everywhere."""
    torch.manual_seed(0)
    beta, mu_d, sigma_d = 0.97, 3.0, 1.2
    price, hold, short_pen, order_cost, maxq = 2.0, 0.1, 1.5, 0.5, 5

    def demand(shock):
        return torch.clamp(mu_d + sigma_d * shock["z"], min=0.0)

    def transition(state, action, shock, _t):
        q = action["order"].to(state["inv"].dtype)
        return {"inv": torch.clamp(state["inv"] + q - demand(shock), min=0.0)}

    def reward(state, action, shock, _t):
        q = action["order"].to(state["inv"].dtype)
        d = demand(shock)
        avail = state["inv"] + q
        sold = torch.minimum(avail, d)
        leftover = torch.clamp(avail - d, min=0.0)
        short = torch.clamp(d - avail, min=0.0)
        return price * sold - hold * leftover - short_pen * short - order_cost * q

    problem = Problem(
        states=[ContinuousState("inv", range=(0.0, 20.0))],
        actions=[DiscreteAction("order", n=maxq + 1)],
        transition=transition, reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 4), discount=beta,
    )

    pol_g, val_g = solve(
        problem, state_grid={"inv": RegularGrid(n=400)}, action_grid={},
        solver=BackwardInduction(n_quad=9), device="cpu",
    )
    pol_ac, val_ac = solve(
        problem,
        solver=ActorCritic(n_quad=9, twin_critic=True, steps=250,
                           state_samples=2048, hidden=(128, 128), seed=0),
        device="cpu",
    )

    xs = torch.linspace(0.5, 18.0, 40)
    for t in range(4):
        state = {"inv": xs}
        vg, va = val_g(state, t), val_ac(state, t)
        relmae = ((vg - va).abs().mean() / vg.abs().mean()).item()
        match = (pol_g(state, t)["order"].long()
                 == pol_ac(state, t)["order"].long()).float().mean().item()
        # Value is the tight certification; the discrete policy can differ from the
        # grid on the odd indifference-boundary cell (both actions near-optimal there).
        assert relmae < 0.03, f"t={t} value relMAE {relmae:.3%}"
        assert match >= 0.85, f"t={t} policy match {match:.1%}"


def test_discrete_action_ergodic_runs():
    """The discrete-action ergodic path (path sampling from `init_state`) should run
    end-to-end and price the start near the grid oracle — exercises
    `_collect_visited_discrete` and the ergodic refinement branch."""
    torch.manual_seed(0)
    beta, mu_d, sigma_d, maxq = 0.97, 3.0, 1.0, 4

    def demand(shock):
        return torch.clamp(mu_d + sigma_d * shock["z"], min=0.0)

    def transition(state, action, shock, _t):
        q = action["order"].to(state["inv"].dtype)
        return {"inv": torch.clamp(state["inv"] + q - demand(shock), min=0.0)}

    def reward(state, action, shock, _t):
        q = action["order"].to(state["inv"].dtype)
        d = demand(shock)
        avail = state["inv"] + q
        return (2.0 * torch.minimum(avail, d) - 0.1 * torch.clamp(avail - d, min=0.0)
                - torch.clamp(d - avail, min=0.0) - 0.5 * q)

    problem = Problem(
        states=[ContinuousState("inv", range=(0.0, 20.0))],
        actions=[DiscreteAction("order", n=maxq + 1)],
        transition=transition, reward=reward, shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 3), discount=beta,
    )
    pol_g, val_g = solve(
        problem, state_grid={"inv": RegularGrid(n=300)}, action_grid={},
        solver=BackwardInduction(n_quad=9), device="cpu",
    )
    pol_a, val_a = solve(
        problem,
        solver=ActorCritic(n_quad=9, twin_critic=True, steps=150, state_samples=1536,
                           hidden=(128, 128), seed=0, ergodic=True,
                           init_state={"inv": 8.0}, ergodic_passes=1,
                           ergodic_sim_paths=1024),
        device="cpu",
    )
    st = {"inv": torch.tensor([8.0])}
    vg, va = float(val_g(st, 0)), float(val_a(st, 0))
    assert abs(vg - va) / abs(vg) < 0.06, f"ergodic value at start: grid {vg:.3f} vs AC {va:.3f}"


def test_grid_solver_still_requires_grids():
    """The solve() signature change keeps grids optional, but grid solvers must
    still be given both."""
    problem = _lq_problem()
    with pytest.raises(ValueError, match="state_grid and action_grid"):
        solve(problem, solver=BackwardInduction(), device="cpu")
