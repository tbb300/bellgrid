import math

import pytest
import torch

from bellgrid import ContinuousAction, ContinuousState, Problem, simulate, solve
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction


# --- helpers -------------------------------------------------------------


def _deterministic_consume_problem(T=3):
    """1-D wealth, 1-D consumption, no shocks, scalar discount."""

    def transition(state, action, shock, t):
        return {"wealth": state["wealth"] - action["consume"]}

    def reward(state, action, shock, t):
        return action["consume"]

    return Problem(
        states=[ContinuousState("wealth", range=(0.0, 100.0))],
        actions=[ContinuousAction("consume", bounds=(0.0, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, T),
        discount=0.95,
    )


class _ConstantRatePolicy:
    """A policy that consumes a fixed fraction of wealth each period."""

    def __init__(self, rate: float):
        self.rate = rate

    def __call__(self, state, t):
        return {"consume": state["wealth"] * self.rate}


# --- shape and key tests -------------------------------------------------


def test_paths_have_expected_keys_and_shapes():
    problem = _deterministic_consume_problem(T=4)
    paths = simulate(
        policy=_ConstantRatePolicy(0.3),
        problem=problem,
        n=8,
        initial_state={"wealth": 10.0},
        seed=0,
    )
    assert set(paths) == {"wealth", "consume", "reward", "discounted_total"}
    assert paths["wealth"].shape == (8, 4)
    assert paths["consume"].shape == (8, 4)
    assert paths["reward"].shape == (8, 4)
    assert paths["discounted_total"].shape == (8,)


def test_initial_state_records_at_first_period():
    problem = _deterministic_consume_problem(T=3)
    paths = simulate(
        policy=_ConstantRatePolicy(0.5),
        problem=problem,
        n=5,
        initial_state={"wealth": 7.0},
        seed=0,
    )
    paths = {k: v.cpu() for k, v in paths.items()}
    assert torch.allclose(
        paths["wealth"][:, 0], torch.full((5,), 7.0, dtype=torch.float64)
    )


def test_initial_state_missing_raises():
    problem = _deterministic_consume_problem(T=3)
    with pytest.raises(ValueError, match="initial_state missing"):
        simulate(
            policy=_ConstantRatePolicy(0.5),
            problem=problem,
            n=5,
            initial_state={},
            seed=0,
        )


# --- deterministic correctness ------------------------------------------


def test_deterministic_constant_rate_matches_hand_calculation():
    """w_0 = 10, consume rate = 0.4, T = 3, discount = 0.95.
    w_1 = 6, w_2 = 3.6; consumes are 4, 2.4, 1.44.
    discounted_total = 4 + 0.95 * 2.4 + 0.95^2 * 1.44 = 7.58
    """
    problem = _deterministic_consume_problem(T=3)
    paths = simulate(
        policy=_ConstantRatePolicy(0.4),
        problem=problem,
        n=1,
        initial_state={"wealth": 10.0},
        seed=0,
    )
    paths = {k: v.cpu() for k, v in paths.items()}
    expected_wealth = torch.tensor([[10.0, 6.0, 3.6]], dtype=torch.float64)
    expected_consume = torch.tensor([[4.0, 2.4, 1.44]], dtype=torch.float64)
    expected_dtotal = 4.0 + 0.95 * 2.4 + 0.95**2 * 1.44

    assert torch.allclose(paths["wealth"], expected_wealth)
    assert torch.allclose(paths["consume"], expected_consume)
    assert torch.allclose(paths["reward"], expected_consume)
    assert paths["discounted_total"].item() == pytest.approx(expected_dtotal)


def test_seed_reproducibility():
    """Same seed -> same draws under a stochastic transition."""

    def transition(state, action, shock, t):
        return {"wealth": state["wealth"] + shock["z"]}

    def reward(state, action, shock, t):
        return action["a"]

    problem = Problem(
        states=[ContinuousState("wealth", range=(-10.0, 10.0))],
        actions=[ContinuousAction("a", bounds=(0.0, 1.0))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 5),
        discount=1.0,
    )

    class _ZeroAction:
        def __call__(self, state, t):
            return {"a": torch.zeros_like(state["wealth"])}

    p1 = simulate(
        policy=_ZeroAction(), problem=problem, n=100,
        initial_state={"wealth": 0.0}, seed=42,
    )
    p2 = simulate(
        policy=_ZeroAction(), problem=problem, n=100,
        initial_state={"wealth": 0.0}, seed=42,
    )
    assert torch.allclose(p1["wealth"], p2["wealth"])

    p3 = simulate(
        policy=_ZeroAction(), problem=problem, n=100,
        initial_state={"wealth": 0.0}, seed=43,
    )
    assert not torch.allclose(p1["wealth"], p3["wealth"])


# --- statistical: simulator agrees with solver --------------------------


def test_simulator_matches_solver_value_on_merton():
    """E[discounted total realized reward] over many paths matches V(w_0)
    for the same Merton problem."""
    import math

    beta, mu, sigma = 0.96, 0.04, 0.15
    A = (
        math.log(1.0 - beta) / (1.0 - beta)
        + (beta / (1.0 - beta) ** 2) * (math.log(beta) + mu)
    )
    B = 1.0 / (1.0 - beta)

    def transition(state, action, shock, t):
        return {
            "wealth": (state["wealth"] - action["consume"])
            * torch.exp(mu + sigma * shock["z"])
        }

    def reward(state, action, shock, t):
        return torch.log(action["consume"])

    def terminal_reward(state):
        return A + B * torch.log(state["wealth"])

    problem = Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 50.0))],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 30),
        discount=beta,
        terminal_reward=terminal_reward,
    )

    policy, value = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    w0 = 10.0
    n_paths = 20_000
    paths = simulate(
        policy=policy,
        problem=problem,
        n=n_paths,
        initial_state={"wealth": w0},
        seed=0,
    )
    paths = {k: v.cpu() for k, v in paths.items()}

    # E[sum discount^t * r_t] over the realized horizon
    # plus the terminal_reward at the post-horizon wealth
    final_w = paths["wealth"][:, -1] * torch.exp(
        torch.as_tensor(mu, dtype=torch.float64)
        + sigma * torch.randn(n_paths, generator=torch.Generator().manual_seed(1))
    )
    # Actually the discount on the terminal: it's at step T, so discount**T
    # But our simulate sums discount^i for i=0..T-1. To match V_0 we need to
    # add discount**T * terminal_reward(w_T_plus_1). We don't have w_T+1
    # cleanly; for a long horizon (T=30) the contribution is small.
    mean_dtotal = paths["discounted_total"].mean().item()
    v_solver = value({"wealth": torch.tensor([w0])}, t=0).item()

    # With T=30 and beta=0.96, beta^30 ≈ 0.294 — the residual terminal-value
    # contribution at w_T is V_T * 0.294. For typical w_T near 10, V_T ≈ A
    # + B*log(10) = -23.4, so the omitted term is about -6.9. We add it back
    # using the empirical terminal V across paths.
    final_t_v = (A + B * torch.log(paths["wealth"][:, -1])).mean().item()
    beta_T = beta ** len(problem.horizon)
    # paths["wealth"][:, -1] is the state at the LAST horizon entry, which
    # is t = horizon[-1]. The post-horizon V (terminal_reward) is applied
    # at one period later, i.e. with discount beta**T relative to t=0.
    # But the per-step reward at the last horizon entry is already counted
    # with discount beta**(T-1). So the terminal V's appropriate discount
    # is beta**T.
    mean_dtotal_full = mean_dtotal + beta_T * final_t_v

    # The per-path variance is dominated by the discounted-cumulative-log term
    # (Var[log(c_t)] grows ~ sigma^2 * t), so per-path stderr is order 1 even
    # at the same (mu, sigma). With 20k paths the mean-stderr is ~0.15.
    assert mean_dtotal_full == pytest.approx(v_solver, abs=0.4), (
        f"sim mean = {mean_dtotal_full:.4f}, solver V = {v_solver:.4f}"
    )
