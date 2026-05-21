"""American option pricing via backward induction with the
absorbing-sentinel encoding of the exercise decision.

Encoding: the state is the underlying spot price. The agent picks a binary
``DiscreteAction("exercise", n=2)``. Exercise transitions the price to a
"dead" sentinel value below zero (where the payoff is defined to be 0 and
the dynamics keep it dead); hold transitions normally under GBM. The
standard max-over-actions Bellman then reproduces the textbook American
recursion ``V_t(S) = max(payoff(S), exp(-r dt) * E[V_{t+1}(S')])``.

We validate against a high-resolution binomial-tree reference.
"""

import math

import pytest
import torch

from bellgrid import (
    ContinuousState,
    DiscreteAction,
    Problem,
    solve,
)
from bellgrid.grids import WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction


# --- binomial-tree reference --------------------------------------------


def _binomial_american_put(
    S0: float, K: float, r: float, sigma: float, T: float, n_steps: int
) -> float:
    """CRR American-put value at the root of an n_steps binomial tree."""
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    # Terminal-node values
    V = [max(K - S0 * (u ** (n_steps - i)) * (d ** i), 0.0) for i in range(n_steps + 1)]

    # Backward sweep
    for step in range(n_steps - 1, -1, -1):
        new_V = []
        for i in range(step + 1):
            S = S0 * (u ** (step - i)) * (d ** i)
            hold = disc * (p * V[i] + (1.0 - p) * V[i + 1])
            exercise = max(K - S, 0.0)
            new_V.append(max(hold, exercise))
        V = new_V
    return V[0]


def _european_put_black_scholes(
    S0: float, K: float, r: float, sigma: float, T: float
) -> float:
    """Closed-form European put."""
    from math import erf, log, sqrt
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    Nm1 = 0.5 * (1.0 - erf(d1 / math.sqrt(2.0)))   # N(-d1)
    Nm2 = 0.5 * (1.0 - erf(d2 / math.sqrt(2.0)))   # N(-d2)
    return K * math.exp(-r * T) * Nm2 - S0 * Nm1


# --- bellgrid problem ---------------------------------------------------


def _build_american_put(
    *,
    K: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 50,
    dead_sentinel: float = -0.4,
    state_range: tuple[float, float] = (-0.5, 3.0),
):
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)

    def payoff(S: torch.Tensor) -> torch.Tensor:
        # Below-zero "dead" prices yield zero payoff; otherwise standard put.
        return torch.where(
            S < 0,
            torch.zeros_like(S),
            torch.clamp(K - S, min=0.0),
        )

    def transition(state, action, shock, t):
        S = state["price"]
        next_alive = S * torch.exp(drift + diffusion * shock["z"])
        is_alive = S >= 0
        not_exercised = action["exercise"] == 0
        dead = torch.full_like(S, dead_sentinel)
        next_S = torch.where(is_alive & not_exercised, next_alive, dead)
        return {"price": next_S}

    def reward(state, action, shock, t):
        return action["exercise"].to(state["price"].dtype) * payoff(state["price"])

    def terminal_reward(state):
        return payoff(state["price"])

    return Problem(
        states=[ContinuousState("price", warp="asinh", range=state_range)],
        actions=[
            DiscreteAction("exercise", n=2, labels=("hold", "exercise")),
        ],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, n_steps),
        discount=math.exp(-r * dt),
        terminal_reward=terminal_reward,
    )


# --- tests --------------------------------------------------------------


@pytest.mark.parametrize("S0", [0.8, 1.0, 1.2])
def test_american_put_matches_binomial(S0):
    K, r, sigma, T, n_steps = 1.0, 0.05, 0.2, 1.0, 50

    problem = _build_american_put(K=K, r=r, sigma=sigma, T=T, n_steps=n_steps)
    _, value = solve(
        problem,
        state_grid={"price": WarpedGrid(n=256)},
        action_grid={},
        solver=BackwardInduction(n_quad=11),
    )

    v_bellgrid = value({"price": torch.tensor([S0])}, t=0).item()
    v_binom = _binomial_american_put(S0, K, r, sigma, T, n_steps=2000)
    assert v_bellgrid == pytest.approx(v_binom, rel=0.05, abs=0.005), (
        f"S0={S0}: bellgrid={v_bellgrid:.5f}, binomial={v_binom:.5f}"
    )


def test_american_put_exceeds_european():
    """Early exercise premium is positive for an American put."""
    K, r, sigma, T = 1.0, 0.05, 0.2, 1.0
    problem = _build_american_put(K=K, r=r, sigma=sigma, T=T, n_steps=50)
    _, value = solve(
        problem,
        state_grid={"price": WarpedGrid(n=256)},
        action_grid={},
        solver=BackwardInduction(n_quad=11),
    )

    v_amer = value({"price": torch.tensor([1.0])}, t=0).item()
    v_eur = _european_put_black_scholes(1.0, K, r, sigma, T)
    assert v_amer > v_eur, f"american={v_amer:.5f} should exceed european={v_eur:.5f}"


def test_deep_itm_put_exercises_immediately():
    """Deep ITM put: exercise now is optimal; V ≈ intrinsic K - S."""
    K = 1.0
    problem = _build_american_put(K=K, r=0.05, sigma=0.2, T=1.0, n_steps=50)
    policy, value = solve(
        problem,
        state_grid={"price": WarpedGrid(n=256)},
        action_grid={},
        solver=BackwardInduction(n_quad=11),
    )

    S_deep = torch.tensor([0.3])
    a = policy({"price": S_deep}, t=5)
    v = value({"price": S_deep}, t=5).item()

    # Deep ITM put should exercise (action == 1) and V ≈ K - S
    assert a["exercise"].item() == 1
    assert v == pytest.approx(K - S_deep.item(), abs=0.01)


def test_otm_put_holds():
    """Deep OTM put: never optimal to exercise."""
    K = 1.0
    problem = _build_american_put(K=K, r=0.05, sigma=0.2, T=1.0, n_steps=50)
    policy, _ = solve(
        problem,
        state_grid={"price": WarpedGrid(n=256)},
        action_grid={},
        solver=BackwardInduction(n_quad=11),
    )

    S_otm = torch.tensor([2.0])
    a = policy({"price": S_otm}, t=5)
    assert a["exercise"].item() == 0


# --- diagnostic table ----------------------------------------------------


def test_print_american_put_results():
    """Side-by-side: bellgrid V vs binomial reference vs European."""
    K, r, sigma, T, n_steps = 1.0, 0.05, 0.2, 1.0, 50

    problem = _build_american_put(K=K, r=r, sigma=sigma, T=T, n_steps=n_steps)
    policy, value = solve(
        problem,
        state_grid={"price": WarpedGrid(n=256)},
        action_grid={},
        solver=BackwardInduction(n_quad=11),
    )

    print()
    print(f"American put  |  K={K}, r={r}, sigma={sigma}, T={T}, n_steps={n_steps}")
    print()
    print(f"{'S0':>6} {'V bellgrid':>12} {'V binomial':>12} {'V european':>12}"
          f" {'EE premium':>11} {'optimal a':>10}")
    print("-" * 70)

    for S0 in (0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4):
        S0_t = torch.tensor([S0])
        v_bg = value({"price": S0_t}, t=0).item()
        v_bin = _binomial_american_put(S0, K, r, sigma, T, n_steps=2000)
        v_eur = _european_put_black_scholes(S0, K, r, sigma, T)
        ee = v_bg - v_eur
        a = policy({"price": S0_t}, t=0)["exercise"].item()
        print(f"{S0:>6.2f} {v_bg:>12.5f} {v_bin:>12.5f} {v_eur:>12.5f}"
              f" {ee:>+11.5f} {a:>10d}")
    print()
