"""Tests for the ``k_howard`` knob on ``PolicyIteration``.

What we want to confirm:

1. **Construction** — ``k_howard < 1`` is a clean error.
2. **Equivalence at every k_howard** — the converged value function and
   policy are independent of ``k_howard`` (it only controls *how* we
   reach the fixed point, not *which* fixed point).
3. **Fewer outer iterations** — modified PI with ``k_howard = 10``
   converges in dramatically fewer outer iterations than ``k_howard = 1``
   (value iteration) on a smooth contractive Merton problem.
"""

import math

import pytest
import torch

from bellgrid import ContinuousAction, ContinuousState, Problem, solve
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import PolicyIteration
from bellgrid.solvers._common import (
    _evaluate_at_policy, bellman_step, setup_solve, terminal_value,
)


def _build_merton_infinite(beta=0.96, mu=0.04, sigma=0.15):
    def transition(state, action, shock, _t):
        return {
            "wealth": (state["wealth"] - action["consume"])
            * torch.exp(mu + sigma * shock["z"])
        }

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    return Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 200.0))],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=None,
        discount=beta,
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_k_howard_zero_raises():
    with pytest.raises(ValueError, match="k_howard >= 1"):
        PolicyIteration(k_howard=0)


def test_k_howard_one_reverts_to_value_iteration():
    """``k_howard = 1`` skips the evaluation loop entirely, so behaviour
    matches the historical value-iteration semantics. Sanity check: the
    rate still matches closed form."""
    beta = 0.96
    policy, _ = solve(
        _build_merton_infinite(beta=beta),
        state_grid={"wealth": WarpedGrid(n=64)},
        action_grid={"consume": RegularGrid(n=200)},
        solver=PolicyIteration(n_quad=7, tol=1e-6, k_howard=1),
        device="cpu",
    )
    w = torch.tensor([5.0, 20.0], dtype=torch.float64)
    rate = (policy({"wealth": w}, t=None)["consume"] / w).tolist()
    for r in rate:
        assert r == pytest.approx(1.0 - beta, abs=0.005)


# ---------------------------------------------------------------------------
# k_howard doesn't change the fixed point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k_howard", [1, 5, 25, 100])
def test_k_howard_converges_to_same_fixed_point(k_howard):
    """Every ``k_howard`` setting reaches the same fixed point of the
    Bellman operator. Compare V and policy at a few test wealths against
    the k=1 baseline; they should agree to well below the discretisation
    floor (which is ~1e-3 on this grid)."""
    problem = _build_merton_infinite()
    state_grid = {"wealth": WarpedGrid(n=64)}
    action_grid = {"consume": RegularGrid(n=200)}
    common = dict(n_quad=7, tol=1e-8, max_iters=10_000)

    policy_ref, value_ref = solve(
        problem, state_grid=state_grid, action_grid=action_grid,
        solver=PolicyIteration(k_howard=1, **common), device="cpu",
    )
    policy_k, value_k = solve(
        problem, state_grid=state_grid, action_grid=action_grid,
        solver=PolicyIteration(k_howard=k_howard, **common), device="cpu",
    )

    w = torch.tensor([2.0, 5.0, 20.0, 50.0], dtype=torch.float64)
    v_ref = value_ref({"wealth": w}, t=None)
    v_k = value_k({"wealth": w}, t=None)
    c_ref = policy_ref({"wealth": w}, t=None)["consume"]
    c_k = policy_k({"wealth": w}, t=None)["consume"]

    assert torch.allclose(v_k, v_ref, atol=1e-4), (
        f"k_howard={k_howard}: V mismatch ref={v_ref.tolist()} k={v_k.tolist()}"
    )
    assert torch.allclose(c_k, c_ref, atol=1e-4), (
        f"k_howard={k_howard}: consume mismatch ref={c_ref.tolist()} k={c_k.tolist()}"
    )


# ---------------------------------------------------------------------------
# Howard cuts outer iteration count
# ---------------------------------------------------------------------------


def _count_outer_iters(problem, k_howard, *, tol=1e-7, state_n=64, action_n=200):
    """Manually run the PolicyIteration loop and report how many outer
    iterations are needed to hit ``tol``. Bypasses ``solve`` so we can
    inspect iteration counts directly (the public API doesn't surface them)."""
    ctx = setup_solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=state_n)},
        action_grid={"consume": RegularGrid(n=action_n)},
        n_quad=7,
        device="cpu",
        dtype=torch.float64,
        chunk_size=2**20,
    )
    V = terminal_value(ctx)
    for n in range(5000):
        V_new, pol = bellman_step(ctx, V, t=None)
        for _ in range(k_howard - 1):
            V_new = _evaluate_at_policy(ctx, V_new, pol, t=None)
        delta = (V_new - V).abs().max().item()
        V = V_new
        if delta < tol:
            return n + 1
    raise AssertionError(f"k_howard={k_howard} did not converge in 5000 iters")


def test_howard_cuts_outer_iterations():
    """Default Howard (k_howard=10) should need at least 5x fewer outer
    iterations than value iteration (k_howard=1) on smooth contractive
    Merton. Empirically the ratio is closer to 9x; we leave headroom."""
    problem = _build_merton_infinite()
    n_value_iter = _count_outer_iters(problem, k_howard=1)
    n_howard = _count_outer_iters(problem, k_howard=10)
    assert n_howard * 5 <= n_value_iter, (
        f"Howard k=10 took {n_howard} outer iters vs value-iter {n_value_iter}"
        f" — expected at least 5x speedup in outer-iter count"
    )


def test_large_k_howard_approximates_true_pi():
    """k_howard ≥ 100 should converge in single-digit outer iterations on
    a well-conditioned smooth problem — the policy stabilises within a
    handful of improvements once each evaluation is run nearly to the
    fixed point of T_σ."""
    problem = _build_merton_infinite()
    n_outer = _count_outer_iters(problem, k_howard=200, tol=1e-7)
    assert n_outer <= 12, (
        f"k_howard=200 took {n_outer} outer iters — expected single digits "
        f"on this smooth concave problem"
    )
