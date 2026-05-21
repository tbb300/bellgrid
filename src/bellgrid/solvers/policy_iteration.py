"""`PolicyIteration` solver — infinite-horizon stationary problems.

Iterate the Bellman operator from an initial guess (``terminal_reward``
if supplied, else zeros) until ``||V_new - V||_∞ < tol``, or until
``max_iters`` is hit. The Bellman operator is a contraction with factor
``γ`` (the discount), so convergence is geometric: ``error_n ≈ γ^n *
error_0``. For ``γ=0.96`` and ``tol=1e-6``, expect ~300 iterations.

The name is "PolicyIteration" to match the documented API even though
the algorithm under the hood is value iteration (i.e. iterating the
Bellman operator directly). True alternating policy-iteration would
also work and might converge faster on some problems, but value
iteration is simpler and reuses the same per-step machinery as
``BackwardInduction``.

Requires ``problem.horizon is None``. The user's ``transition`` and
``reward`` callables receive ``t=None`` — they should not depend on time.
"""

from dataclasses import dataclass

import torch

from ..problem import Problem
from ._common import _Policy, _Value, bellman_step, setup_solve, terminal_value


@dataclass(frozen=True)
class PolicyIteration:
    """Iterate the Bellman operator to convergence (value iteration).

    Attributes
    ----------
    n_quad : int
        Number of Gauss-Hermite quadrature nodes per shock dimension.
    tol : float
        Convergence threshold on ``||V_new - V||_∞``.
    max_iters : int
        Safety cap on the number of Bellman iterations. Exceeding this
        is a hard error (better to fail loudly than silently return a
        non-converged answer).
    """

    n_quad: int = 7
    tol: float = 1e-6
    max_iters: int = 10_000


def _policy_iteration(
    problem: Problem,
    state_grid: dict,
    action_grid: dict,
    solver: PolicyIteration,
    *,
    device,
    dtype: torch.dtype,
    chunk_size: int,
) -> tuple[_Policy, _Value]:
    if problem.horizon is not None:
        raise ValueError(
            "PolicyIteration requires problem.horizon=None (infinite horizon); "
            "use BackwardInduction for finite horizons"
        )

    ctx = setup_solve(
        problem, state_grid, action_grid, solver.n_quad,
        device=device, dtype=dtype,
    )

    V = terminal_value(ctx)

    policy_now: dict = {}
    delta = float("inf")
    for n_iter in range(solver.max_iters):
        V_new, policy_now = bellman_step(ctx, V, t=None)
        delta = (V_new - V).abs().max().item()
        V = V_new
        if delta < solver.tol:
            break
    else:
        raise RuntimeError(
            f"PolicyIteration did not converge in {solver.max_iters} iterations "
            f"(final ||ΔV||_∞ = {delta:.3e}, target {solver.tol:.3e})"
        )

    # Store under key None so policy(state, t=None) and value(state, t=None)
    # work uniformly with the finite-horizon API.
    return (
        _Policy(
            ctx.state_names, ctx.state_kinds, ctx.axes_for_lookup,
            ctx.transforms, {None: policy_now}, ctx.action_kinds,
        ),
        _Value(
            ctx.state_names, ctx.state_kinds, ctx.axes_for_lookup,
            ctx.transforms, {None: V},
        ),
    )
