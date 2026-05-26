"""`PolicyIteration` solver — infinite-horizon stationary problems.

Iterate the Bellman operator from an initial guess (``terminal_reward``
if supplied, else zeros) until ``||V_new - V||_∞ < tol``, or until
``max_iters`` is hit.

The default ``k_howard = 10`` runs **modified policy iteration**: each
outer iteration is one Bellman *improvement* (a full ``max_a`` over
actions, the expensive step) followed by ``k_howard − 1`` cheap
Bellman *evaluations* at the just-improved policy (no max, just a
single action per state). The eval step costs ``1 / N_a`` of an
improvement, so a moderate ``k_howard`` is nearly free per outer iter
but typically cuts the outer-iter count by ~5–50× on smooth contractive
problems.

``k_howard = 1`` reverts to plain value iteration (the historical
behaviour). Very large ``k_howard`` (say 100+) approximates Howard's
exact policy iteration, where each outer iter solves ``V = T_σ V`` to
near-fixed-point before re-improving. The total Bellman applications
are ``n_outer × k_howard``; the sweet spot trades improvement count
against eval count.

Requires ``problem.horizon is None``. The user's ``transition`` and
``reward`` callables receive ``t=None`` — they should not depend on time.
"""

from dataclasses import dataclass

import torch

from ..problem import Problem
from ._common import (
    _evaluate_at_policy, _Policy, _Value, bellman_step,
    check_boundary_escape, setup_solve, terminal_value,
)


@dataclass(frozen=True)
class PolicyIteration:
    """Modified policy iteration for infinite-horizon stationary problems.

    Attributes
    ----------
    n_quad : int
        Number of Gauss-Hermite quadrature nodes per shock dimension.
    tol : float
        Convergence threshold on ``||V_new - V||_∞`` between successive
        outer iterations.
    max_iters : int
        Safety cap on the outer-iteration count. Exceeding this is a
        hard error (better to fail loudly than silently return a
        non-converged answer).
    k_howard : int
        Number of Bellman applications per outer iteration: 1 full
        improvement (``max_a``) followed by ``k_howard − 1`` evaluations
        at the fixed policy. ``k_howard = 1`` reverts to plain value
        iteration. Defaults to 10 — a sweet spot on most contractive
        problems where each eval step is ``~1 / N_a`` of an improvement,
        and the extra evals sharpen V enough to roughly halve the outer
        iter count.
    boundary_check : bool
        After convergence, run one extra ``problem.transition`` call with
        the stationary optimal policy and warn if a non-trivial fraction
        of next-states fall outside any ``ContinuousState``'s range. See
        ``BackwardInduction.boundary_check`` — same trade-off.
    """

    n_quad: int = 7
    tol: float = 1e-6
    max_iters: int = 10_000
    k_howard: int = 10
    boundary_check: bool = True

    def __post_init__(self):
        if self.k_howard < 1:
            raise ValueError(
                f"PolicyIteration requires k_howard >= 1, got {self.k_howard}"
            )


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
        device=device, dtype=dtype, chunk_size=chunk_size,
    )

    V = terminal_value(ctx)

    policy_now: dict = {}
    delta = float("inf")
    for n_iter in range(solver.max_iters):
        # Improvement step: full Bellman max → updated V and policy.
        V_new, policy_now = bellman_step(ctx, V, t=None)
        # Policy-evaluation steps: hold the freshly-improved policy
        # fixed and apply T_σ a further (k_howard − 1) times. Each is
        # ~1/N_a the cost of bellman_step; the contraction here keeps V
        # converging toward the fixed point of T_σ between improvements.
        for _ in range(solver.k_howard - 1):
            V_new = _evaluate_at_policy(ctx, V_new, policy_now, t=None)
        delta = (V_new - V).abs().max().item()
        V = V_new
        if delta < solver.tol:
            break
    else:
        raise RuntimeError(
            f"PolicyIteration did not converge in {solver.max_iters} iterations "
            f"(final ||ΔV||_∞ = {delta:.3e}, target {solver.tol:.3e})"
        )

    # Boundary diagnostic on the converged stationary policy.
    if solver.boundary_check:
        check_boundary_escape(ctx, policy_now, t=None)

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
