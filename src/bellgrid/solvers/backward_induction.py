"""`BackwardInduction` solver — finite-horizon backward sweep.

T sweeps of the Bellman operator from a terminal-reward initialisation,
storing V and π at every t. The per-step Bellman update logic lives in
``_common.bellman_step``; this file is the thin outer loop and the
``BackwardInduction`` solver dataclass.
"""

from dataclasses import dataclass

import torch

from ..problem import Problem
from ._common import (
    _Policy, _Value, bellman_step, check_boundary_escape, setup_solve,
    terminal_value,
)


@dataclass(frozen=True)
class BackwardInduction:
    """Backward induction over a discretized state grid.

    Attributes
    ----------
    n_quad : int
        Number of Gauss-Hermite quadrature nodes per shock dimension.
    boundary_check : bool
        After the solve, run one extra ``problem.transition`` call with
        the optimal policy and warn if a non-trivial fraction of next-
        states fall outside any ``ContinuousState``'s range. Cheap
        (under ~5% overhead in the worst case, near-zero on big solves)
        and has caught real boundary bugs in development. Disable only
        for tight inner loops (e.g., gradient-based calibration that
        re-solves repeatedly).
    """

    n_quad: int = 7
    boundary_check: bool = True


def _backward_induction(
    problem: Problem,
    state_grid: dict,
    action_grid: dict,
    solver: BackwardInduction,
    *,
    device,
    dtype: torch.dtype,
    chunk_size: int,
) -> tuple[_Policy, _Value]:
    if problem.horizon is None:
        raise NotImplementedError(
            "BackwardInduction requires a finite horizon; pass "
            "`solver=PolicyIteration(...)` for the infinite-horizon case"
        )

    ctx = setup_solve(
        problem, state_grid, action_grid, solver.n_quad,
        device=device, dtype=dtype, chunk_size=chunk_size,
    )

    V_next = terminal_value(ctx)
    V_by_t: dict = {}
    policy_by_t: dict = {}
    last_t = None
    last_policy: dict = {}
    for t in reversed(list(problem.horizon)):
        V_now, pol_now = bellman_step(ctx, V_next, t)
        V_by_t[t] = V_now
        policy_by_t[t] = pol_now
        V_next = V_now
        last_t = t
        last_policy = pol_now

    # Boundary diagnostic on the final (earliest-t) optimal policy. The
    # boundary issue is usually consistent across t so one check is enough.
    if solver.boundary_check and last_policy:
        check_boundary_escape(ctx, last_policy, last_t)

    return (
        _Policy(
            ctx.state_names, ctx.state_kinds, ctx.axes_for_lookup,
            ctx.transforms, policy_by_t, ctx.action_kinds,
        ),
        _Value(
            ctx.state_names, ctx.state_kinds, ctx.axes_for_lookup,
            ctx.transforms, V_by_t,
        ),
    )
