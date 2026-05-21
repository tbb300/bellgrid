"""`BackwardInduction` solver — finite-horizon backward sweep.

T sweeps of the Bellman operator from a terminal-reward initialisation,
storing V and π at every t. The per-step Bellman update logic lives in
``_common.bellman_step``; this file is the thin outer loop and the
``BackwardInduction`` solver dataclass.
"""

from dataclasses import dataclass

import torch

from ..problem import Problem
from ._common import _Policy, _Value, bellman_step, setup_solve, terminal_value


@dataclass(frozen=True)
class BackwardInduction:
    """Backward induction over a discretized state grid.

    Attributes
    ----------
    n_quad : int
        Number of Gauss-Hermite quadrature nodes per shock dimension.
    """

    n_quad: int = 7


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
        device=device, dtype=dtype,
    )

    V_next = terminal_value(ctx)
    V_by_t: dict = {}
    policy_by_t: dict = {}
    for t in reversed(list(problem.horizon)):
        V_now, pol_now = bellman_step(ctx, V_next, t)
        V_by_t[t] = V_now
        policy_by_t[t] = pol_now
        V_next = V_now

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
