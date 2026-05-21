"""Top-level `solve()` entry point — dispatches on solver type."""

import torch

from .problem import Problem
from .solvers.backward_induction import BackwardInduction, _backward_induction


def solve(
    problem: Problem,
    *,
    state_grid: dict,
    action_grid: dict,
    solver,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    chunk_size: int = 2**20,
):
    """Solve a `Problem`, returning `(policy, value)` callables.

    `policy(state, t)` returns a dict of action values; `value(state, t)`
    returns the value-function scalar. Both interpolate from the per-`t`
    grid arrays produced by the solver and return tensors on the same
    device as the queried state.

    ``device=None`` (default) picks CUDA if available, else CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(solver, BackwardInduction):
        return _backward_induction(
            problem,
            state_grid,
            action_grid,
            solver,
            device=device,
            dtype=dtype,
            chunk_size=chunk_size,
        )
    raise TypeError(f"unknown solver: {type(solver).__name__}")
