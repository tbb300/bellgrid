"""Top-level `solve()` entry point — dispatches on solver type."""

import torch

from .problem import Problem
from .solvers.backward_induction import BackwardInduction, _backward_induction
from .solvers.policy_iteration import PolicyIteration, _policy_iteration


def _default_chunk_size(device, dtype):
    """Default per-chunk working-tensor budget.

    Empirically, 2**20 = 1M elements wins on both small problems
    (better fits the GPU's L2 cache; lifecycle solve is 14% faster
    than at 256M chunk_size) and large action-chunked problems (peaks
    are bandwidth-bound, so bigger chunks don't reduce time — and they
    can cause OOM on a 48 GB GPU once you account for the 5-10×
    intermediate-tensor footprint of a Bellman step). Users who want
    to experiment for their specific problem can override via the
    ``chunk_size`` kwarg.
    """
    return 2**20


def solve(
    problem: Problem,
    *,
    state_grid: dict,
    action_grid: dict,
    solver,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    chunk_size: int | None = None,
):
    """Solve a `Problem`, returning `(policy, value)` callables.

    `policy(state, t)` returns a dict of action values; `value(state, t)`
    returns the value-function scalar. Both interpolate from the per-`t`
    grid arrays produced by the solver and return tensors on the same
    device as the queried state. For ``PolicyIteration`` (infinite
    horizon) the value/policy are stationary — pass ``t=None``.

    ``device=None`` (default) picks CUDA if available, else CPU.
    ``chunk_size=None`` (default) auto-picks based on free GPU memory.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if chunk_size is None:
        chunk_size = _default_chunk_size(device, dtype)
    if isinstance(solver, BackwardInduction):
        return _backward_induction(
            problem, state_grid, action_grid, solver,
            device=device, dtype=dtype, chunk_size=chunk_size,
        )
    if isinstance(solver, PolicyIteration):
        return _policy_iteration(
            problem, state_grid, action_grid, solver,
            device=device, dtype=dtype, chunk_size=chunk_size,
        )
    raise TypeError(f"unknown solver: {type(solver).__name__}")
