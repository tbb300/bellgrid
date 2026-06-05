"""Top-level `solve()` entry point — dispatches on solver type."""

import torch

from .problem import Problem
from .rl.pgrad import PolicyGradient, _policy_gradient
from .rl.solver import ActorCritic, _actor_critic
from .solvers.backward_induction import BackwardInduction, _backward_induction
from .solvers.ilqg import iLQG, _ilqg
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
    state_grid: dict | None = None,
    action_grid: dict | None = None,
    solver,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    chunk_size: int | None = None,
):
    """Solve a `Problem`, returning `(policy, value)` callables.

    `policy(state, t)` returns a dict of action values; `value(state, t)`
    returns the value-function scalar. Both return tensors on the same
    device as the queried state. For ``PolicyIteration`` (infinite
    horizon) the value/policy are stationary — pass ``t=None``.

    The grid solvers (``BackwardInduction``, ``PolicyIteration``) require
    ``state_grid`` and ``action_grid``. The ``ActorCritic`` solver samples
    states and learns ``π``/``V`` as networks, so it ignores both grids and
    reads the mesh region from the states' ``range``/``warp`` directly.

    ``device=None`` (default) picks CUDA if available, else CPU.
    ``chunk_size=None`` (default) auto-picks based on free GPU memory (grid
    solvers only).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(solver, ActorCritic):
        return _actor_critic(problem, solver, device=device, dtype=dtype)

    if isinstance(solver, PolicyGradient):
        return _policy_gradient(problem, solver, device=device, dtype=dtype)

    if isinstance(solver, iLQG):
        return _ilqg(problem, solver, device=device, dtype=dtype)

    if isinstance(solver, (BackwardInduction, PolicyIteration)):
        if state_grid is None or action_grid is None:
            raise ValueError(
                f"{type(solver).__name__} requires both state_grid and "
                "action_grid"
            )
        if chunk_size is None:
            chunk_size = _default_chunk_size(device, dtype)
        if isinstance(solver, BackwardInduction):
            return _backward_induction(
                problem, state_grid, action_grid, solver,
                device=device, dtype=dtype, chunk_size=chunk_size,
            )
        return _policy_iteration(
            problem, state_grid, action_grid, solver,
            device=device, dtype=dtype, chunk_size=chunk_size,
        )

    raise TypeError(f"unknown solver: {type(solver).__name__}")
