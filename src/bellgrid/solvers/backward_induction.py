"""`BackwardInduction` solver — default for finite-horizon problems.

Tracer-slice implementation: supports exactly one `ContinuousState`, any
number of `ContinuousAction`s, and zero or one `Normal` shock. Bigger
state/action/shock combinations land as the rest of the library does.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from ..grids.regular import RegularGrid
from ..interpolation.multilinear import multilinear
from ..problem import ContinuousAction, ContinuousState, Problem
from ..shocks.lognormal import Lognormal
from ..shocks.normal import Normal

_SUPPORTED_SHOCKS = (Normal, Lognormal)


@dataclass(frozen=True)
class BackwardInduction:
    """Backward induction over a discretized state grid.

    Attributes
    ----------
    n_quad : int
        Number of Gauss-Hermite quadrature nodes for each `Normal` shock.
    """

    n_quad: int = 7


class _Policy:
    """Callable wrapping the time-indexed optimal action arrays."""

    def __init__(
        self,
        state_name: str,
        u_pts: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor],
        policy_by_t: dict,
    ) -> None:
        self._state_name = state_name
        self._u_pts = u_pts
        self._transform = transform
        self._policy_by_t = policy_by_t

    def __call__(self, state: dict, t):
        s = torch.as_tensor(state[self._state_name], dtype=self._u_pts.dtype)
        u = self._transform(s)
        return {
            name: multilinear([self._u_pts], arr, [u])
            for name, arr in self._policy_by_t[t].items()
        }


class _Value:
    """Callable wrapping the time-indexed value-function arrays."""

    def __init__(
        self,
        state_name: str,
        u_pts: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor],
        V_by_t: dict,
    ) -> None:
        self._state_name = state_name
        self._u_pts = u_pts
        self._transform = transform
        self._V_by_t = V_by_t

    def __call__(self, state: dict, t):
        s = torch.as_tensor(state[self._state_name], dtype=self._u_pts.dtype)
        u = self._transform(s)
        return multilinear([self._u_pts], self._V_by_t[t], [u])


def _backward_induction(
    problem: Problem,
    state_grid: dict,
    action_grid: dict,
    solver: BackwardInduction,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    chunk_size: int,
) -> tuple[_Policy, _Value]:
    # ---- scope restrictions for the tracer slice ------------------------
    cont_states = [s for s in problem.states if isinstance(s, ContinuousState)]
    if len(cont_states) != 1 or len(cont_states) != len(problem.states):
        raise NotImplementedError(
            f"tracer supports exactly one ContinuousState (and no other state "
            f"types); got {len(problem.states)} state(s)"
        )
    cont_actions = [a for a in problem.actions if isinstance(a, ContinuousAction)]
    if len(cont_actions) != len(problem.actions):
        raise NotImplementedError("tracer supports only ContinuousAction")
    if len(cont_actions) == 0:
        raise ValueError("Problem has no actions")

    supported = [s for s in problem.shocks if isinstance(s, _SUPPORTED_SHOCKS)]
    if len(supported) != len(problem.shocks):
        raise NotImplementedError(
            f"tracer supports only {[c.__name__ for c in _SUPPORTED_SHOCKS]} shocks"
        )
    if len(supported) > 1:
        raise NotImplementedError(
            f"tracer supports at most one shock; got {len(supported)}"
        )

    if problem.horizon is None:
        raise NotImplementedError("infinite horizon not implemented in tracer")
    if callable(problem.discount):
        raise NotImplementedError("callable discount not implemented in tracer")

    state = cont_states[0]
    state_name = state.name

    # ---- build state grid ----------------------------------------------
    if state_name not in state_grid:
        raise ValueError(f"state_grid missing entry for state {state_name!r}")
    grid_spec = state_grid[state_name]
    from ..grids.warped import WarpedGrid

    if not isinstance(grid_spec, (RegularGrid, WarpedGrid)):
        raise NotImplementedError(
            f"tracer only supports RegularGrid/WarpedGrid; "
            f"got {type(grid_spec).__name__}"
        )
    s_pts = grid_spec.points(
        *state.range, dtype=dtype, device=device, warp=state.warp
    )
    N_s = s_pts.shape[0]
    # Coordinates used for interpolation. Identity under RegularGrid; the
    # forward warp under WarpedGrid — so a log-warp + log-utility problem
    # is interp-exact, asinh shaves a chunk off the V error vs. physical
    # interp, etc.
    u_pts = grid_spec.transform_for_interp(s_pts, warp=state.warp)

    def _to_u(x: torch.Tensor) -> torch.Tensor:
        return grid_spec.transform_for_interp(x, warp=state.warp)

    # ---- build action grids (cartesian product on normalized [0, 1]) ---
    for action in cont_actions:
        if action.name not in action_grid:
            raise ValueError(
                f"action_grid missing entry for action {action.name!r}"
            )

    action_axes = [
        action_grid[a.name].points(0.0, 1.0, dtype=dtype, device=device)
        for a in cont_actions
    ]
    mesh = torch.meshgrid(*action_axes, indexing="ij")
    action_normalized = {
        a.name: m.reshape(-1) for a, m in zip(cont_actions, mesh)
    }
    N_a = next(iter(action_normalized.values())).numel()

    # Rescale to actual bounds. State-dependent upper/lower become (N_s, N_a).
    action_tensors: dict[str, torch.Tensor] = {}
    for action in cont_actions:
        norm = action_normalized[action.name]  # (N_a,)

        def _resolve(b):
            if isinstance(b, str):
                if b == state_name:
                    return s_pts.unsqueeze(-1)  # (N_s, 1)
                raise NotImplementedError(
                    f"action bound reference to {b!r}: only the single "
                    f"declared state is supported in the tracer"
                )
            return torch.as_tensor(float(b), dtype=dtype, device=device)

        lo = _resolve(action.bounds[0])
        hi = _resolve(action.bounds[1])
        a = lo + (hi - lo) * norm  # broadcasts to (N_s, N_a) or (N_a,)
        if a.ndim == 1:
            a = a.unsqueeze(0).expand(N_s, -1)
        action_tensors[action.name] = a.contiguous()

    # ---- shock quadrature ----------------------------------------------
    if supported:
        shock = supported[0]
        shock_nodes, shock_weights = shock.nodes_and_weights(
            solver.n_quad, dtype=dtype, device=device
        )
        N_q = shock_nodes.numel()
        shock_name = shock.name
    else:
        shock_nodes = torch.zeros(1, dtype=dtype, device=device)
        shock_weights = torch.ones(1, dtype=dtype, device=device)
        N_q = 1
        shock_name = None

    # ---- initialize V at the post-horizon boundary ---------------------
    if problem.terminal_reward is None:
        V_next = torch.zeros(N_s, dtype=dtype, device=device)
    else:
        tr = problem.terminal_reward({state_name: s_pts})
        V_next = torch.as_tensor(tr, dtype=dtype, device=device)
        if V_next.ndim == 0:
            V_next = V_next.expand(N_s).contiguous()

    discount = float(problem.discount)

    # ---- backward sweep ------------------------------------------------
    horizon = list(problem.horizon)
    V_by_t: dict = {}
    policy_by_t: dict = {}

    # Broadcasting setup (constant across t)
    state_b = s_pts.view(N_s, 1, 1).expand(N_s, N_a, N_q)
    shock_b = shock_nodes.view(1, 1, N_q).expand(N_s, N_a, N_q)
    state_dict_b = {state_name: state_b}
    shock_dict_b = {shock_name: shock_b} if shock_name is not None else {}

    action_b = {
        name: tensor.unsqueeze(-1).expand(N_s, N_a, N_q)
        for name, tensor in action_tensors.items()
    }

    weights_b = shock_weights.view(1, 1, N_q)

    for t in reversed(horizon):
        # reward and transition can be (N_s, N_a) or (N_s, N_a, N_q);
        # broadcast_to handles either case uniformly.
        r = problem.reward(state_dict_b, action_b, shock_dict_b, t)
        r = torch.as_tensor(r, dtype=dtype, device=device)
        r = r.broadcast_to((N_s, N_a, N_q)).contiguous()

        next_state_dict = problem.transition(state_dict_b, action_b, shock_dict_b, t)
        if state_name not in next_state_dict:
            raise ValueError(
                f"transition return dict missing state key {state_name!r}"
            )
        next_s = torch.as_tensor(next_state_dict[state_name], dtype=dtype, device=device)
        next_s = next_s.broadcast_to((N_s, N_a, N_q)).contiguous()

        u_next = _to_u(next_s)
        V_at_next = multilinear([u_pts], V_next, [u_next])  # (N_s, N_a, N_q)

        # Bellman: max_a E_shock[ r(s,a,xi) + discount * V(s'(s,a,xi)) ]
        integrand = r + discount * V_at_next
        bellman = (integrand * weights_b).sum(dim=-1)  # (N_s, N_a)

        V_now, argmax = bellman.max(dim=1)  # (N_s,), (N_s,)
        V_by_t[t] = V_now
        policy_by_t[t] = {
            name: torch.gather(tensor, 1, argmax.unsqueeze(1)).squeeze(1)
            for name, tensor in action_tensors.items()
        }

        V_next = V_now

    return (
        _Policy(state_name, u_pts, _to_u, policy_by_t),
        _Value(state_name, u_pts, _to_u, V_by_t),
    )
