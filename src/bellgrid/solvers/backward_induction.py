"""`BackwardInduction` solver — default for finite-horizon problems.

Supports any number of ``ContinuousState``s (joint state grid is their
cartesian product, value function is K-D), any mix of ``ContinuousAction``s
and ``DiscreteAction``s, zero or one ``Normal``/``Lognormal`` shock,
finite horizon, scalar discount.

Memory scales as ``∏ n_k × N_a × N_q``. Chunking will land when ``K > 2``
problems start running into memory pressure.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from ..grids.regular import RegularGrid
from ..grids.warped import WarpedGrid
from ..interpolation.multilinear import multilinear
from ..problem import ContinuousAction, ContinuousState, DiscreteAction, Problem
from ..shocks.lognormal import Lognormal
from ..shocks.normal import Normal

_SUPPORTED_SHOCKS = (Normal, Lognormal)
_SUPPORTED_ACTIONS = (ContinuousAction, DiscreteAction)
_SUPPORTED_GRIDS = (RegularGrid, WarpedGrid)


@dataclass(frozen=True)
class BackwardInduction:
    """Backward induction over a discretized state grid.

    Attributes
    ----------
    n_quad : int
        Number of Gauss-Hermite quadrature nodes for each `Normal`/`Lognormal`
        shock.
    """

    n_quad: int = 7


class _Policy:
    """Callable wrapping the time-indexed optimal action arrays.

    Continuous actions are interpolated multilinearly in the (warped) state
    coordinates. Discrete actions return the optimal index at the nearest
    state grid point (linear interp on integer indices isn't meaningful).
    """

    def __init__(
        self,
        state_names: list[str],
        u_pts_list: list[torch.Tensor],
        transforms: list[Callable[[torch.Tensor], torch.Tensor]],
        policy_by_t: dict,
        action_kinds: dict,
    ) -> None:
        self._state_names = state_names
        self._u_pts_list = u_pts_list
        self._transforms = transforms
        self._policy_by_t = policy_by_t
        self._action_kinds = action_kinds

    def __call__(self, state: dict, t):
        u_queries = [
            transform(torch.as_tensor(state[name], dtype=u.dtype))
            for name, u, transform in zip(
                self._state_names, self._u_pts_list, self._transforms
            )
        ]
        result = {}
        for name, arr in self._policy_by_t[t].items():
            if self._action_kinds[name] == "discrete":
                if len(self._u_pts_list) != 1:
                    raise NotImplementedError(
                        "discrete-action lookup currently supports K=1 only"
                    )
                result[name] = _nearest_neighbor(
                    self._u_pts_list[0], arr, u_queries[0]
                )
            else:
                result[name] = multilinear(self._u_pts_list, arr, u_queries)
        return result


def _nearest_neighbor(
    u_pts: torch.Tensor, values: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Look up `values` at the grid index closest (in u-space) to each query."""
    idx = torch.searchsorted(u_pts, query, right=False)
    idx = torch.clamp(idx, 1, u_pts.numel() - 1)
    left = idx - 1
    right = idx
    d_left = (query - u_pts[left]).abs()
    d_right = (u_pts[right] - query).abs()
    closer = torch.where(d_left <= d_right, left, right)
    return values[closer]


class _Value:
    """Callable wrapping the time-indexed value-function arrays."""

    def __init__(
        self,
        state_names: list[str],
        u_pts_list: list[torch.Tensor],
        transforms: list[Callable[[torch.Tensor], torch.Tensor]],
        V_by_t: dict,
    ) -> None:
        self._state_names = state_names
        self._u_pts_list = u_pts_list
        self._transforms = transforms
        self._V_by_t = V_by_t

    def __call__(self, state: dict, t):
        u_queries = [
            transform(torch.as_tensor(state[name], dtype=u.dtype))
            for name, u, transform in zip(
                self._state_names, self._u_pts_list, self._transforms
            )
        ]
        return multilinear(self._u_pts_list, self._V_by_t[t], u_queries)


def _make_warp_transform(grid_spec, warp):
    """Return a closure for transform_for_interp that captures grid_spec and warp."""
    def _transform(x: torch.Tensor) -> torch.Tensor:
        return grid_spec.transform_for_interp(x, warp=warp)
    return _transform


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
    # ---- scope checks ---------------------------------------------------
    if any(not isinstance(s, ContinuousState) for s in problem.states):
        raise NotImplementedError(
            "tracer supports only ContinuousState (no DiscreteState/MarkovChain yet)"
        )
    if len(problem.states) == 0:
        raise ValueError("Problem has no states")
    if any(not isinstance(a, _SUPPORTED_ACTIONS) for a in problem.actions):
        raise NotImplementedError(
            f"tracer supports only {[c.__name__ for c in _SUPPORTED_ACTIONS]}"
        )
    if len(problem.actions) == 0:
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

    cont_states = list(problem.states)
    K = len(cont_states)
    state_names = [s.name for s in cont_states]
    state_name_set = set(state_names)

    cont_actions = [a for a in problem.actions if isinstance(a, ContinuousAction)]
    for action in cont_actions:
        if action.name not in action_grid:
            raise ValueError(
                f"action_grid missing entry for action {action.name!r}"
            )

    # ---- build per-state grids and interp transforms --------------------
    s_pts_list: list[torch.Tensor] = []   # physical-space grid per state (n_k,)
    u_pts_list: list[torch.Tensor] = []   # interp-space grid per state (n_k,)
    transforms: list[Callable] = []
    state_dims: list[int] = []
    for state in cont_states:
        if state.name not in state_grid:
            raise ValueError(f"state_grid missing entry for state {state.name!r}")
        grid_spec = state_grid[state.name]
        if not isinstance(grid_spec, _SUPPORTED_GRIDS):
            raise NotImplementedError(
                f"tracer only supports {[c.__name__ for c in _SUPPORTED_GRIDS]}; "
                f"got {type(grid_spec).__name__}"
            )
        s_pts = grid_spec.points(
            *state.range, dtype=dtype, device=device, warp=state.warp
        )
        u_pts = grid_spec.transform_for_interp(s_pts, warp=state.warp)
        s_pts_list.append(s_pts)
        u_pts_list.append(u_pts)
        transforms.append(_make_warp_transform(grid_spec, state.warp))
        state_dims.append(s_pts.numel())

    state_dims_tup = tuple(state_dims)  # (n_1, ..., n_K)

    # Joint state mesh, one per state name. Each tensor has shape (n_1, ..., n_K).
    s_meshes = torch.meshgrid(*s_pts_list, indexing="ij")
    state_meshes_dict = {name: m for name, m in zip(state_names, s_meshes)}

    # ---- build joint action grid (cartesian product over actions) -----
    action_sizes = []
    for a in problem.actions:
        if isinstance(a, ContinuousAction):
            action_sizes.append(action_grid[a.name].n)
        else:  # DiscreteAction
            action_sizes.append(a.n)
    N_a = 1
    for sz in action_sizes:
        N_a *= sz

    index_axes = [
        torch.arange(sz, dtype=torch.long, device=device) for sz in action_sizes
    ]
    index_mesh = torch.meshgrid(*index_axes, indexing="ij")
    index_flat = [m.reshape(-1) for m in index_mesh]  # K tensors of shape (N_a,)

    def _resolve_bound(b):
        if isinstance(b, str):
            if b not in state_name_set:
                raise ValueError(
                    f"action bound references undeclared state {b!r}"
                )
            if K != 1:
                raise NotImplementedError(
                    "state-dependent action bounds with K > 1 not supported yet"
                )
            return s_pts_list[0].unsqueeze(-1)  # (n_1, 1)
        return torch.as_tensor(float(b), dtype=dtype, device=device)

    # Expand any per-action value tensor to broadcastable shape
    # (n_1, ..., n_K, N_a). Static values arrive as (N_a,); state-dependent
    # 1-D continuous values arrive as (n_1, N_a).
    def _to_state_shape(val: torch.Tensor) -> torch.Tensor:
        if val.ndim == 1:
            view_shape = (1,) * K + (N_a,)
            return val.view(view_shape).expand(state_dims_tup + (N_a,)).contiguous()
        if val.ndim == 2 and K == 1 and val.shape == (state_dims[0], N_a):
            return val.contiguous()
        raise NotImplementedError(
            f"action tensor with shape {tuple(val.shape)} not supported for K={K}"
        )

    action_tensors: dict[str, torch.Tensor] = {}
    action_kinds: dict[str, str] = {}
    for a, idx in zip(problem.actions, index_flat):
        if isinstance(a, ContinuousAction):
            norm_grid = action_grid[a.name].points(
                0.0, 1.0, dtype=dtype, device=device
            )
            norm = norm_grid[idx]  # (N_a,)
            lo = _resolve_bound(a.bounds[0])
            hi = _resolve_bound(a.bounds[1])
            val = lo + (hi - lo) * norm  # (N_a,) or (n_1, N_a)
            action_tensors[a.name] = _to_state_shape(val)
            action_kinds[a.name] = "continuous"
        else:  # DiscreteAction
            action_tensors[a.name] = _to_state_shape(idx)
            action_kinds[a.name] = "discrete"

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
        V_next = torch.zeros(state_dims_tup, dtype=dtype, device=device)
    else:
        tr = problem.terminal_reward(state_meshes_dict)
        V_next = torch.as_tensor(tr, dtype=dtype, device=device)
        V_next = V_next.broadcast_to(state_dims_tup).contiguous()

    discount = float(problem.discount)

    # ---- backward sweep ------------------------------------------------
    horizon = list(problem.horizon)
    V_by_t: dict = {}
    policy_by_t: dict = {}

    # Full broadcast shape: (n_1, ..., n_K, N_a, N_q)
    full_shape = state_dims_tup + (N_a, N_q)

    # State broadcasting: each state's mesh (n_1, ..., n_K) → (n_1, ..., n_K, 1, 1)
    state_b_dict = {
        name: m.reshape(state_dims_tup + (1, 1)).expand(full_shape)
        for name, m in state_meshes_dict.items()
    }

    # Action broadcasting: (n_1, ..., n_K, N_a) → (n_1, ..., n_K, N_a, 1) → full
    action_b = {
        name: tensor.unsqueeze(-1).expand(full_shape)
        for name, tensor in action_tensors.items()
    }

    # Shock broadcasting: (N_q,) → (1, ..., 1, 1, N_q) → full
    shock_view = (1,) * K + (1, N_q)
    if shock_name is not None:
        shock_b = shock_nodes.view(shock_view).expand(full_shape)
        shock_dict_b = {shock_name: shock_b}
    else:
        shock_dict_b = {}

    weights_b = shock_weights.view(shock_view)

    for t in reversed(horizon):
        r = problem.reward(state_b_dict, action_b, shock_dict_b, t)
        r = torch.as_tensor(r, dtype=dtype, device=device)
        r = r.broadcast_to(full_shape).contiguous()

        next_state_dict = problem.transition(state_b_dict, action_b, shock_dict_b, t)
        u_next_list = []
        for name, transform in zip(state_names, transforms):
            if name not in next_state_dict:
                raise ValueError(
                    f"transition return dict missing state key {name!r}"
                )
            next_val = torch.as_tensor(
                next_state_dict[name], dtype=dtype, device=device
            )
            next_val = next_val.broadcast_to(full_shape).contiguous()
            u_next_list.append(transform(next_val))

        V_at_next = multilinear(u_pts_list, V_next, u_next_list)  # full_shape

        integrand = r + discount * V_at_next
        bellman = (integrand * weights_b).sum(dim=-1)  # (n_1, ..., n_K, N_a)

        V_now, argmax = bellman.max(dim=-1)  # (n_1, ..., n_K), (n_1, ..., n_K)
        V_by_t[t] = V_now
        policy_by_t[t] = {
            name: torch.gather(tensor, -1, argmax.unsqueeze(-1)).squeeze(-1)
            for name, tensor in action_tensors.items()
        }
        V_next = V_now

    return (
        _Policy(state_names, u_pts_list, transforms, policy_by_t, action_kinds),
        _Value(state_names, u_pts_list, transforms, V_by_t),
    )
