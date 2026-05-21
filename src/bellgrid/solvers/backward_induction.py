"""`BackwardInduction` solver — default for finite-horizon problems.

Supports any number of states (any mix of ``ContinuousState``,
``DiscreteState``, and at most one ``MarkovChain``), any mix of
``ContinuousAction``s and ``DiscreteAction``s, zero or one
``Normal``/``Lognormal``/``MultivariateNormal`` shock, finite horizon,
scalar discount.

Internal state-axis ordering: continuous first, then discrete, then
markov. The user provides ``problem.states`` in any order; the solver
reorders to canonical for the Bellman update so the markov matrix-
contraction broadcasts cleanly. The user-facing state dict in their
``transition`` / ``reward`` callables is keyed by name, so the order
they declared doesn't leak through.

Memory scales as ``∏ n_k × N_a × N_q`` (and ``× n_m_next`` while a
markov chain is being integrated).
"""

from dataclasses import dataclass
from typing import Callable

import torch

from ..grids.regular import RegularGrid
from ..grids.warped import WarpedGrid
from ..interpolation.multilinear import multilinear
from ..problem import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    MarkovChain,
    Problem,
)
from ..shocks.lognormal import Lognormal
from ..shocks.multivariate_normal import MultivariateNormal
from ..shocks.normal import Normal

_SUPPORTED_SHOCKS = (Normal, Lognormal, MultivariateNormal)
_SUPPORTED_ACTIONS = (ContinuousAction, DiscreteAction)
_SUPPORTED_STATES = (ContinuousState, DiscreteState, MarkovChain)
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


def _query_device(sample) -> torch.device:
    if isinstance(sample, torch.Tensor):
        return sample.device
    return torch.device("cpu")


def _identity_transform(x: torch.Tensor) -> torch.Tensor:
    return x


class _Policy:
    """Callable wrapping the time-indexed optimal action arrays.

    Continuous actions are interpolated multilinearly across continuous
    state axes and exact-gather'd across discrete / markov axes. Discrete
    actions return the optimal index: with a single continuous state and
    no discrete/markov axes we use nearest-neighbor (linear interp on
    integer indices isn't meaningful); for K>1 or mixed states we
    multilinear and round, which is approximate near sharp policy
    transitions but fine elsewhere.
    """

    def __init__(
        self,
        state_names: list[str],
        state_kinds: list[str],
        axes_for_lookup: list[torch.Tensor],
        transforms: list[Callable[[torch.Tensor], torch.Tensor]],
        policy_by_t: dict,
        action_kinds: dict,
    ) -> None:
        self._state_names = state_names
        self._state_kinds = state_kinds
        self._axes_for_lookup = axes_for_lookup
        self._transforms = transforms
        self._policy_by_t = policy_by_t
        self._action_kinds = action_kinds

    def __call__(self, state: dict, t):
        target_device = _query_device(state[self._state_names[0]])
        queries = _build_queries(
            state, self._state_names, self._state_kinds,
            self._axes_for_lookup, self._transforms,
        )
        single_cont = (
            len(self._state_kinds) == 1 and self._state_kinds[0] == "continuous"
        )
        result = {}
        for name, arr in self._policy_by_t[t].items():
            if self._action_kinds[name] == "discrete":
                if single_cont:
                    out = _nearest_neighbor(
                        self._axes_for_lookup[0], arr, queries[0]
                    )
                else:
                    out = multilinear(
                        self._axes_for_lookup, arr.to(torch.float64), queries
                    ).round().to(arr.dtype)
            else:
                out = multilinear(self._axes_for_lookup, arr, queries)
            result[name] = out.to(target_device)
        return result


def _nearest_neighbor(
    u_pts: torch.Tensor, values: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
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
        state_kinds: list[str],
        axes_for_lookup: list[torch.Tensor],
        transforms: list[Callable[[torch.Tensor], torch.Tensor]],
        V_by_t: dict,
    ) -> None:
        self._state_names = state_names
        self._state_kinds = state_kinds
        self._axes_for_lookup = axes_for_lookup
        self._transforms = transforms
        self._V_by_t = V_by_t

    def __call__(self, state: dict, t):
        target_device = _query_device(state[self._state_names[0]])
        queries = _build_queries(
            state, self._state_names, self._state_kinds,
            self._axes_for_lookup, self._transforms,
        )
        out = multilinear(self._axes_for_lookup, self._V_by_t[t], queries)
        return out.to(target_device)


def _build_queries(
    state: dict,
    state_names: list[str],
    state_kinds: list[str],
    axes_for_lookup: list[torch.Tensor],
    transforms: list[Callable],
) -> list[torch.Tensor]:
    """Per-axis lookup queries from the user-supplied state dict."""
    queries = []
    for name, kind, axis, transform in zip(
        state_names, state_kinds, axes_for_lookup, transforms
    ):
        val = state[name]
        if kind == "continuous":
            tval = torch.as_tensor(val, dtype=axis.dtype, device=axis.device)
            queries.append(transform(tval))
        else:
            tval = torch.as_tensor(val, dtype=torch.long, device=axis.device)
            queries.append(tval)
    return queries


def _make_warp_transform(grid_spec, warp):
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
    if any(not isinstance(s, _SUPPORTED_STATES) for s in problem.states):
        raise NotImplementedError(
            f"tracer supports only {[c.__name__ for c in _SUPPORTED_STATES]}"
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

    mc_states_user = [s for s in problem.states if isinstance(s, MarkovChain)]
    if len(mc_states_user) > 1:
        raise NotImplementedError(
            "tracer supports at most one MarkovChain per problem"
        )

    if problem.horizon is None:
        raise NotImplementedError("infinite horizon not implemented in tracer")
    if callable(problem.discount):
        raise NotImplementedError("callable discount not implemented in tracer")

    # ---- canonical ordering: continuous → discrete → markov ------------
    cont_states = [s for s in problem.states if isinstance(s, ContinuousState)]
    disc_states = [s for s in problem.states if isinstance(s, DiscreteState)]
    mc_states = mc_states_user
    state_order = cont_states + disc_states + mc_states
    state_names = [s.name for s in state_order]
    state_name_set = set(state_names)
    K = len(state_order)
    K_cont = len(cont_states)
    K_disc = len(disc_states)
    K_mc = len(mc_states)

    # ---- build per-state axes ------------------------------------------
    state_axes: list[torch.Tensor] = []      # 1-D physical-space (or arange) axis
    axes_for_lookup: list[torch.Tensor] = [] # axis to pass to multilinear
    transforms: list[Callable] = []
    state_kinds: list[str] = []
    state_dims: list[int] = []

    for state in state_order:
        if isinstance(state, ContinuousState):
            if state.name not in state_grid:
                raise ValueError(
                    f"state_grid missing entry for state {state.name!r}"
                )
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
            state_axes.append(s_pts)
            axes_for_lookup.append(u_pts)
            transforms.append(_make_warp_transform(grid_spec, state.warp))
            state_kinds.append("continuous")
            state_dims.append(s_pts.numel())
        elif isinstance(state, DiscreteState):
            arange = torch.arange(state.n, dtype=torch.long, device=device)
            state_axes.append(arange)
            axes_for_lookup.append(arange)
            transforms.append(_identity_transform)
            state_kinds.append("discrete")
            state_dims.append(state.n)
        else:  # MarkovChain
            arange = torch.arange(state.n, dtype=torch.long, device=device)
            state_axes.append(arange)
            axes_for_lookup.append(arange)
            transforms.append(_identity_transform)
            state_kinds.append("markov")
            state_dims.append(state.n)

    state_dims_tup = tuple(state_dims)

    # Build the joint state mesh manually rather than via torch.meshgrid:
    # mixed continuous (float) and discrete/markov (long) axes have
    # different dtypes, which meshgrid doesn't allow.
    state_meshes_dict = {}
    for k, (name, axis) in enumerate(zip(state_names, state_axes)):
        view_shape = [1] * K
        view_shape[k] = axis.numel()
        state_meshes_dict[name] = axis.view(view_shape).expand(state_dims_tup)

    # ---- joint action grid ---------------------------------------------
    cont_actions = [a for a in problem.actions if isinstance(a, ContinuousAction)]
    for action in cont_actions:
        if action.name not in action_grid:
            raise ValueError(
                f"action_grid missing entry for action {action.name!r}"
            )

    action_sizes = []
    for a in problem.actions:
        if isinstance(a, ContinuousAction):
            action_sizes.append(action_grid[a.name].n)
        else:
            action_sizes.append(a.n)
    N_a = 1
    for sz in action_sizes:
        N_a *= sz

    index_axes = [
        torch.arange(sz, dtype=torch.long, device=device) for sz in action_sizes
    ]
    index_mesh = torch.meshgrid(*index_axes, indexing="ij")
    index_flat = [m.reshape(-1) for m in index_mesh]

    def _resolve_bound(b):
        if isinstance(b, str):
            if b not in state_name_set:
                raise ValueError(
                    f"action bound references undeclared state {b!r}"
                )
            pos = state_names.index(b)
            if state_kinds[pos] != "continuous":
                raise NotImplementedError(
                    "state-dependent action bounds may only reference a "
                    f"ContinuousState (got {type(state_order[pos]).__name__})"
                )
            if K_cont != 1:
                raise NotImplementedError(
                    "state-dependent action bounds require exactly one "
                    "ContinuousState (currently)"
                )
            # The (single) continuous state is at position 0 in canonical order.
            return state_axes[0].unsqueeze(-1)  # (n_c, 1)
        return torch.as_tensor(float(b), dtype=dtype, device=device)

    def _to_state_shape(val: torch.Tensor) -> torch.Tensor:
        if val.ndim == 1:
            view_shape = (1,) * K + (N_a,)
            return val.view(view_shape).expand(state_dims_tup + (N_a,)).contiguous()
        if val.ndim == 2 and K_cont == 1 and val.shape == (state_dims[0], N_a):
            view_shape = (state_dims[0],) + (1,) * (K - 1) + (N_a,)
            return val.view(view_shape).expand(state_dims_tup + (N_a,)).contiguous()
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
            norm = norm_grid[idx]
            lo = _resolve_bound(a.bounds[0])
            hi = _resolve_bound(a.bounds[1])
            val = lo + (hi - lo) * norm
            action_tensors[a.name] = _to_state_shape(val)
            action_kinds[a.name] = "continuous"
        else:
            action_tensors[a.name] = _to_state_shape(idx)
            action_kinds[a.name] = "discrete"

    # ---- shock quadrature ----------------------------------------------
    if supported:
        shock = supported[0]
        raw, shock_weights = shock.nodes_and_weights(
            solver.n_quad, dtype=dtype, device=device
        )
        if isinstance(raw, dict):
            shock_values = raw
            N_q = next(iter(shock_values.values())).numel()
        else:
            shock_values = {shock.name: raw}
            N_q = raw.numel()
    else:
        shock_values = {}
        shock_weights = torch.ones(1, dtype=dtype, device=device)
        N_q = 1

    # ---- terminal V ----------------------------------------------------
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

    full_shape = state_dims_tup + (N_a, N_q)
    state_b_dict = {
        name: m.reshape(state_dims_tup + (1, 1)).expand(full_shape)
        for name, m in state_meshes_dict.items()
    }
    action_b = {
        name: tensor.unsqueeze(-1).expand(full_shape)
        for name, tensor in action_tensors.items()
    }
    shock_view = (1,) * K + (1, N_q)
    shock_dict_b = {
        name: nodes.view(shock_view).expand(full_shape)
        for name, nodes in shock_values.items()
    }
    weights_b = shock_weights.view(shock_view)

    # The V-lookup shape and (if present) the markov matrix-contract
    # tensor are the same every backward step — set them up once.
    if K_mc == 1:
        mc = mc_states[0]
        n_m = mc.n
        lookup_shape = full_shape + (n_m,)
        matrix_t = torch.as_tensor(mc.matrix, dtype=dtype, device=device)
        m_axis_pos = K_cont + K_disc  # current_m position in state_dims
        view_shape = [1] * (len(full_shape) + 1)
        view_shape[m_axis_pos] = n_m
        view_shape[-1] = n_m  # next_m
        matrix_b = matrix_t.view(view_shape)
    else:
        lookup_shape = full_shape
        matrix_b = None
        n_m = 0

    for t in reversed(horizon):
        r = problem.reward(state_b_dict, action_b, shock_dict_b, t)
        r = torch.as_tensor(r, dtype=dtype, device=device)
        r = r.broadcast_to(full_shape).contiguous()

        next_state_dict = problem.transition(state_b_dict, action_b, shock_dict_b, t)

        queries = []
        for name, kind, transform in zip(state_names, state_kinds, transforms):
            if kind == "continuous":
                if name not in next_state_dict:
                    raise ValueError(
                        f"transition return dict missing state key {name!r}"
                    )
                nv = torch.as_tensor(next_state_dict[name], dtype=dtype, device=device)
                nv = nv.broadcast_to(full_shape).contiguous()
                u_next = transform(nv)
                if K_mc == 1:
                    u_next = u_next.unsqueeze(-1).expand(lookup_shape)
                queries.append(u_next)
            elif kind == "discrete":
                if name not in next_state_dict:
                    raise ValueError(
                        f"transition return dict missing state key {name!r}"
                    )
                nv = torch.as_tensor(
                    next_state_dict[name], dtype=torch.long, device=device
                )
                nv = nv.broadcast_to(full_shape).contiguous()
                if K_mc == 1:
                    nv = nv.unsqueeze(-1).expand(lookup_shape)
                queries.append(nv)
            else:  # markov — solver-controlled
                if name in next_state_dict:
                    raise ValueError(
                        f"transition must not return MarkovChain state {name!r} "
                        "(advanced via its transition matrix)"
                    )
                arange = torch.arange(n_m, dtype=torch.long, device=device)
                view_arange = [1] * len(full_shape) + [n_m]
                arange_b = arange.view(view_arange).expand(lookup_shape)
                queries.append(arange_b)

        V_lookup = multilinear(axes_for_lookup, V_next, queries)  # lookup_shape

        if K_mc == 1:
            V_at_next = (V_lookup * matrix_b).sum(dim=-1)
        else:
            V_at_next = V_lookup

        integrand = r + discount * V_at_next
        bellman = (integrand * weights_b).sum(dim=-1)

        V_now, argmax = bellman.max(dim=-1)
        V_by_t[t] = V_now
        policy_by_t[t] = {
            name: torch.gather(tensor, -1, argmax.unsqueeze(-1)).squeeze(-1)
            for name, tensor in action_tensors.items()
        }
        V_next = V_now

    return (
        _Policy(state_names, state_kinds, axes_for_lookup, transforms, policy_by_t, action_kinds),
        _Value(state_names, state_kinds, axes_for_lookup, transforms, V_by_t),
    )
