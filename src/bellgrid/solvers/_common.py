"""Shared setup and per-step Bellman code for backward-induction and
policy-iteration solvers.

The two solvers differ only in their outer loop (T fixed sweeps vs.
iterate-to-convergence). Everything else — state mesh construction,
action enumeration, shock quadrature, the per-step Bellman update with
markov matrix contraction — is identical, lives here, and is shared
via the ``SolveContext`` dataclass that bundles up the broadcast arrays.
"""

import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ..grids.golden import GoldenSearch
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
from ..shocks.categorical import Categorical
from ..shocks.jump import Jump
from ..shocks.lognormal import Lognormal
from ..shocks.multivariate_normal import MultivariateNormal
from ..shocks.normal import Normal
from ..shocks.uniform import Uniform

_SUPPORTED_SHOCKS = (Normal, Lognormal, MultivariateNormal, Jump, Categorical, Uniform)
_SUPPORTED_ACTIONS = (ContinuousAction, DiscreteAction)
_SUPPORTED_STATES = (ContinuousState, DiscreteState, MarkovChain)
_SUPPORTED_GRIDS = (RegularGrid, WarpedGrid)


def _identity_transform(x: torch.Tensor) -> torch.Tensor:
    return x


def _make_warp_transform(grid_spec, warp):
    def _transform(x: torch.Tensor) -> torch.Tensor:
        return grid_spec.transform_for_interp(x, warp=warp)

    return _transform


def _query_device(sample) -> torch.device:
    if isinstance(sample, torch.Tensor):
        return sample.device
    return torch.device("cpu")


@dataclass
class SolveContext:
    """All the broadcast arrays + metadata for one Bellman step.

    Built once by ``setup_solve`` and reused for every step of the outer
    loop in both backward-induction and policy-iteration solvers.
    """

    problem: Problem
    discount: float

    # State axes in canonical order: continuous → discrete → markov.
    state_names: list[str]
    state_kinds: list[str]                # "continuous" / "discrete" / "markov" per axis
    state_axes: list[torch.Tensor]         # physical-space axis (float for continuous, long for disc/mc)
    axes_for_lookup: list[torch.Tensor]    # warp-transformed (continuous) or arange (disc/mc)
    transforms: list[Callable]             # query → lookup-space transform per axis
    state_dims_tup: tuple
    state_meshes_dict: dict
    K: int
    K_mc: int

    action_tensors: dict
    action_kinds: dict
    N_a: int

    N_q: int

    full_shape: tuple
    lookup_shape: tuple
    state_b_dict: dict
    action_b: dict
    shock_dict_b: dict
    weights_b: torch.Tensor

    # Markov-chain integration data. One entry per chain in canonical
    # ordering (continuous → discrete → markov, so markov chains are at
    # the end of state_dims). Each ``matrix_b`` is broadcastable to the
    # lookup tensor with size n_mk at the chain's current-state axis and
    # size n_mk at the last kept axis (we contract chains in reverse,
    # always summing the trailing axis).
    n_ms: list                             # [n_m1, n_m2, ...] empty if no chains
    matrices_b: list                       # [matrix_b1, matrix_b2, ...]

    dtype: torch.dtype
    device: Any

    # Soft cap on per-Bellman-step memory: at most this many tensor elements
    # in any intermediate. The Bellman update chunks the shock axis so each
    # chunk stays under the cap (well, modulo a few constant-size temporaries).
    chunk_size: int = 2**20

    # Continuous-state range cache, used by the boundary diagnostic.
    state_ranges: dict = field(default_factory=dict)

    # True if the user's ``reward`` accepts a 5th positional argument
    # ``next_state`` — for next-state-dependent payoffs like a per-period
    # bequest. Detected from ``inspect.signature`` at setup time.
    reward_takes_next_state: bool = False

    # Per-continuous-action search strategy: "grid" (enumerate
    # ``action_tensors`` and ``max``) or "golden" (use ``action_tensors``
    # as a coarse seed, then refine via vectorized golden-section).
    # Populated by ``setup_solve`` from the user's ``action_grid`` dict.
    # Empty if no continuous actions.
    action_search: dict = field(default_factory=dict)

    # Per-continuous-action GoldenSearch spec (n_iter, n_coord). Absent
    # for actions on the grid path. Keeps the per-action refinement
    # parameters close to the rest of the action metadata.
    golden_specs: dict = field(default_factory=dict)

    # Resolved action bounds per continuous action, used by golden-search
    # to build per-state brackets. Each entry is a (lo, hi) pair of
    # tensors broadcastable to ``state_dims + (1,)`` (or 0-d scalar
    # tensors for fixed bounds).
    action_bounds: dict = field(default_factory=dict)

    # Raw 1-D shock-node tensors, length ``N_q`` each (after tensor-product
    # combination). Used by the golden-search path to rebuild Bellman
    # broadcasts at arbitrary action values, since ``shock_dict_b`` is
    # already expanded to ``full_shape``.
    shock_values: dict = field(default_factory=dict)

    # Joint-axis layout for the golden-search path. Discrete and grid-
    # continuous actions stay enumerated through refinement; golden-
    # continuous actions get their grid axes collapsed and refined.
    # ``enum_sizes`` lists the per-action sizes along the joint enumeration
    # axis in the action declaration order (== ``action_sizes`` from
    # setup); ``enum_strides[i]`` is the row-major stride of action i in
    # the joint axis. Populated regardless of whether any action is
    # golden so callers can decompose ``argmax`` deterministically.
    enum_sizes: list = field(default_factory=list)
    enum_strides: list = field(default_factory=list)


def setup_solve(
    problem: Problem,
    state_grid: dict,
    action_grid: dict,
    n_quad: int,
    *,
    device,
    dtype: torch.dtype,
    chunk_size: int = 2**20,
) -> SolveContext:
    """Build a ``SolveContext`` from the problem and grid specs."""
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

    mc_states_user = [s for s in problem.states if isinstance(s, MarkovChain)]

    # Callable discount (``discount(state, t)`` returning a scalar tensor or
    # something broadcastable to the state mesh) is supported; scalar
    # discount is the common case. We don't need to do anything special at
    # setup time — ``bellman_step`` checks ``callable()`` per call.

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
    state_axes: list[torch.Tensor] = []
    axes_for_lookup: list[torch.Tensor] = []
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
        """Return either a scalar or a tensor of shape ``[1]*K + [1]``
        with size ``n_c`` at the referenced state's axis and 1 elsewhere
        (plus a trailing 1 for the action axis). ``lo + (hi - lo) * norm``
        then broadcasts correctly through ``_to_state_shape``.
        """
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
            # K state axes + 1 trailing action axis. Size n_c at the
            # referenced state's position, 1 everywhere else.
            view_shape = [1] * (K + 1)
            view_shape[pos] = state_dims[pos]
            return state_axes[pos].view(view_shape)
        return torch.as_tensor(float(b), dtype=dtype, device=device)

    # Heuristic: materialise action_tensors to a contiguous joint
    # state × action tensor when it's small (downstream ops are slightly
    # faster on contiguous inputs), but keep it as a broadcast view
    # when the joint size exceeds ~4 GB (which would OOM on a typical
    # GPU). The view costs nothing at allocation time and downstream
    # ops (action_b expand, transition arithmetic, policy gather) read
    # through the strides correctly.
    _ACTION_MATERIALIZE_BYTES = 4 * (1024 ** 3)
    _dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    _action_tensor_bytes = math.prod(state_dims_tup + (N_a,)) * _dtype_bytes

    def _to_state_shape(val: torch.Tensor) -> torch.Tensor:
        """Broadcast an action value tensor to the joint state-action shape."""
        target = state_dims_tup + (N_a,)
        if val.ndim == 0:
            val = val.view((1,) * (K + 1))
        elif val.ndim == 1 and val.shape[0] == N_a:
            val = val.view((1,) * K + (N_a,))
        bc = val.broadcast_to(target)
        # Only contiguous-materialise genuinely *state-dependent* values (e.g.
        # state-dependent action bounds, which carry a real size on some state
        # axis). Fixed-bound continuous actions and all discrete actions are
        # identical across states, so materialising them to the full
        # state × N_a shape just burns memory and bandwidth — keep the
        # broadcast view (downstream slicing/expand/gather read it fine).
        state_dependent = any(val.shape[k] != 1 for k in range(K))
        if state_dependent and _action_tensor_bytes <= _ACTION_MATERIALIZE_BYTES:
            return bc.contiguous()
        return bc

    action_tensors: dict = {}
    action_kinds: dict = {}
    action_search: dict = {}
    golden_specs: dict = {}
    action_bounds: dict = {}
    for a, idx in zip(problem.actions, index_flat):
        if isinstance(a, ContinuousAction):
            grid_spec = action_grid[a.name]
            norm_grid = grid_spec.points(0.0, 1.0, dtype=dtype, device=device)
            norm = norm_grid[idx]
            lo = _resolve_bound(a.bounds[0])
            hi = _resolve_bound(a.bounds[1])
            val = lo + (hi - lo) * norm
            action_tensors[a.name] = _to_state_shape(val)
            action_kinds[a.name] = "continuous"
            # Stash bounds and search strategy for the golden-search path.
            # Even grid-path continuous actions get bounds saved — the
            # golden path holds them constant via ``action_tensors`` but a
            # uniform layout simplifies the code.
            action_bounds[a.name] = (lo, hi)
            if isinstance(grid_spec, GoldenSearch):
                action_search[a.name] = "golden"
                golden_specs[a.name] = grid_spec
            else:
                action_search[a.name] = "grid"
        else:
            action_tensors[a.name] = _to_state_shape(idx)
            action_kinds[a.name] = "discrete"

    # ---- shock quadrature (tensor product over independent shocks) -----
    if supported:
        shock_values: dict = {}
        shock_weights = torch.tensor([1.0], dtype=dtype, device=device)
        N_q = 1
        for shock in supported:
            raw, w = shock.nodes_and_weights(
                n_quad, dtype=dtype, device=device
            )
            this_nodes = raw if isinstance(raw, dict) else {shock.name: raw}
            n_this = w.numel()
            new_values = {
                name: val.unsqueeze(-1).expand(N_q, n_this).reshape(-1).contiguous()
                for name, val in shock_values.items()
            }
            for name, val in this_nodes.items():
                new_values[name] = (
                    val.unsqueeze(0).expand(N_q, n_this).reshape(-1).contiguous()
                )
            shock_weights = (
                shock_weights.unsqueeze(-1) * w.unsqueeze(0)
            ).reshape(-1).contiguous()
            shock_values = new_values
            N_q = N_q * n_this
    else:
        shock_values = {}
        shock_weights = torch.ones(1, dtype=dtype, device=device)
        N_q = 1

    # Pass through; bellman_step handles both scalar and callable discount.
    discount = problem.discount if callable(problem.discount) else float(problem.discount)

    # Detect whether reward takes a 5th positional argument (next_state).
    # We accept either ``reward(state, action, shock, t)`` (the historical
    # signature) or ``reward(state, action, shock, t, next_state)``.
    _sig = inspect.signature(problem.reward)
    _positional = [
        p for p in _sig.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(_positional) == 4:
        reward_takes_next_state = False
    elif len(_positional) == 5:
        reward_takes_next_state = True
    else:
        raise ValueError(
            f"reward must take 4 or 5 positional arguments "
            f"(state, action, shock, t [, next_state]); got {len(_positional)}"
        )

    # ---- broadcast arrays for the Bellman update -----------------------
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

    # For each markov chain, build a matrix tensor broadcastable to the
    # lookup tensor AT THE TIME WE CONTRACT IT. We contract chains in
    # REVERSE order (chain K_mc - 1 first, chain 0 last), summing the
    # trailing axis each step. When chain k is contracted, chains
    # k+1, ..., K_mc-1 have already been summed out, so V_at_next has
    # dim count ``len(full_shape) + (k + 1)``. matrix_b for chain k
    # therefore has that dim count, with size n_mk at the chain's
    # current-state axis (K_cont + K_disc + k) and at the last axis
    # (the kept axis being summed).
    n_ms: list = [mc.n for mc in mc_states]
    matrices_b: list = []
    lookup_shape = full_shape + tuple(n_ms)
    for k, mc in enumerate(mc_states):
        matrix_t = torch.as_tensor(mc.matrix, dtype=dtype, device=device)
        m_axis_pos = K_cont + K_disc + k
        n_dims_at_contraction = len(full_shape) + (k + 1)
        view_shape = [1] * n_dims_at_contraction
        view_shape[m_axis_pos] = mc.n
        view_shape[-1] = mc.n
        matrices_b.append(matrix_t.view(view_shape))

    state_ranges = {
        s.name: tuple(s.range)
        for s in problem.states
        if isinstance(s, ContinuousState)
    }

    # Joint-action axis strides — row-major over ``action_sizes``. The
    # joint axis flat index decomposes as
    # ``sum_i (action_index_i * enum_strides[i])``. Used by golden-search
    # to recover per-action seed indices from a single ``argmax`` over the
    # joint axis (cheap; one ``%``/``//`` per action).
    enum_strides: list = [1] * len(action_sizes)
    for i in range(len(action_sizes) - 2, -1, -1):
        enum_strides[i] = enum_strides[i + 1] * action_sizes[i + 1]

    return SolveContext(
        problem=problem,
        discount=discount,
        state_names=state_names,
        state_kinds=state_kinds,
        state_axes=state_axes,
        axes_for_lookup=axes_for_lookup,
        transforms=transforms,
        state_dims_tup=state_dims_tup,
        state_meshes_dict=state_meshes_dict,
        K=K,
        K_mc=K_mc,
        action_tensors=action_tensors,
        action_kinds=action_kinds,
        N_a=N_a,
        N_q=N_q,
        full_shape=full_shape,
        lookup_shape=lookup_shape,
        state_b_dict=state_b_dict,
        action_b=action_b,
        shock_dict_b=shock_dict_b,
        weights_b=weights_b,
        chunk_size=chunk_size,
        state_ranges=state_ranges,
        reward_takes_next_state=reward_takes_next_state,
        n_ms=n_ms,
        matrices_b=matrices_b,
        dtype=dtype,
        device=device,
        action_search=action_search,
        golden_specs=golden_specs,
        action_bounds=action_bounds,
        shock_values=shock_values,
        enum_sizes=list(action_sizes),
        enum_strides=enum_strides,
    )


def terminal_value(ctx: SolveContext) -> torch.Tensor:
    """Evaluate ``problem.terminal_reward`` on the joint state mesh (or zeros)."""
    if ctx.problem.terminal_reward is None:
        return torch.zeros(ctx.state_dims_tup, dtype=ctx.dtype, device=ctx.device)
    tr = ctx.problem.terminal_reward(ctx.state_meshes_dict)
    return torch.as_tensor(tr, dtype=ctx.dtype, device=ctx.device).broadcast_to(
        ctx.state_dims_tup
    ).contiguous()


def _validate_transition_keys(ctx: SolveContext, next_state_dict) -> None:
    """One-shot validation of the user's ``transition`` return dict.

    Hoisted out of the integrand so the golden-search path can validate
    once per Bellman step rather than once per refinement eval. Cheap
    either way (set comparisons on small dicts), but the savings add up
    over ``n_iter × n_coord × N_golden`` calls.
    """
    if not isinstance(next_state_dict, dict):
        raise ValueError(
            f"problem.transition() must return a dict; got "
            f"{type(next_state_dict).__name__}"
        )
    _expected = {
        name for name, kind in zip(ctx.state_names, ctx.state_kinds)
        if kind != "markov"
    }
    _forbidden = {
        name for name, kind in zip(ctx.state_names, ctx.state_kinds)
        if kind == "markov"
    }
    _got = set(next_state_dict.keys())
    _missing = _expected - _got
    if _missing:
        raise ValueError(
            f"problem.transition() return dict missing keys for non-markov "
            f"states: {sorted(_missing)}. Expected entries for every "
            f"ContinuousState and DiscreteState (MarkovChain states are "
            f"advanced internally and must NOT be returned)."
        )
    _forbidden_present = _forbidden & _got
    if _forbidden_present:
        raise ValueError(
            f"problem.transition() return dict contains entries for "
            f"MarkovChain states {sorted(_forbidden_present)}. These are "
            f"advanced internally by the solver via their transition "
            f"matrix — drop them from the return dict."
        )


def _bellman_core(
    ctx: SolveContext,
    V_next: torch.Tensor,
    t,
    state_b: dict,
    action_b: dict,
    shock_b: dict,
    weights_b: torch.Tensor,
    discount_chunk,
    chunk_shape: tuple,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """The Bellman integrand: ``Σ_q (r + γ V_next ∘ f) · w`` at the
    pre-broadcast tensors the caller supplies.

    Both callers — chunked grid enumeration (via ``_bellman_partial``) and
    golden-search refinement (via ``_bellman_at_action_values``) — share
    this body. The only difference is how they shape ``action_b``:
    grid enumeration slices the prebuilt ``ctx.action_b``; golden search
    broadcasts a per-state action-value tensor.

    Returns a ``state_dims + (M,)`` tensor where ``M`` is the size of the
    action axis in ``chunk_shape`` (i.e. ``chunk_shape[-2]``). The shock
    axis (``chunk_shape[-1]``) is summed out by ``weights_b``.
    """
    # Always evaluate the transition first — its output may be needed by a
    # next-state-aware reward.
    next_state_dict = ctx.problem.transition(state_b, action_b, shock_b, t)
    if validate:
        _validate_transition_keys(ctx, next_state_dict)

    if ctx.reward_takes_next_state:
        r = ctx.problem.reward(state_b, action_b, shock_b, t, next_state_dict)
    else:
        r = ctx.problem.reward(state_b, action_b, shock_b, t)
    # Keep r in its natural shape; downstream arithmetic broadcasts.
    r = torch.as_tensor(r, dtype=ctx.dtype, device=ctx.device)

    # Build the lookup tensor shape. With K_mc markov chains, the lookup
    # has one extra "kept" axis per chain, appended in chain order.
    lookup_shape = chunk_shape + tuple(ctx.n_ms)

    # Build the multilinear queries. Non-markov queries carry trailing
    # size-1 axes for the markov kept-dims rather than being expanded to the
    # full lookup_shape: the bracket (searchsorted + interp weights) is
    # identical across the stride-0 markov axes, so computing it at chunk size
    # and letting multilinear broadcast the gather up to lookup_shape avoids
    # prod(n_ms)× redundant searchsorted work and the matching materialisation
    # (in the lifecycle case, ~2.3 GB / period not written to GPU memory). The
    # markov query itself stays a size-1-everywhere-but-its-kept-axis view.
    queries = []
    mc_idx = 0
    for name, kind, transform in zip(
        ctx.state_names, ctx.state_kinds, ctx.transforms
    ):
        if kind == "continuous":
            nv = torch.as_tensor(
                next_state_dict[name], dtype=ctx.dtype, device=ctx.device
            )
            nv = nv.broadcast_to(chunk_shape)
            u_next = transform(nv)
            for _ in range(ctx.K_mc):
                u_next = u_next.unsqueeze(-1)
            queries.append(u_next)
        elif kind == "discrete":
            nv = torch.as_tensor(
                next_state_dict[name], dtype=torch.long, device=ctx.device
            )
            nv = nv.broadcast_to(chunk_shape)
            for _ in range(ctx.K_mc):
                nv = nv.unsqueeze(-1)
            queries.append(nv)
        else:  # markov — solver-controlled
            n_mk = ctx.n_ms[mc_idx]
            arange = torch.arange(n_mk, dtype=torch.long, device=ctx.device)
            view_arange = [1] * len(lookup_shape)
            view_arange[len(chunk_shape) + mc_idx] = n_mk
            queries.append(arange.view(view_arange))
            mc_idx += 1

    V_lookup = multilinear(ctx.axes_for_lookup, V_next, queries)

    # Contract markov-chain kept axes in REVERSE order so the trailing
    # axis is always the next one to sum. Each contraction reduces dim
    # count by 1; after K_mc contractions, shape is back to chunk_shape.
    # matrix_b for each chain is size-1 in the shock dim (and every dim
    # except the chain's current-axis and kept-axis), so it broadcasts
    # cleanly across the chunked shock slice.
    V_at_next = V_lookup
    for matrix_b in reversed(ctx.matrices_b):
        V_at_next = (V_at_next * matrix_b).sum(dim=-1)

    integrand = r + discount_chunk * V_at_next
    return (integrand * weights_b).sum(dim=-1)


def _bellman_partial(
    ctx: SolveContext,
    V_next: torch.Tensor,
    t,
    q_start: int,
    q_end: int,
    a_start: int,
    a_end: int,
    discount,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Contribution of shock nodes ``[q_start, q_end)`` and action chunk
    ``[a_start, a_end)`` to ``Σ_q (r + γ V) · w``.

    Returns a ``state_dims + (a_end - a_start,)``-shaped tensor: the
    partial Bellman integrand summed over the shock-node slice for this
    action subset. The caller composes across shock-slice calls (sum)
    and across action-chunk calls (running max + argmax). ``discount``
    is either a scalar or a tensor broadcastable to ``full_shape``
    (precomputed once per Bellman step).

    Thin wrapper around ``_bellman_core`` that slices the pre-built
    enumeration tensors held on ``ctx``.
    """
    n_q = q_end - q_start
    n_a_chunk = a_end - a_start
    chunk_shape = ctx.state_dims_tup + (n_a_chunk, n_q)

    # Slice the expanded views along the action and shock axes. Each
    # slice stays a view (no allocation) since slicing along an existing
    # axis is just an offset + size change.
    # Fast path: when the action axis is full (the common case — only
    # OOM-class problems chunk the action axis), avoid the redundant
    # ``[..., 0:N_a, ...]`` slice on every broadcast tensor. Each slice
    # is "free" in memory but allocates a new view object; over the
    # course of a finite-horizon solve with shock chunking that adds
    # up to hundreds of milliseconds.
    if a_start == 0 and a_end == ctx.N_a:
        state_b = {name: x[..., q_start:q_end] for name, x in ctx.state_b_dict.items()}
        action_b = {name: x[..., q_start:q_end] for name, x in ctx.action_b.items()}
        shock_b = {name: x[..., q_start:q_end] for name, x in ctx.shock_dict_b.items()}
    else:
        state_b = {name: x[..., a_start:a_end, q_start:q_end] for name, x in ctx.state_b_dict.items()}
        action_b = {name: x[..., a_start:a_end, q_start:q_end] for name, x in ctx.action_b.items()}
        shock_b = {name: x[..., a_start:a_end, q_start:q_end] for name, x in ctx.shock_dict_b.items()}
    # weights_b is held as a broadcast view (size 1 on the action axis,
    # not expanded). Always slice only the shock axis: slicing the
    # action axis on a size-1 view with ``a_start > 0`` would produce
    # size 0, and the view broadcasts cleanly downstream anyway.
    weights_b = ctx.weights_b[..., q_start:q_end]
    if isinstance(discount, torch.Tensor) and discount.ndim == len(ctx.full_shape):
        # Full-shape discount tensor — slice along both chunked axes.
        discount_chunk = discount[..., a_start:a_end, q_start:q_end]
    else:
        # Scalar tensor, smaller-shape tensor, or plain Python scalar —
        # broadcasts naturally in ``r + discount_chunk * V_at_next`` below.
        discount_chunk = discount

    return _bellman_core(
        ctx, V_next, t,
        state_b=state_b, action_b=action_b, shock_b=shock_b,
        weights_b=weights_b, discount_chunk=discount_chunk,
        chunk_shape=chunk_shape, validate=validate,
    )


def _bellman_at_action_values(
    ctx: SolveContext,
    V_next: torch.Tensor,
    t,
    action_values: dict,
    discount,
    *,
    validate: bool = False,
) -> torch.Tensor:
    """Bellman value at user-supplied per-state action values.

    Parameters
    ----------
    action_values : dict
        One entry per declared action. Each value is a tensor of shape
        ``state_dims + (M,)``: ``M`` candidate joint-action configurations
        per state. The semantics of "joint configuration" are up to the
        caller — typically ``M = N_disc`` (one continuous value per
        discrete combo) for the golden-search refinement loop.

    Returns
    -------
    Tensor of shape ``state_dims + (M,)``: the Bellman value at each
    candidate. The caller takes ``.max(dim=-1)`` to fold over the M
    candidates (e.g. to pick the optimal discrete combo after continuous
    refinement).
    """
    # M is the trailing-axis size of any action-value tensor. They all
    # share it by construction; we read it from an arbitrary one.
    sample = next(iter(action_values.values()))
    M = sample.shape[-1]
    chunk_shape = ctx.state_dims_tup + (M, ctx.N_q)

    state_b = {
        name: m.reshape(ctx.state_dims_tup + (1, 1)).expand(chunk_shape)
        for name, m in ctx.state_meshes_dict.items()
    }
    action_b = {
        name: val.unsqueeze(-1).expand(chunk_shape)
        for name, val in action_values.items()
    }
    shock_view = (1,) * ctx.K + (1, ctx.N_q)
    shock_b = {
        name: nodes.view(shock_view).expand(chunk_shape)
        for name, nodes in ctx.shock_values.items()
    }
    weights_b = ctx.weights_b  # already shaped (1,)*K + (1, N_q)

    # The discount is action-independent — β(s, t). A *state-dependent*
    # callable discount is evaluated on ctx.state_b_dict, so it comes in at
    # full_shape with size N_a on the action axis. This path's action axis is
    # M (the candidate count: 1 for policy-evaluation, N_enum for the golden
    # refinement), not N_a, so we collapse the discount's action axis to 1 and
    # let it broadcast. Without this the multiply re-expands the action axis to
    # N_a and the result blows up (crashing _evaluate_at_policy's squeeze and
    # the golden-search refinement). Scalars and smaller tensors pass through.
    if isinstance(discount, torch.Tensor) and discount.ndim == len(ctx.full_shape):
        discount = discount[..., :1, :]

    return _bellman_core(
        ctx, V_next, t,
        state_b=state_b, action_b=action_b, shock_b=shock_b,
        weights_b=weights_b, discount_chunk=discount,
        chunk_shape=chunk_shape,
        validate=validate,
    )


def _evaluate_at_policy(
    ctx: SolveContext,
    V_next: torch.Tensor,
    policy: dict,
    t,
) -> torch.Tensor:
    """One Bellman application at a *fixed* policy — no maximisation.

    ``T_σ V (s) = E_w[ r(s, σ(s), w) + γ V(f(s, σ(s), w)) ]``

    Used by ``PolicyIteration``'s Howard / modified-policy-iteration
    inner loop: after a full Bellman improvement (``bellman_step``), the
    policy is held fixed for ``k_howard − 1`` cheap eval steps that
    sharpen ``V`` toward the fixed point of ``T_σ`` before the next
    improvement. Each call here is the cost of a single grid action
    (``M = 1``), i.e. ``1 / N_a`` of an improvement step — so even
    moderate ``k_howard`` (~10) typically pays for itself many times
    over in fewer outer iterations.
    """
    # M = 1: each action is a single per-state value held fixed across
    # the refinement. Reuses the broadcast machinery in
    # ``_bellman_at_action_values``.
    action_values = {name: val.unsqueeze(-1) for name, val in policy.items()}

    if callable(ctx.discount):
        disc_raw = ctx.discount(ctx.state_b_dict, t)
        discount = torch.as_tensor(disc_raw, dtype=ctx.dtype, device=ctx.device)
    else:
        discount = ctx.discount

    V_at_policy = _bellman_at_action_values(
        ctx, V_next, t, action_values, discount, validate=False,
    )
    return V_at_policy.squeeze(-1)


# Golden ratio conjugate: 1 / φ = (√5 − 1) / 2 ≈ 0.618. Each iteration of
# the standard golden-section search contracts the bracket by this factor.
_GOLDEN_PHI_INV = (math.sqrt(5.0) - 1.0) / 2.0


def _golden_section_axis(
    ctx: SolveContext,
    V_next: torch.Tensor,
    t,
    discount,
    action_values: dict,
    axis_name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    n_iter: int,
) -> torch.Tensor:
    """One axis of vectorized golden-section search.

    Maximise the Bellman value over ``action_values[axis_name]`` while
    holding every other action fixed at its current value. ``a`` and
    ``b`` are the (per-cell) bracket bounds — same shape as
    ``action_values[axis_name]``.

    Uses the standard one-fresh-eval-per-iter trick: after contracting,
    one of the two new interior points coincides with the previous
    interior point on the other side (a consequence of φ² = 1 − φ⁻¹),
    so we can reuse its Bellman value and only evaluate the genuinely
    new point. We pick the per-cell fresh point with ``torch.where`` and
    do a single batched eval — so the GPU work is one Bellman call per
    iter regardless of the mask pattern.
    """
    h = b - a
    c = a + (1.0 - _GOLDEN_PHI_INV) * h
    d = a + _GOLDEN_PHI_INV * h

    def eval_at(val: torch.Tensor) -> torch.Tensor:
        av = dict(action_values)
        av[axis_name] = val
        return _bellman_at_action_values(
            ctx, V_next, t, av, discount, validate=False,
        )

    fc = eval_at(c)
    fd = eval_at(d)

    for _ in range(n_iter):
        keep_left = fc > fd                              # max is in [a, d]
        b = torch.where(keep_left, d, b)
        a = torch.where(keep_left, a, c)
        h = b - a
        c_new = a + (1.0 - _GOLDEN_PHI_INV) * h
        d_new = a + _GOLDEN_PHI_INV * h
        # In keep_left cells the genuinely new point is c_new; in the
        # other cells it's d_new. The OTHER new interior point coincides
        # with the previous interior point on the opposite side, so its
        # Bellman value carries over without re-evaluation.
        fresh = torch.where(keep_left, c_new, d_new)
        f_fresh = eval_at(fresh)
        fc_new = torch.where(keep_left, f_fresh, fd)
        fd_new = torch.where(keep_left, fc, f_fresh)
        c, d = c_new, d_new
        fc, fd = fc_new, fd_new

    # Pick whichever of the two final interior points has the higher
    # Bellman value (bracket midpoint would be slightly biased).
    use_c = fc > fd
    return torch.where(use_c, c, d)


def _refine_with_golden_section(
    ctx: SolveContext,
    V_next: torch.Tensor,
    t,
    bellman: torch.Tensor,
    discount,
) -> tuple[torch.Tensor, dict]:
    """Coordinate-descent golden-section refinement on top of a seed grid.

    ``bellman`` is the result of the grid Bellman: shape
    ``state_dims + (N_a,)``, with action indices in declaration order
    after a row-major flatten of the per-action sizes.

    We split the actions into:
      - **enumerated** axes — discrete actions and any continuous actions
        whose grid spec is *not* ``GoldenSearch``. These stay enumerated
        through refinement so the discrete choice can re-pick after the
        continuous values move.
      - **refined** axes — continuous actions with ``GoldenSearch``. Their
        seed grid indices are extracted per ``(state, enum_combo)``;
        their values are then coordinate-descended via
        ``_golden_section_axis``.

    Returns the final ``(V_now, policy_now)`` post-refinement.
    """
    action_names = [a.name for a in ctx.problem.actions]
    K = len(ctx.state_dims_tup)

    enum_positions: list = []
    refined_positions: list = []
    for i, name in enumerate(action_names):
        if ctx.action_search.get(name) == "golden":
            refined_positions.append(i)
        else:
            enum_positions.append(i)

    enum_sizes = [ctx.enum_sizes[i] for i in enum_positions]
    refined_sizes = [ctx.enum_sizes[i] for i in refined_positions]
    refined_names = [action_names[i] for i in refined_positions]
    N_enum = 1
    for s in enum_sizes:
        N_enum *= s

    # Reshape (state_dims + (N_a,)) → (state_dims + per_action_sizes).
    per_action_shape = ctx.state_dims_tup + tuple(ctx.enum_sizes)
    bellman_pa = bellman.view(per_action_shape)

    # Permute so enum axes come before refined axes (state_dims fixed).
    perm = (
        list(range(K))
        + [K + i for i in enum_positions]
        + [K + i for i in refined_positions]
    )
    bellman_perm = bellman_pa.permute(perm).contiguous()

    # Collapse enum axes into one, refined axes into one.
    n_refined = len(refined_sizes)
    refined_flat_size = 1
    for s in refined_sizes:
        refined_flat_size *= s
    bellman_collapsed = bellman_perm.reshape(
        ctx.state_dims_tup + (N_enum, refined_flat_size)
    )

    # Per-(state, enum_combo) best seed in the refined sub-grid.
    _, refined_flat_argmax = bellman_collapsed.max(dim=-1)  # state_dims + (N_enum,)

    # Decompose refined-flat argmax into per-refined-axis indices.
    refined_strides_local = [1] * n_refined
    for i in range(n_refined - 2, -1, -1):
        refined_strides_local[i] = refined_strides_local[i + 1] * refined_sizes[i + 1]
    seed_idx_per_refined: dict = {}
    rem = refined_flat_argmax
    for j, name in enumerate(refined_names):
        s = refined_strides_local[j]
        seed_idx_per_refined[name] = rem // s
        rem = rem % s

    # Per-(state, enum_combo) enum-axis indices, decomposed from the
    # arange over [0, N_enum). enum_strides_local is the stride in the
    # *collapsed* enum axis; ctx.enum_strides[i] is the stride of the
    # i-th action in the *original* N_a layout — both needed below.
    enum_strides_local = [1] * len(enum_sizes)
    for i in range(len(enum_sizes) - 2, -1, -1):
        enum_strides_local[i] = enum_strides_local[i + 1] * enum_sizes[i + 1]

    enum_combo_idx = torch.arange(N_enum, dtype=torch.long, device=ctx.device)
    enum_combo_b = enum_combo_idx.view((1,) * K + (N_enum,)).expand(
        ctx.state_dims_tup + (N_enum,)
    )

    per_axis_idx: list = [None] * len(action_names)
    rem = enum_combo_b
    for j, pos in enumerate(enum_positions):
        s = enum_strides_local[j]
        per_axis_idx[pos] = rem // s
        rem = rem % s
    for pos in refined_positions:
        per_axis_idx[pos] = seed_idx_per_refined[action_names[pos]]

    # Compose back into the original N_a flat index for gather.
    joint_flat = torch.zeros(
        ctx.state_dims_tup + (N_enum,), dtype=torch.long, device=ctx.device,
    )
    for i, idx_tensor in enumerate(per_axis_idx):
        if ctx.enum_strides[i] != 0:
            joint_flat = joint_flat + idx_tensor * ctx.enum_strides[i]

    # Gather per-action seed values at joint_flat.
    action_values: dict = {}
    target_a_shape = ctx.state_dims_tup + (ctx.N_a,)
    for name, at in ctx.action_tensors.items():
        at_full = at.broadcast_to(target_a_shape)
        action_values[name] = torch.gather(at_full, -1, joint_flat)

    # Coordinate-descent refinement. n_coord is the max across all golden
    # specs so every axis gets enough rounds; the per-axis n_iter still
    # controls how tightly each axis contracts.
    n_coord = max(ctx.golden_specs[name].n_coord for name in refined_names)
    for _ in range(n_coord):
        for name in refined_names:
            spec = ctx.golden_specs[name]
            lo_t, hi_t = ctx.action_bounds[name]
            current = action_values[name]
            lo_b = (
                lo_t.broadcast_to(current.shape)
                if isinstance(lo_t, torch.Tensor) and lo_t.ndim > 0
                else torch.full_like(current, float(lo_t))
            )
            hi_b = (
                hi_t.broadcast_to(current.shape)
                if isinstance(hi_t, torch.Tensor) and hi_t.ndim > 0
                else torch.full_like(current, float(hi_t))
            )
            # Bracket: one seed-cell width on each side of the current value,
            # clamped to the action bounds. This bounds the search to the
            # original seed cell on the first round and to a tight
            # neighbourhood on subsequent rounds — the global basin is
            # already pinned down by the seed grid.
            cell = (hi_b - lo_b) / (spec.n_init - 1)
            a = torch.maximum(current - cell, lo_b)
            b = torch.minimum(current + cell, hi_b)
            # If bracket is degenerate (current at boundary AND cell width
            # would put both ends at the same value), skip — there's
            # nothing to refine.
            action_values[name] = _golden_section_axis(
                ctx, V_next, t, discount, action_values, name, a, b, spec.n_iter,
            )

    # Final Bellman per (state, enum_combo), then pick best enum combo.
    V_per_enum = _bellman_at_action_values(
        ctx, V_next, t, action_values, discount, validate=False,
    )
    V_now, best_enum_idx = V_per_enum.max(dim=-1)

    policy_now: dict = {}
    gather_idx = best_enum_idx.unsqueeze(-1)
    for name, av in action_values.items():
        gathered = torch.gather(av, -1, gather_idx).squeeze(-1)
        if ctx.action_kinds.get(name) == "discrete":
            # Discrete action values are stored as floats in action_tensors
            # via _to_state_shape; restore the long dtype the user expects.
            gathered = gathered.to(torch.long)
        policy_now[name] = gathered

    return V_now, policy_now


def bellman_step(ctx: SolveContext, V_next: torch.Tensor, t) -> tuple[torch.Tensor, dict]:
    """One Bellman update.

    Returns ``(V_now, policy_now)`` where ``V_now`` has shape
    ``state_dims_tup`` and ``policy_now`` is a dict of optimal action
    tensors with the same shape.

    The shock axis is chunked so that no intermediate tensor exceeds
    roughly ``ctx.chunk_size`` elements. For small problems the chunk
    spans the entire shock axis and the loop is a single iteration; for
    larger problems we accumulate the Bellman expectation across chunks.

    When any continuous action uses ``GoldenSearch`` as its action grid
    spec, the grid Bellman is followed by a coordinate-descent
    golden-section refinement (see ``_refine_with_golden_section``). The
    seed grid (``n_init`` points per refined axis) pins the global
    basin; the refinement then sharpens the continuous values to ~1e-5
    of the action range in ~20 evals per axis — much tighter than a
    practical ``RegularGrid``.
    """
    # Pre-compute the discount once per Bellman step — same across all
    # shock-slice chunks. We keep it in whatever natural shape the user
    # returned (scalar tensors stay scalar; state-dependent tensors stay
    # at their natural shape). Broadcasting in the Bellman arithmetic
    # below handles all the shape variants without us having to
    # materialise to ``full_shape``.
    if callable(ctx.discount):
        disc_raw = ctx.discount(ctx.state_b_dict, t)
        discount = torch.as_tensor(disc_raw, dtype=ctx.dtype, device=ctx.device)
    else:
        discount = ctx.discount

    state_count = (
        math.prod(ctx.state_dims_tup) if ctx.state_dims_tup else 1
    )

    # Pick chunk sizes for the action and shock axes. Two distinct
    # questions:
    #
    #   (a) Does the state × action accumulator tensor fit in GPU
    #       memory? This tensor accumulates shock-chunked partial
    #       Bellman expectations; it has shape state × N_a regardless
    #       of how we chunk. If it doesn't fit, we MUST action-chunk
    #       (which avoids ever materialising it — we instead carry a
    #       running max + argmax of shape ``state_dims``).
    #
    #   (b) Does the per-chunk working tensor (state × action_chunk ×
    #       shock_chunk) fit in ``ctx.chunk_size``? This is a much
    #       smaller budget, applied to the transient Bellman-step
    #       tensors. Within whichever chunking mode (a) picked, we
    #       size shock chunks to honour it.
    #
    # The accumulator threshold is ``chunk_size * ACCUMULATOR_MULTIPLIER``;
    # the multiplier (16×) reflects that the accumulator is a single
    # tensor with no intermediate-tensor overhead, so it can be much
    # larger than the per-chunk working budget without OOM. For
    # ``chunk_size = 2**20`` (1M elements) on fp64 GPUs, the accumulator
    # may grow to 16M elements = 128 MB — comfortably small.
    ACCUMULATOR_MULTIPLIER = 16
    # The accumulator is the post-contraction (state × N_a) Bellman tensor —
    # no markov fan-out (markov current-state axes are already in state_count).
    accumulator_elements = state_count * ctx.N_a
    accumulator_fits = (
        accumulator_elements <= ctx.chunk_size * ACCUMULATOR_MULTIPLIER
    )
    # The per-chunk WORKING tensor, by contrast, is the *pre-contraction*
    # lookup: state × action_chunk × shock_chunk × prod(n_ms). Each markov
    # chain adds a "kept" next-state axis, so the working set is prod(n_ms)×
    # larger than state × action × shock. Size the shock/action chunks against
    # that — otherwise the cap silently under-counts and OOMs on problems with
    # sizable chains (the budget would be off by exactly prod(n_ms)).
    mc_fanout = math.prod(ctx.n_ms) if ctx.n_ms else 1
    working_per_shock_full_action = state_count * ctx.N_a * mc_fanout
    if accumulator_fits:
        # Shock-only chunking. Use as many shock nodes per chunk as
        # the per-chunk working budget allows.
        chunk_n_q = max(1, ctx.chunk_size // max(working_per_shock_full_action, 1))
        chunk_n_a = ctx.N_a
    else:
        # Action chunking. Take one shock at a time so the per-chunk
        # working tensor is state_count × chunk_n_a × 1 × prod(n_ms).
        chunk_n_q = 1
        chunk_n_a = max(1, ctx.chunk_size // max(state_count * mc_fanout, 1))

    full_q = (chunk_n_q >= ctx.N_q)
    full_a = (chunk_n_a >= ctx.N_a)

    has_golden = any(s == "golden" for s in ctx.action_search.values())

    if has_golden and not full_a:
        # The golden-search refinement needs the full state × N_a Bellman
        # tensor to extract per-(state, enum_combo) seed indices for the
        # refined axes. With a GoldenSearch seed grid (small n_init), N_a
        # is typically small enough that this is comfortable — but it
        # *would* OOM if combined with a huge non-golden RegularGrid.
        # That combination is wasteful anyway (you're paying full grid
        # cost on the non-golden axis and getting no speedup from golden),
        # so we error rather than silently fall back.
        raise RuntimeError(
            "GoldenSearch refinement requires the full state × N_a "
            "Bellman accumulator to fit in memory, but the current "
            f"chunk_size={ctx.chunk_size} would action-chunk this "
            f"problem (state={ctx.state_dims_tup}, N_a={ctx.N_a}). "
            "Drop other actions' grid sizes (or remove GoldenSearch) "
            "or raise chunk_size."
        )

    # The transition's return-key validation is structural (independent of
    # which shock/action slice we evaluate), so validate only on the first
    # _bellman_partial call of the step rather than once per chunk.
    if full_q and full_a:
        # Fast path: no chunking. Single _bellman_partial call, then
        # max over the action axis. Identical to pre-chunking behaviour.
        bellman = _bellman_partial(
            ctx, V_next, t, 0, ctx.N_q, 0, ctx.N_a, discount,
        )
        if has_golden:
            return _refine_with_golden_section(ctx, V_next, t, bellman, discount)
        V_now, argmax = bellman.max(dim=-1)
    elif full_a:
        # Shock-only chunking: accumulate the shock expectation into a
        # full (state, action) tensor, then max over action.
        bellman = torch.zeros(
            ctx.state_dims_tup + (ctx.N_a,),
            dtype=ctx.dtype, device=ctx.device,
        )
        for i_chunk, q_start in enumerate(range(0, ctx.N_q, chunk_n_q)):
            q_end = min(q_start + chunk_n_q, ctx.N_q)
            bellman += _bellman_partial(
                ctx, V_next, t, q_start, q_end, 0, ctx.N_a, discount,
                validate=(i_chunk == 0),
            )
        if has_golden:
            return _refine_with_golden_section(ctx, V_next, t, bellman, discount)
        V_now, argmax = bellman.max(dim=-1)
    else:
        # Action chunking: we can never materialise the full (state,
        # action) tensor, so per action chunk we run shock chunking to
        # accumulate that slice's Bellman expectation, then take the
        # max within the slice and update a running maximum + argmax.
        V_now = torch.full(
            ctx.state_dims_tup, float("-inf"),
            dtype=ctx.dtype, device=ctx.device,
        )
        argmax = torch.zeros(
            ctx.state_dims_tup, dtype=torch.long, device=ctx.device,
        )
        validated = False
        for a_start in range(0, ctx.N_a, chunk_n_a):
            a_end = min(a_start + chunk_n_a, ctx.N_a)
            if chunk_n_q >= ctx.N_q:
                chunk_bellman = _bellman_partial(
                    ctx, V_next, t, 0, ctx.N_q, a_start, a_end, discount,
                    validate=not validated,
                )
                validated = True
            else:
                chunk_bellman = torch.zeros(
                    ctx.state_dims_tup + (a_end - a_start,),
                    dtype=ctx.dtype, device=ctx.device,
                )
                for q_start in range(0, ctx.N_q, chunk_n_q):
                    q_end = min(q_start + chunk_n_q, ctx.N_q)
                    chunk_bellman += _bellman_partial(
                        ctx, V_next, t,
                        q_start, q_end, a_start, a_end, discount,
                        validate=not validated,
                    )
                    validated = True
            chunk_max, chunk_argmax_rel = chunk_bellman.max(dim=-1)
            chunk_argmax = chunk_argmax_rel + a_start
            update = chunk_max > V_now
            V_now = torch.where(update, chunk_max, V_now)
            argmax = torch.where(update, chunk_argmax, argmax)

    policy_now = {
        name: torch.gather(tensor, -1, argmax.unsqueeze(-1)).squeeze(-1)
        for name, tensor in ctx.action_tensors.items()
    }
    return V_now, policy_now


_BOUNDARY_ESCAPE_THRESHOLD = 0.10  # interior-mean across the state mesh
_BOUNDARY_INTERIOR_FRACTION = 0.10  # exclude this much of the outer mesh on each side per axis


def check_boundary_escape(
    ctx: SolveContext, policy_actions: dict, t,
    threshold: float = _BOUNDARY_ESCAPE_THRESHOLD,
    interior_fraction: float = _BOUNDARY_INTERIOR_FRACTION,
) -> dict:
    """Diagnose how much of next-state probability mass lands outside each
    continuous state's grid range under the optimal policy.

    Runs one extra ``problem.transition`` call with the optimal action
    plugged in for every state, then for each ContinuousState computes
    the shock-weighted fraction of next-states that fall outside ``[low,
    high]``. Returns a ``{name: {"max": fraction, "interior_mean":
    fraction}}`` dict. Emits a ``UserWarning`` for each state whose
    ``interior_mean`` exceeds ``threshold`` (default 10%).

    Why an **interior** mean: the literal grid-edge cell always overshoots
    100% of the time under any positive shock, but those cells are rarely
    visited in practice — a well-configured problem can have ~5% of its
    mesh "boundary-affected" purely from the topmost grid points and be
    completely fine. We exclude the outer ``interior_fraction`` of each
    continuous axis (default 10% on each side per axis) before averaging,
    so the diagnostic measures "is the boundary biting *interior* states
    the agent actually inhabits". (Without an explicit state distribution
    this is the best cheap proxy.)

    Multilinear clamps overshoot to the grid edge, which underestimates V
    in concave regions and biases the Bellman optimisation (we shipped
    two real bugs of this shape in the Merton and LQG examples before
    this check existed).
    """
    if not ctx.state_ranges:
        return {}

    state_dims = ctx.state_dims_tup
    N_q = ctx.N_q
    diagnostic_shape = state_dims + (N_q,)

    state_b = {
        name: m.reshape(state_dims + (1,)).expand(diagnostic_shape)
        for name, m in ctx.state_meshes_dict.items()
    }
    action_b = {
        name: tensor.unsqueeze(-1).expand(diagnostic_shape)
        for name, tensor in policy_actions.items()
    }
    shock_b = {
        name: x.select(-2, 0)  # state_dims + (N_q,)
        for name, x in ctx.shock_dict_b.items()
    }
    weights = ctx.weights_b.select(-2, 0)  # (1,)*K + (N_q,)

    next_state_dict = ctx.problem.transition(state_b, action_b, shock_b, t)

    # Build an interior mask: True for cells away from the edge of every
    # continuous axis. Discrete and markov axes don't contribute (they're
    # never "outside" since they're enumerated).
    interior_mask = torch.ones(state_dims, dtype=torch.bool, device=ctx.device)
    for k, kind in enumerate(ctx.state_kinds):
        if kind != "continuous":
            continue
        n_k = state_dims[k]
        cut = max(1, int(round(n_k * interior_fraction)))
        # Indices [cut, n_k - cut) are "interior" along axis k.
        axis_mask = torch.zeros(n_k, dtype=torch.bool, device=ctx.device)
        if n_k - 2 * cut > 0:
            axis_mask[cut : n_k - cut] = True
        else:
            axis_mask[:] = True  # axis too small to meaningfully trim
        # Broadcast onto state_dims
        view_shape = [1] * len(state_dims)
        view_shape[k] = n_k
        interior_mask = interior_mask & axis_mask.view(view_shape)

    interior_count = interior_mask.sum().item()
    if interior_count == 0:
        return {}

    stats: dict = {}
    for name, (low, high) in ctx.state_ranges.items():
        nv = torch.as_tensor(
            next_state_dict[name], dtype=ctx.dtype, device=ctx.device
        ).broadcast_to(diagnostic_shape)
        outside = (nv < low) | (nv > high)
        per_state = (outside.to(ctx.dtype) * weights).sum(dim=-1)  # state_dims
        # interior_mean excludes outer cells; max stays over the whole mesh.
        interior_sum = (per_state * interior_mask.to(ctx.dtype)).sum().item()
        stats[name] = {
            "max": float(per_state.max().item()),
            "interior_mean": float(interior_sum / interior_count),
            "mean": float(per_state.mean().item()),
        }

    for name, s in stats.items():
        if s["interior_mean"] > threshold:
            low, high = ctx.state_ranges[name]
            warnings.warn(
                f"bellgrid: under the optimal policy, state {name!r} has an "
                f"average of {s['interior_mean']*100:.1f}% of probability "
                f"mass landing outside its grid range [{low}, {high}] across "
                f"the interior of the state mesh (worst-cell escape is "
                f"{s['max']*100:.1f}%). Multilinear interpolation clamps to "
                f"the edge there, which biases V — consider widening the "
                f"state's range.",
                UserWarning,
                stacklevel=3,
            )

    return stats


def _build_queries(
    state: dict,
    state_names: list,
    state_kinds: list,
    axes_for_lookup: list,
    transforms: list,
) -> list:
    """Per-axis lookup queries from the user-supplied state dict.

    Used by both ``_Policy`` and ``_Value`` callables; lives here so it
    doesn't import solver internals."""
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


class _Policy:
    """Callable wrapping the time-indexed optimal action arrays.

    Continuous actions interpolate multilinearly across continuous state
    axes and exact-gather across discrete / markov axes. Discrete actions
    return the optimal index: for single-continuous-state problems we use
    nearest-neighbor; for K>1 or mixed states we multilinear and round.
    """

    def __init__(
        self,
        state_names: list,
        state_kinds: list,
        axes_for_lookup: list,
        transforms: list,
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


class _Value:
    """Callable wrapping the time-indexed value-function arrays."""

    def __init__(
        self,
        state_names: list,
        state_kinds: list,
        axes_for_lookup: list,
        transforms: list,
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
