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

    def _to_state_shape(val: torch.Tensor) -> torch.Tensor:
        """Broadcast an action value tensor to the joint state-action shape.

        Accepts scalars, 1-D values of shape ``(N_a,)`` (static bound), or
        any tensor already broadcastable to ``state_dims_tup + (N_a,)``
        (state-dependent bounds, which arrive shaped via ``_resolve_bound``).
        """
        target = state_dims_tup + (N_a,)
        if val.ndim == 0:
            val = val.view((1,) * (K + 1))
        elif val.ndim == 1 and val.shape[0] == N_a:
            val = val.view((1,) * K + (N_a,))
        return val.broadcast_to(target).contiguous()

    action_tensors: dict = {}
    action_kinds: dict = {}
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
    )


def terminal_value(ctx: SolveContext) -> torch.Tensor:
    """Evaluate ``problem.terminal_reward`` on the joint state mesh (or zeros)."""
    if ctx.problem.terminal_reward is None:
        return torch.zeros(ctx.state_dims_tup, dtype=ctx.dtype, device=ctx.device)
    tr = ctx.problem.terminal_reward(ctx.state_meshes_dict)
    return torch.as_tensor(tr, dtype=ctx.dtype, device=ctx.device).broadcast_to(
        ctx.state_dims_tup
    ).contiguous()


def _bellman_partial(
    ctx: SolveContext,
    V_next: torch.Tensor,
    t,
    q_start: int,
    q_end: int,
    discount,
) -> torch.Tensor:
    """Contribution of shock nodes ``[q_start, q_end)`` to ``Σ_q (r + γ V) · w``.

    Returns a ``state_dims + (N_a,)``-shaped tensor: the partial Bellman
    integrand summed over the slice of shock nodes. Callers concatenate
    the partials by summing across all shock-slice calls, then take the
    max over the action axis. ``discount`` is either a scalar or a tensor
    broadcastable to ``full_shape`` (precomputed once per Bellman step).
    """
    n_q = q_end - q_start
    chunk_shape = ctx.state_dims_tup + (ctx.N_a, n_q)

    # Slice the expanded views along the shock axis. Each view stays a
    # view (no allocation) since slicing along an existing axis is just
    # an offset + size change.
    state_b = {name: x[..., q_start:q_end] for name, x in ctx.state_b_dict.items()}
    action_b = {name: x[..., q_start:q_end] for name, x in ctx.action_b.items()}
    shock_b = {name: x[..., q_start:q_end] for name, x in ctx.shock_dict_b.items()}
    weights_b = ctx.weights_b[..., q_start:q_end]
    if isinstance(discount, torch.Tensor):
        discount_chunk = discount[..., q_start:q_end]
    else:
        discount_chunk = discount

    # Always evaluate the transition first — its output may be needed by a
    # next-state-aware reward.
    next_state_dict = ctx.problem.transition(state_b, action_b, shock_b, t)

    # Validate the return dict's keys up-front so the user sees one
    # aggregated error rather than a per-axis "missing key" deep in the
    # multilinear-query loop below. Cheap; ``next_state_dict`` is already
    # in hand.
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

    if ctx.reward_takes_next_state:
        r = ctx.problem.reward(state_b, action_b, shock_b, t, next_state_dict)
    else:
        r = ctx.problem.reward(state_b, action_b, shock_b, t)
    r = torch.as_tensor(r, dtype=ctx.dtype, device=ctx.device).broadcast_to(
        chunk_shape
    ).contiguous()

    # Build the lookup tensor shape. With K_mc markov chains, the lookup
    # has one extra "kept" axis per chain, appended in chain order.
    lookup_shape = chunk_shape + tuple(ctx.n_ms)

    queries = []
    mc_idx = 0  # which markov chain (in canonical order) we're on
    for name, kind, transform in zip(
        ctx.state_names, ctx.state_kinds, ctx.transforms
    ):
        if kind == "continuous":
            nv = torch.as_tensor(
                next_state_dict[name], dtype=ctx.dtype, device=ctx.device
            )
            nv = nv.broadcast_to(chunk_shape).contiguous()
            u_next = transform(nv)
            if ctx.K_mc >= 1:
                # Extend by K_mc trailing 1-dims, then expand to lookup_shape.
                for _ in range(ctx.K_mc):
                    u_next = u_next.unsqueeze(-1)
                u_next = u_next.expand(lookup_shape).contiguous()
            queries.append(u_next)
        elif kind == "discrete":
            nv = torch.as_tensor(
                next_state_dict[name], dtype=torch.long, device=ctx.device
            )
            nv = nv.broadcast_to(chunk_shape).contiguous()
            if ctx.K_mc >= 1:
                for _ in range(ctx.K_mc):
                    nv = nv.unsqueeze(-1)
                nv = nv.expand(lookup_shape).contiguous()
            queries.append(nv)
        else:  # markov — solver-controlled
            n_mk = ctx.n_ms[mc_idx]
            arange = torch.arange(n_mk, dtype=torch.long, device=ctx.device)
            # This chain's kept axis is at position len(chunk_shape) + mc_idx
            # in the lookup shape; all other axes are size 1 for this query.
            view_arange = [1] * len(lookup_shape)
            view_arange[len(chunk_shape) + mc_idx] = n_mk
            arange_b = arange.view(view_arange).expand(lookup_shape).contiguous()
            queries.append(arange_b)
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


def bellman_step(ctx: SolveContext, V_next: torch.Tensor, t) -> tuple[torch.Tensor, dict]:
    """One Bellman update.

    Returns ``(V_now, policy_now)`` where ``V_now`` has shape
    ``state_dims_tup`` and ``policy_now`` is a dict of optimal action
    tensors with the same shape.

    The shock axis is chunked so that no intermediate tensor exceeds
    roughly ``ctx.chunk_size`` elements. For small problems the chunk
    spans the entire shock axis and the loop is a single iteration; for
    larger problems we accumulate the Bellman expectation across chunks.
    """
    # Pre-compute the discount once per Bellman step — same across all
    # shock-slice chunks. Scalar discounts stay scalar; callable discounts
    # are materialised to ``full_shape`` so chunked slicing works
    # uniformly.
    if callable(ctx.discount):
        disc_raw = ctx.discount(ctx.state_b_dict, t)
        discount = torch.as_tensor(
            disc_raw, dtype=ctx.dtype, device=ctx.device
        ).broadcast_to(ctx.full_shape).contiguous()
    else:
        discount = ctx.discount

    state_count = (
        math.prod(ctx.state_dims_tup) if ctx.state_dims_tup else 1
    )
    elements_per_shock = state_count * ctx.N_a
    chunk_n_q = max(1, ctx.chunk_size // max(elements_per_shock, 1))

    if chunk_n_q >= ctx.N_q:
        bellman = _bellman_partial(ctx, V_next, t, 0, ctx.N_q, discount)
    else:
        bellman = torch.zeros(
            ctx.state_dims_tup + (ctx.N_a,),
            dtype=ctx.dtype, device=ctx.device,
        )
        for q_start in range(0, ctx.N_q, chunk_n_q):
            q_end = min(q_start + chunk_n_q, ctx.N_q)
            bellman = bellman + _bellman_partial(
                ctx, V_next, t, q_start, q_end, discount,
            )

    V_now, argmax = bellman.max(dim=-1)
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
