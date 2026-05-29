"""Lean setup for the model-based actor-critic solver.

Deliberately independent of ``solvers/_common.py`` (the grid ``SolveContext``):
the RL solver samples states rather than meshing them, learns ``V``/``π`` as
networks rather than tensors, and computes the shock expectation by the *same*
quadrature the grid solver uses. What it reuses from the rest of the package is
only the public ``Problem`` primitives, the shock ``nodes_and_weights``, and the
warp definitions — so the two solver families share a problem spec without
sharing machinery.

Scope of v1 (enforced here): ``ContinuousState`` + ``DiscreteState`` states,
``ContinuousAction`` actions, any supported shock, finite horizon, scalar or
callable discount, 4- or 5-arg reward. ``MarkovChain`` states, ``DiscreteAction``
actions, and the infinite-horizon (stationary) case raise ``NotImplementedError``
with a pointer to the grid solver — they are planned follow-ups.
"""

import inspect
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..grids.warped import _BUILTIN_WARPS
from ..problem import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    MarkovChain,
    Problem,
)


def _warp_transforms(warp):
    """Return ``(forward, inverse)`` torch transforms and the scalar forward.

    ``forward`` maps physical → warped coordinate (for net-input normalisation),
    ``inverse`` maps warped → physical (for sampling evenly in warped space).
    ``None`` warp is the identity (linear/uniform), matching ``RegularGrid``.
    """
    if warp is None:
        return (lambda x: x), (lambda u: u), (lambda v: float(v))
    if not isinstance(warp, str) or warp not in _BUILTIN_WARPS:
        raise NotImplementedError(
            f"ActorCritic supports warps {sorted(_BUILTIN_WARPS)} or None; got {warp!r}"
        )
    scalar_fwd, tensor_fwd, tensor_inv = _BUILTIN_WARPS[warp]
    if warp == "log":
        # Clamp non-positive inputs before log so the forward transform stays
        # finite on out-of-range next-state queries (mirrors WarpedGrid).
        def fwd(x):
            return torch.log(torch.clamp(x, min=torch.finfo(x.dtype).tiny))
        return fwd, tensor_inv, scalar_fwd
    return tensor_fwd, tensor_inv, scalar_fwd


@dataclass
class RLSetup:
    """Canonicalised, network-agnostic view of a ``Problem`` for the RL solver."""

    problem: Problem
    device: object
    dtype: torch.dtype

    # Canonical state ordering: continuous then discrete (no markov in v1).
    cont_states: list
    disc_states: list
    state_names: list

    cont_actions: list

    # Per continuous state: (low, high, fwd, inv, u_low, u_high) for
    # normalisation (fwd, u_low, u_high) and warped-uniform sampling (inv).
    _cont_meta: dict = field(default_factory=dict)

    # Shock quadrature (tensor product over independent shocks).
    shock_nodes: dict = field(default_factory=dict)
    shock_weights: torch.Tensor = None
    n_q: int = 1

    discount: object = None                 # float or callable(state, t)
    reward_takes_next_state: bool = False

    n_feat: int = 0

    def featurize(self, state: dict) -> torch.Tensor:
        """State dict (each value shape ``S``) → features of shape ``S + (n_feat,)``.

        Continuous states contribute one normalised (warped, scaled to ``[-1, 1]``)
        column each; discrete states contribute a one-hot block. Arbitrary leading
        dims pass through, so this works on flat ``[B]`` query batches and on the
        ``[B, n_q]`` next-state grids alike.
        """
        cols = []
        for s in self.cont_states:
            x = torch.as_tensor(state[s.name], dtype=self.dtype, device=self.device)
            fwd, _inv, _sf = self._cont_meta[s.name][2:5]
            u_low, u_high = self._cont_meta[s.name][5:7]
            u = fwd(x)
            norm = 2.0 * (u - u_low) / (u_high - u_low) - 1.0
            cols.append(norm.unsqueeze(-1))
        for s in self.disc_states:
            idx = torch.as_tensor(state[s.name], dtype=torch.long, device=self.device)
            oh = torch.nn.functional.one_hot(idx, num_classes=s.n).to(self.dtype)
            cols.append(oh)
        return torch.cat(cols, dim=-1)

    def resolve_bounds(self, state: dict) -> dict:
        """Per-action ``(lo, hi)`` tensors broadcast to the state batch shape.

        A bound that names a ``ContinuousState`` resolves to that state's current
        value (state-dependent bounds, e.g. ``consume ≤ wealth``); a float bound
        broadcasts as a constant.
        """
        sample = torch.as_tensor(
            state[self.cont_states[0].name] if self.cont_states else
            state[self.state_names[0]], dtype=self.dtype, device=self.device,
        )
        out = {}
        for a in self.cont_actions:
            lohi = []
            for b in a.bounds:
                if isinstance(b, str):
                    lohi.append(torch.as_tensor(
                        state[b], dtype=self.dtype, device=self.device,
                    ).broadcast_to(sample.shape))
                else:
                    lohi.append(torch.full_like(sample, float(b)))
            out[a.name] = (lohi[0], lohi[1])
        return out

    def sample_states(self, n: int, generator: torch.Generator) -> dict:
        """Draw ``n`` states: continuous uniform in *warped* coords (so denser
        where the warp concentrates, matching the grid philosophy and the region
        the grid solver covers), discrete uniform over categories."""
        state = {}
        for s in self.cont_states:
            _low, _high, _fwd, inv, _sf, u_low, u_high = self._cont_meta[s.name]
            u = u_low + (u_high - u_low) * torch.rand(
                n, generator=generator, dtype=self.dtype, device=self.device,
            )
            state[s.name] = inv(u)
        for s in self.disc_states:
            state[s.name] = torch.randint(
                0, s.n, (n,), generator=generator, dtype=torch.long, device=self.device,
            )
        return state


def build_setup(problem: Problem, n_quad: int, *, device, dtype) -> RLSetup:
    """Validate scope and build an :class:`RLSetup` from a ``Problem``."""
    if problem.horizon is None:
        raise NotImplementedError(
            "ActorCritic is finite-horizon only; use PolicyIteration "
            "(grid) for the infinite-horizon stationary case"
        )
    if any(isinstance(s, MarkovChain) for s in problem.states):
        raise NotImplementedError(
            "ActorCritic does not yet support MarkovChain states; use the "
            "grid solver, or model the regime as a DiscreteState for now"
        )
    if any(not isinstance(s, (ContinuousState, DiscreteState)) for s in problem.states):
        raise NotImplementedError(
            "ActorCritic supports only ContinuousState and DiscreteState states"
        )
    if any(isinstance(a, DiscreteAction) for a in problem.actions):
        raise NotImplementedError(
            "ActorCritic does not yet support DiscreteAction; use the grid solver"
        )
    if any(not isinstance(a, ContinuousAction) for a in problem.actions):
        raise NotImplementedError("ActorCritic supports only ContinuousAction actions")
    if not problem.actions:
        raise ValueError("Problem has no actions")

    cont_states = [s for s in problem.states if isinstance(s, ContinuousState)]
    disc_states = [s for s in problem.states if isinstance(s, DiscreteState)]
    state_names = [s.name for s in cont_states + disc_states]
    cont_actions = [a for a in problem.actions if isinstance(a, ContinuousAction)]

    cont_meta = {}
    n_feat = 0
    for s in cont_states:
        low, high = float(s.range[0]), float(s.range[1])
        fwd, inv, scalar_fwd = _warp_transforms(s.warp)
        if s.warp == "log" and low <= 0:
            raise ValueError(f"log warp requires low > 0 for state {s.name!r}, got {low}")
        u_low, u_high = scalar_fwd(low), scalar_fwd(high)
        cont_meta[s.name] = (low, high, fwd, inv, scalar_fwd, u_low, u_high)
        n_feat += 1
    for s in disc_states:
        n_feat += s.n

    # Shock quadrature — tensor product over independent shocks (same as the
    # grid solver's setup, reproduced here to stay decoupled from _common).
    shock_values: dict = {}
    shock_weights = torch.tensor([1.0], dtype=dtype, device=device)
    n_q = 1
    for shock in problem.shocks:
        raw, w = shock.nodes_and_weights(n_quad, dtype=dtype, device=device)
        this_nodes = raw if isinstance(raw, dict) else {shock.name: raw}
        n_this = w.numel()
        new_values = {
            name: val.unsqueeze(-1).expand(n_q, n_this).reshape(-1).contiguous()
            for name, val in shock_values.items()
        }
        for name, val in this_nodes.items():
            new_values[name] = (
                val.unsqueeze(0).expand(n_q, n_this).reshape(-1).contiguous()
            )
        shock_weights = (
            shock_weights.unsqueeze(-1) * w.unsqueeze(0)
        ).reshape(-1).contiguous()
        shock_values = new_values
        n_q *= n_this

    discount = problem.discount if callable(problem.discount) else float(problem.discount)

    sig = inspect.signature(problem.reward)
    n_pos = len([
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ])
    if n_pos not in (4, 5):
        raise ValueError(
            f"reward must take 4 or 5 positional arguments "
            f"(state, action, shock, t [, next_state]); got {n_pos}"
        )

    return RLSetup(
        problem=problem, device=device, dtype=dtype,
        cont_states=cont_states, disc_states=disc_states, state_names=state_names,
        cont_actions=cont_actions, _cont_meta=cont_meta,
        shock_nodes=shock_values, shock_weights=shock_weights, n_q=n_q,
        discount=discount, reward_takes_next_state=(n_pos == 5), n_feat=n_feat,
    )
