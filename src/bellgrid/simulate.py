"""Forward Monte Carlo simulator.

Uses the *same* ``transition`` and ``reward`` callables as the solver, so
the simulator and solver cannot drift apart.

Scope: any number of ``ContinuousState`` / ``DiscreteState`` / at most one
``MarkovChain``, any number of ``ContinuousAction`` / ``DiscreteAction``,
zero or one ``Normal``/``Lognormal``/``MultivariateNormal`` shock, finite
horizon, scalar discount.

For ``MarkovChain`` states: the simulator samples each path's next category
from row ``P[current[i], :]`` via ``torch.multinomial``. The user's
``transition`` callable must NOT return an entry for a markov chain (same
rule as the solver).
"""

from typing import Callable, Optional

import torch

from .problem import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    MarkovChain,
    Problem,
)
from .shocks.jump import Jump
from .shocks.lognormal import Lognormal
from .shocks.multivariate_normal import MultivariateNormal
from .shocks.normal import Normal

_SUPPORTED_SHOCKS = (Normal, Lognormal, MultivariateNormal, Jump)
_SUPPORTED_STATES = (ContinuousState, DiscreteState, MarkovChain)
_SUPPORTED_ACTIONS = (ContinuousAction, DiscreteAction)


def simulate(
    *,
    policy: Callable,
    problem: Problem,
    n: int,
    initial_state: dict,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
) -> dict:
    """Simulate ``n`` forward paths under ``policy``.

    Parameters
    ----------
    policy
        Callable ``policy(state, t) -> dict``. Receives a dict of batched
        state tensors and returns a dict of batched action tensors.
    problem
        The same ``Problem`` passed to ``solve()``; the simulator uses its
        ``transition``, ``reward``, ``discount``, ``horizon``, and ``shocks``.
    n
        Number of paths to draw.
    initial_state
        Dict of scalar state values used for every path at the first
        period in ``horizon``. Continuous values are coerced to ``dtype``;
        discrete and markov values to ``torch.long``.
    seed
        Optional RNG seed for reproducibility.
    dtype, device
        Storage for continuous-state path tensors. Discrete and markov
        path tensors are always ``torch.long``.

    Returns
    -------
    paths : dict
        - One ``(n, T)`` tensor per declared state name (the state at each
          ``t`` in ``horizon``). Continuous states use ``dtype``; discrete
          and markov states use ``torch.long``.
        - One ``(n, T)`` tensor per declared action name (continuous in
          ``dtype``; discrete in ``torch.long``).
        - ``paths["reward"]``: per-step realized reward, shape ``(n, T)``.
        - ``paths["discounted_total"]``: per-path sum
          ``sum_i discount**i * reward[i]``, shape ``(n,)``.
    """
    # ---- scope checks --------------------------------------------------
    if any(not isinstance(s, _SUPPORTED_STATES) for s in problem.states):
        raise NotImplementedError(
            f"simulate supports only {[c.__name__ for c in _SUPPORTED_STATES]}"
        )
    if any(not isinstance(a, _SUPPORTED_ACTIONS) for a in problem.actions):
        raise NotImplementedError(
            f"simulate supports only {[c.__name__ for c in _SUPPORTED_ACTIONS]}"
        )
    if any(not isinstance(s, _SUPPORTED_SHOCKS) for s in problem.shocks):
        raise NotImplementedError(
            f"simulate supports only {[c.__name__ for c in _SUPPORTED_SHOCKS]} shocks"
        )
    mc_states = [s for s in problem.states if isinstance(s, MarkovChain)]
    if len(mc_states) > 1:
        raise NotImplementedError(
            "simulate supports at most one MarkovChain per problem"
        )
    if problem.horizon is None:
        raise NotImplementedError("simulate does not support infinite horizon yet")
    if callable(problem.discount):
        raise NotImplementedError("simulate does not support callable discount yet")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    horizon = list(problem.horizon)
    T = len(horizon)

    state_names = [s.name for s in problem.states]
    action_names = [a.name for a in problem.actions]

    state_kinds = {}
    for s in problem.states:
        if isinstance(s, ContinuousState):
            state_kinds[s.name] = "continuous"
        elif isinstance(s, DiscreteState):
            state_kinds[s.name] = "discrete"
        else:
            state_kinds[s.name] = "markov"
    action_kinds = {
        a.name: "continuous" if isinstance(a, ContinuousAction) else "discrete"
        for a in problem.actions
    }
    state_dtypes = {
        name: (dtype if kind == "continuous" else torch.long)
        for name, kind in state_kinds.items()
    }
    action_dtypes = {
        name: (dtype if kind == "continuous" else torch.long)
        for name, kind in action_kinds.items()
    }

    for name in state_names:
        if name not in initial_state:
            raise ValueError(f"initial_state missing entry for state {name!r}")

    # ---- RNG -----------------------------------------------------------
    gen = torch.Generator(device="cpu" if str(device) == "cpu" else device)
    if seed is not None:
        gen.manual_seed(int(seed))

    # ---- storage -------------------------------------------------------
    paths: dict[str, torch.Tensor] = {}
    for name in state_names:
        paths[name] = torch.empty((n, T), dtype=state_dtypes[name], device=device)
    for name in action_names:
        paths[name] = torch.empty((n, T), dtype=action_dtypes[name], device=device)
    paths["reward"] = torch.empty((n, T), dtype=dtype, device=device)
    paths["discounted_total"] = torch.zeros((n,), dtype=dtype, device=device)

    # ---- initial state -------------------------------------------------
    state = {}
    for name in state_names:
        raw = initial_state[name]
        if state_kinds[name] == "continuous":
            state[name] = torch.full(
                (n,), float(raw), dtype=dtype, device=device
            )
        else:
            state[name] = torch.full(
                (n,), int(raw), dtype=torch.long, device=device
            )

    # Pre-cache markov chain matrix on device for sampling.
    mc_matrices = {
        s.name: torch.as_tensor(s.matrix, dtype=dtype, device=device)
        for s in mc_states
    }

    discount = float(problem.discount)
    disc_factor = 1.0

    # ---- forward sweep -------------------------------------------------
    for i, t in enumerate(horizon):
        for name in state_names:
            paths[name][:, i] = state[name]

        action = policy(state, t)
        for name in action_names:
            paths[name][:, i] = action[name].to(action_dtypes[name])

        # Sample shocks
        shock = {}
        for s in problem.shocks:
            sampled = s.sample(n, generator=gen, dtype=dtype, device=device)
            if isinstance(sampled, dict):
                shock.update(sampled)
            else:
                shock[s.name] = sampled

        r = problem.reward(state, action, shock, t)
        r = torch.as_tensor(r, dtype=dtype, device=device).broadcast_to((n,)).contiguous()
        paths["reward"][:, i] = r
        paths["discounted_total"] += disc_factor * r
        disc_factor *= discount

        next_state_user = problem.transition(state, action, shock, t)

        new_state = {}
        for name in state_names:
            kind = state_kinds[name]
            if kind in ("continuous", "discrete"):
                if name not in next_state_user:
                    raise ValueError(
                        f"transition return dict missing state key {name!r}"
                    )
                tgt_dtype = state_dtypes[name]
                v = torch.as_tensor(
                    next_state_user[name], dtype=tgt_dtype, device=device
                )
                new_state[name] = v.broadcast_to((n,)).contiguous()
            else:  # markov: solver/simulator-controlled
                if name in next_state_user:
                    raise ValueError(
                        f"transition must not return MarkovChain state {name!r} "
                        "(advanced via its transition matrix)"
                    )
                matrix = mc_matrices[name]  # (n_m, n_m)
                # Per-path categorical sample from the row matching current state
                current = state[name]  # (n,) long
                row_probs = matrix[current]  # (n, n_m)
                sampled = torch.multinomial(
                    row_probs, num_samples=1, generator=gen
                ).squeeze(-1)
                new_state[name] = sampled

        state = new_state

    return paths
