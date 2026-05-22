"""Forward Monte Carlo simulator.

Uses the *same* ``transition``, ``reward``, and ``discount`` as the solver,
so the simulator and solver cannot drift apart.

Scope: any number of ``ContinuousState`` / ``DiscreteState`` /
``MarkovChain``, any number of ``ContinuousAction`` / ``DiscreteAction``,
any number of ``Normal`` / ``Lognormal`` / ``MultivariateNormal`` /
``Jump`` / ``Categorical`` / ``Uniform`` shocks, finite **or infinite**
horizon (infinite requires the caller to specify ``n_periods``), scalar
**or callable** discount, and 4-arg or 5-arg reward (the 5-arg form
receives the next-state dict, matching the solver's convention).

For ``MarkovChain`` states: the simulator samples each path's next
category from row ``P[current[i], :]`` via ``torch.multinomial``. The
user's ``transition`` callable must NOT return an entry for a markov
chain (same rule as the solver).
"""

import inspect
from typing import Callable

import torch

from .problem import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    MarkovChain,
    Problem,
)
from .shocks.categorical import Categorical
from .shocks.jump import Jump
from .shocks.lognormal import Lognormal
from .shocks.multivariate_normal import MultivariateNormal
from .shocks.normal import Normal
from .shocks.uniform import Uniform

_SUPPORTED_SHOCKS = (Normal, Lognormal, MultivariateNormal, Jump, Categorical, Uniform)
_SUPPORTED_STATES = (ContinuousState, DiscreteState, MarkovChain)
_SUPPORTED_ACTIONS = (ContinuousAction, DiscreteAction)


def simulate(
    *,
    policy: Callable,
    problem: Problem,
    n: int,
    initial_state: dict,
    n_periods: int | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
) -> dict:
    """Simulate ``n`` forward paths under ``policy``.

    Parameters
    ----------
    policy
        Callable ``policy(state, t) -> dict``. Receives a dict of batched
        state tensors and returns a dict of batched action tensors. For
        infinite-horizon (``problem.horizon=None``) ``t`` is ``None`` at
        every step.
    problem
        The same ``Problem`` passed to ``solve()``; the simulator uses its
        ``transition``, ``reward``, ``discount``, ``horizon``, and ``shocks``.
    n
        Number of paths to draw.
    initial_state
        Dict of scalar state values used for every path at the first
        period. Continuous values are coerced to ``dtype``; discrete and
        markov values to ``torch.long``.
    n_periods
        How many periods to simulate. Required for infinite-horizon
        problems (``problem.horizon=None``); for finite-horizon problems
        defaults to ``len(problem.horizon)`` and must equal that value
        if explicitly specified.
    seed
        Optional RNG seed for reproducibility.
    dtype, device
        Storage for continuous-state path tensors. Discrete and markov
        path tensors are always ``torch.long``.

    Returns
    -------
    paths : dict
        - One ``(n, T)`` tensor per declared state name (the state at each
          period). Continuous states use ``dtype``; discrete and markov
          states use ``torch.long``.
        - One ``(n, T)`` tensor per declared action name (continuous in
          ``dtype``; discrete in ``torch.long``).
        - ``paths["reward"]``: per-step realized reward, shape ``(n, T)``.
        - ``paths["discounted_total"]``: per-path sum of discounted rewards,
          shape ``(n,)``. Discounting respects callable discount if
          ``problem.discount`` is one.
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

    # Horizon handling: finite picks T from problem.horizon (or matches
    # n_periods if given); infinite requires n_periods explicitly and
    # uses t=None at every step (stationary-policy contract).
    if problem.horizon is None:
        if n_periods is None:
            raise ValueError(
                "simulate with horizon=None requires n_periods (the number "
                "of periods to simulate the stationary policy)"
            )
        t_values: list = [None] * int(n_periods)
        T = int(n_periods)
    else:
        horizon_list = list(problem.horizon)
        if n_periods is not None and n_periods != len(horizon_list):
            raise ValueError(
                f"n_periods={n_periods} does not match horizon length "
                f"{len(horizon_list)}; pass n_periods=None or matching value"
            )
        t_values = horizon_list
        T = len(t_values)

    # Detect reward signature: 4 args (state, action, shock, t) or 5
    # (state, action, shock, t, next_state). Matches the solver's
    # detection so a 5-arg reward written for solve() works in simulate()
    # without changes.
    sig = inspect.signature(problem.reward)
    n_reward_params = len([
        p for p in sig.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ])
    if n_reward_params == 4:
        reward_takes_next_state = False
    elif n_reward_params == 5:
        reward_takes_next_state = True
    else:
        raise ValueError(
            f"reward must take 4 or 5 positional arguments "
            f"(state, action, shock, t [, next_state]); got {n_reward_params}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    mc_matrices = {
        s.name: torch.as_tensor(s.matrix, dtype=dtype, device=device)
        for s in mc_states
    }

    discount_is_callable = callable(problem.discount)
    if not discount_is_callable:
        discount_scalar = float(problem.discount)

    # Per-path discount accumulator. Held as a (n,) tensor so the same
    # code path handles scalar and callable discount.
    disc_factor = torch.ones((n,), dtype=dtype, device=device)

    # ---- forward sweep -------------------------------------------------
    for i, t in enumerate(t_values):
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

        # Run transition first so reward (if it takes 5 args) can see
        # next_state.
        next_state_user = problem.transition(state, action, shock, t)

        if reward_takes_next_state:
            r = problem.reward(state, action, shock, t, next_state_user)
        else:
            r = problem.reward(state, action, shock, t)
        r = torch.as_tensor(r, dtype=dtype, device=device).broadcast_to((n,)).contiguous()
        paths["reward"][:, i] = r
        paths["discounted_total"] += disc_factor * r

        # Update the running discount factor *after* booking this period's
        # reward — the factor applied next period is β_now (or
        # β_now(state, t) for callable discount, evaluated on the
        # current pre-transition state).
        if discount_is_callable:
            d = problem.discount(state, t)
            d = torch.as_tensor(d, dtype=dtype, device=device).broadcast_to((n,))
            disc_factor = disc_factor * d
        else:
            disc_factor = disc_factor * discount_scalar

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
            else:  # markov: simulator-controlled
                if name in next_state_user:
                    raise ValueError(
                        f"transition must not return MarkovChain state {name!r} "
                        "(advanced via its transition matrix)"
                    )
                matrix = mc_matrices[name]
                current = state[name]
                row_probs = matrix[current]
                sampled = torch.multinomial(
                    row_probs, num_samples=1, generator=gen
                ).squeeze(-1)
                new_state[name] = sampled

        state = new_state

    return paths
