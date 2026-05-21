"""Forward Monte Carlo simulator.

Uses the *same* ``transition`` and ``reward`` callables as the solver, so
the simulator and solver cannot drift apart.

Tracer-slice scope: one ``ContinuousState``, any number of
``ContinuousAction``s, zero or one ``Normal`` shock, finite horizon,
scalar discount. The signature is the multi-shock/multi-state one so
broadening the scope doesn't change the API.
"""

from typing import Callable, Optional

import torch

from .problem import ContinuousAction, ContinuousState, Problem
from .shocks.normal import Normal


def simulate(
    *,
    policy: Callable,
    problem: Problem,
    n: int,
    initial_state: dict,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
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
        period in ``horizon``.
    seed
        Optional RNG seed for reproducibility.
    dtype, device
        Storage for path tensors.

    Returns
    -------
    paths : dict
        - One ``(n, T)`` tensor per declared state name (the state at each
          ``t`` in ``horizon``).
        - One ``(n, T)`` tensor per declared action name.
        - ``paths["reward"]``: per-step realized reward, shape ``(n, T)``.
        - ``paths["discounted_total"]``: per-path sum
          ``sum_i discount**i * reward[i]``, shape ``(n,)``.
    """
    # ---- scope checks --------------------------------------------------
    cont_states = [s for s in problem.states if isinstance(s, ContinuousState)]
    if len(cont_states) != len(problem.states):
        raise NotImplementedError("simulate currently supports only ContinuousState")
    cont_actions = [a for a in problem.actions if isinstance(a, ContinuousAction)]
    if len(cont_actions) != len(problem.actions):
        raise NotImplementedError("simulate currently supports only ContinuousAction")
    if any(not isinstance(s, Normal) for s in problem.shocks):
        raise NotImplementedError("simulate currently supports only Normal shocks")
    if problem.horizon is None:
        raise NotImplementedError("simulate does not support infinite horizon yet")
    if callable(problem.discount):
        raise NotImplementedError("simulate does not support callable discount yet")

    horizon = list(problem.horizon)
    T = len(horizon)

    state_names = [s.name for s in cont_states]
    action_names = [a.name for a in cont_actions]
    for name in state_names:
        if name not in initial_state:
            raise ValueError(f"initial_state missing entry for state {name!r}")

    # ---- RNG -----------------------------------------------------------
    gen = torch.Generator(device="cpu" if str(device) == "cpu" else device)
    if seed is not None:
        gen.manual_seed(int(seed))

    # ---- storage -------------------------------------------------------
    paths: dict[str, torch.Tensor] = {
        name: torch.empty((n, T), dtype=dtype, device=device) for name in state_names
    }
    for name in action_names:
        paths[name] = torch.empty((n, T), dtype=dtype, device=device)
    paths["reward"] = torch.empty((n, T), dtype=dtype, device=device)
    paths["discounted_total"] = torch.zeros((n,), dtype=dtype, device=device)

    # ---- initial state --------------------------------------------------
    state = {
        name: torch.full(
            (n,), float(initial_state[name]), dtype=dtype, device=device
        )
        for name in state_names
    }

    discount = float(problem.discount)
    disc_factor = 1.0  # discount ** i at step i

    # ---- forward sweep --------------------------------------------------
    for i, t in enumerate(horizon):
        for name in state_names:
            paths[name][:, i] = state[name]

        action = policy(state, t)
        for name in action_names:
            paths[name][:, i] = action[name]

        shock = {}
        for s in problem.shocks:
            z = torch.randn(n, generator=gen, dtype=dtype, device=device)
            shock[s.name] = z * s.sigma

        r = problem.reward(state, action, shock, t)
        r = torch.as_tensor(r, dtype=dtype, device=device).broadcast_to((n,)).contiguous()
        paths["reward"][:, i] = r
        paths["discounted_total"] += disc_factor * r
        disc_factor *= discount

        next_state = problem.transition(state, action, shock, t)
        for name in state_names:
            if name not in next_state:
                raise ValueError(
                    f"transition return dict missing state key {name!r}"
                )
            state[name] = (
                torch.as_tensor(next_state[name], dtype=dtype, device=device)
                .broadcast_to((n,))
                .contiguous()
            )

    return paths
