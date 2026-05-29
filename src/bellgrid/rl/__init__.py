"""Model-based actor-critic solver (neural, finite-horizon).

A function-approximation counterpart to the grid solvers that shares the same
``Problem`` spec and returns the same ``(policy, value)`` callables, so it can be
certified against the grid solver wherever both run. See ``ActorCritic``.
"""

from .solver import ActorCritic

__all__ = ["ActorCritic"]
