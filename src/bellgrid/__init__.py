"""bellgrid: GPU-native backward-induction MDP solver."""

__version__ = "0.0.0"

from .problem import ContinuousAction, ContinuousState, Problem
from .solve import solve

__all__ = [
    "Problem",
    "ContinuousState",
    "ContinuousAction",
    "solve",
    # Planned (see docs/api.md), to be added as they land:
    # "DiscreteState", "MarkovChain", "DiscreteAction", "simulate"
]
