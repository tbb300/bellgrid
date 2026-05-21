"""bellgrid: GPU-native backward-induction MDP solver."""

__version__ = "0.0.0"

from .problem import ContinuousAction, ContinuousState, DiscreteAction, Problem
from .simulate import simulate
from .solve import solve

__all__ = [
    "Problem",
    "ContinuousState",
    "ContinuousAction",
    "DiscreteAction",
    "solve",
    "simulate",
    # Planned (see docs/api.md), to be added as they land:
    # "DiscreteState", "MarkovChain"
]
