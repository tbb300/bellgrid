"""Problem container plus state and action primitives.

Defines: Problem, ContinuousState, DiscreteState, MarkovChain,
ContinuousAction, DiscreteAction.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np


@dataclass(frozen=True)
class ContinuousState:
    name: str
    range: tuple[float, float]
    warp: Optional[Union[str, Callable]] = None


@dataclass(frozen=True)
class DiscreteState:
    """Finite-state variable with ``n`` integer values.

    The user writes the transition dynamics for this state in their
    ``transition`` callable (the solver doesn't supply built-in dynamics).
    For a discrete state with a built-in row-stochastic transition matrix,
    use ``MarkovChain`` instead.
    """

    name: str
    n: int
    labels: Optional[tuple] = None

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"DiscreteState requires n >= 1, got {self.n}")
        if self.labels is not None and len(self.labels) != self.n:
            raise ValueError(
                f"DiscreteState labels must have length n={self.n}, "
                f"got {len(self.labels)}"
            )


@dataclass(frozen=True)
class MarkovChain:
    """Discrete state with a built-in row-stochastic transition matrix.

    ``matrix[i, j] = Pr(next = j | current = i)``; rows sum to 1. The number
    of categories is inferred from ``matrix.shape[0]``. The solver advances
    this state internally — the user's ``transition`` callable must NOT
    return an entry for it.
    """

    name: str
    matrix: object = None  # np.ndarray after __post_init__
    labels: Optional[tuple] = None

    def __post_init__(self):
        if self.matrix is None:
            raise ValueError("MarkovChain requires a transition matrix")
        mat = np.asarray(self.matrix, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"MarkovChain matrix must be square 2-D; got shape {mat.shape}"
            )
        if (mat < 0).any():
            raise ValueError("MarkovChain matrix entries must be non-negative")
        if not np.allclose(mat.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError(
                "MarkovChain matrix must be row-stochastic (rows sum to 1)"
            )
        object.__setattr__(self, "matrix", mat)
        if self.labels is not None and len(self.labels) != mat.shape[0]:
            raise ValueError(
                f"MarkovChain labels must have length matching matrix "
                f"({mat.shape[0]}); got {len(self.labels)}"
            )

    @property
    def n(self) -> int:
        return self.matrix.shape[0]


@dataclass(frozen=True)
class ContinuousAction:
    name: str
    bounds: tuple = (0.0, 1.0)


@dataclass(frozen=True)
class DiscreteAction:
    name: str
    n: int
    labels: Optional[tuple] = None  # human-readable labels for each category

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"DiscreteAction requires n >= 1, got {self.n}")
        if self.labels is not None and len(self.labels) != self.n:
            raise ValueError(
                f"DiscreteAction labels must have length n={self.n}, "
                f"got {len(self.labels)}"
            )


@dataclass(frozen=True)
class Problem:
    states: list
    actions: list
    transition: Callable
    reward: Callable
    shocks: list
    horizon: Optional[range]
    discount: Union[float, Callable]
    terminal_reward: Optional[Callable] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        state_names = [s.name for s in self.states]
        action_names = [a.name for a in self.actions]
        shock_names = [
            s.name for s in self.shocks if getattr(s, "name", None) is not None
        ]

        all_names = state_names + action_names + shock_names
        if len(all_names) != len(set(all_names)):
            dupes = sorted({n for n, c in Counter(all_names).items() if c > 1})
            raise ValueError(
                f"Name collision across states/actions/shocks: {dupes}"
            )

        state_name_set = set(state_names)
        for action in self.actions:
            if isinstance(action, ContinuousAction):
                for b in action.bounds:
                    if isinstance(b, str) and b not in state_name_set:
                        raise ValueError(
                            f"Action {action.name!r} bounds reference "
                            f"undeclared state {b!r}"
                        )
