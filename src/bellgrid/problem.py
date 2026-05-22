"""Problem container plus state and action primitives.

Defines: Problem, ContinuousState, DiscreteState, MarkovChain,
ContinuousAction, DiscreteAction.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ContinuousState:
    """Real-valued state variable on ``[low, high]``.

    ``warp`` selects a coordinate transform under which the state grid is
    placed evenly: ``"asinh"`` concentrates points near zero (good for
    wealth or inventory levels), ``"log"`` concentrates them exponentially
    toward the lower end (requires ``low > 0``), ``None`` is uniform. The
    same warp also drives interpolation, so queries are mapped into warped
    space before linear interp — a warp matched to V's curvature gives a
    tighter approximation (e.g. log-utility under a log warp is interp-
    exact up to floating point).
    """

    name: str
    range: tuple[float, float]
    warp: str | Callable | None = None


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
    labels: tuple | None = None

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
    labels: tuple | None = None

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
    """Real-valued action with optional state-dependent bounds.

    ``bounds`` is a 2-tuple ``(lower, upper)``. Each element is either a
    float (a fixed bound) or a string naming a previously declared
    ``ContinuousState`` (a state-dependent bound). For instance,
    ``bounds=(1e-6, "wealth")`` puts the upper bound at the current
    wealth — used for non-borrowing consumption-savings problems.
    """

    name: str
    bounds: tuple = (0.0, 1.0)


@dataclass(frozen=True)
class DiscreteAction:
    name: str
    n: int
    labels: tuple | None = None    # human-readable labels for each category

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
    """A discrete-time Markov decision problem, fully specified.

    Attributes
    ----------
    states
        List of ``ContinuousState`` / ``DiscreteState`` / ``MarkovChain``.
    actions
        List of ``ContinuousAction`` / ``DiscreteAction``.
    transition
        Callable ``(state, action, shock, t) -> {state_name: next_value}``.
        Continuous and ``DiscreteState`` next values come from this dict;
        ``MarkovChain`` next values must NOT (the solver advances them
        internally via the matrix).
    reward
        Scalar callable ``(state, action, shock, t) -> reward``. bellgrid
        maximises expected discounted sum, so negate for cost minimisation.
    shocks
        List of shock objects from ``bellgrid.shocks``. Empty list is fine
        for deterministic problems; ``shock`` in callables is then ``{}``.
    horizon
        ``range`` of ``t`` values for finite-horizon problems, or ``None``
        for infinite-horizon. The solver iterates in reverse through the
        range; ``PolicyIteration`` requires ``None``.
    discount
        Scalar ``β`` or a callable ``(state, t) -> β`` for state-dependent
        discounting (e.g. mortality). Callable discount is documented but
        not yet implemented.
    terminal_reward
        Optional ``(state) -> reward`` evaluated at the post-horizon
        boundary in finite-horizon solves. Useful as a bequest motive or
        for plugging in a closed-form continuation value.

    All names across ``states``, ``actions``, and ``shocks`` must be
    unique. Action bounds that reference a state name are validated
    against the declared states at construction.
    """

    states: list
    actions: list
    transition: Callable
    reward: Callable
    shocks: list
    horizon: range | None
    discount: float | Callable
    terminal_reward: Callable | None = None

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
