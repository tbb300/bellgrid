"""Problem container plus state and action primitives.

Defines: Problem, ContinuousState, ContinuousAction. DiscreteState,
DiscreteAction, and MarkovChain are planned (see docs/api.md) but not
yet implemented.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Union


@dataclass(frozen=True)
class ContinuousState:
    name: str
    range: tuple[float, float]
    warp: Optional[Union[str, Callable]] = None


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
