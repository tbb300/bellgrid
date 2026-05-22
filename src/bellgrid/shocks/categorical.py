"""Categorical (finite-support discrete) innovation shock.

Quadrature is **exact**: the K values are the quadrature nodes, the K
probabilities are the weights. ``n_quad`` is ignored — categorical
quadrature doesn't need approximation.

Useful for any iid discrete process whose state needn't be tracked
across periods: demand levels (low / medium / high), Bernoulli events
(via ``values=(0., 1.)`` and ``probabilities=(1-p, p)``), contingent
payouts, etc. For state-dependent discrete transitions where the next-
period probabilities depend on the current category, use ``MarkovChain``
instead (declared as a state, not a shock).

``shock[name]`` in user code is a **float tensor** of the value at each
quadrature node (or sample). If you need integer category indices,
pass ``values=(0., 1., 2., ...)`` and cast inside ``transition`` /
``reward``.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Categorical:
    """Iid discrete shock with finite support.

    Parameters
    ----------
    name : str
        Surfaces in ``shock[name]`` in user-supplied callables.
    values : array-like of K floats
        The K possible values the shock can take.
    probabilities : array-like of K floats
        Non-negative weights summing to 1.
    """

    name: str
    values: object = None
    probabilities: object = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Categorical requires a name")
        if self.values is None:
            raise ValueError("Categorical requires values")
        if self.probabilities is None:
            raise ValueError("Categorical requires probabilities")
        values = np.asarray(self.values, dtype=np.float64)
        probs = np.asarray(self.probabilities, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(
                f"values must be 1-D, got shape {values.shape}"
            )
        if probs.ndim != 1:
            raise ValueError(
                f"probabilities must be 1-D, got shape {probs.shape}"
            )
        if values.shape != probs.shape:
            raise ValueError(
                f"values and probabilities must have the same length; "
                f"got {values.size} and {probs.size}"
            )
        if values.size < 1:
            raise ValueError("Categorical requires at least one value")
        if (probs < 0).any():
            raise ValueError("probabilities must be non-negative")
        if not math.isclose(float(probs.sum()), 1.0, abs_tol=1e-10):
            raise ValueError(
                f"probabilities must sum to 1, got {probs.sum()}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "probabilities", probs)

    @property
    def K(self) -> int:
        return self.values.size

    def nodes_and_weights(
        self,
        n_quad: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact quadrature — ``n_quad`` is ignored.

        Returns
        -------
        nodes : (K,) float tensor — the K possible values.
        weights : (K,) float tensor — the K probabilities (sum to 1).
        """
        return (
            torch.as_tensor(self.values, dtype=dtype, device=device),
            torch.as_tensor(self.probabilities, dtype=dtype, device=device),
        )

    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Draw ``n`` iid samples; returns a (n,) float tensor of values."""
        probs = torch.as_tensor(self.probabilities, dtype=dtype, device=device)
        values = torch.as_tensor(self.values, dtype=dtype, device=device)
        idx = torch.multinomial(probs, n, replacement=True, generator=generator)
        return values[idx]
