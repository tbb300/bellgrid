"""Lognormal innovation shock with Gauss-Hermite quadrature.

``X = exp(mu + sigma * Z)`` where ``Z ~ N(0, 1)``. ``mu`` and ``sigma``
parameterise the *underlying normal* (i.e. ``log(X) ~ N(mu, sigma)``),
matching the convention documented in ``docs/api.md``.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Lognormal:
    """Univariate lognormal innovation."""

    name: str | None = None
    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self):
        if not (self.sigma > 0):
            raise ValueError(f"Lognormal sigma must be positive, got {self.sigma}")

    def nodes_and_weights(
        self,
        n_quad: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gauss-Hermite quadrature for ``E[f(X)]`` with ``X ~ Lognormal``.

        Nodes are ``exp(mu + sigma * z_i)`` where ``z_i`` are the standard
        normal GH nodes; weights are unchanged from the standard-normal
        quadrature (they sum to 1).
        """
        if n_quad < 1:
            raise ValueError(f"n_quad must be >= 1, got {n_quad}")

        raw_nodes, raw_weights = np.polynomial.hermite_e.hermegauss(n_quad)
        weights = raw_weights / math.sqrt(2.0 * math.pi)
        underlying = self.mu + self.sigma * raw_nodes
        nodes = np.exp(underlying)

        return (
            torch.as_tensor(nodes, dtype=dtype, device=device),
            torch.as_tensor(weights, dtype=dtype, device=device),
        )

    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        z = torch.randn(n, generator=generator, dtype=dtype, device=device)
        return torch.exp(self.mu + self.sigma * z)
