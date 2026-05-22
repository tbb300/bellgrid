"""Normal (Gaussian) innovation shock with Gauss-Hermite quadrature."""

import math
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class Normal:
    """Univariate normal innovation: ``X ~ N(0, sigma^2)``.

    `sigma=1` (the default) is the standard normal — used as the canonical
    standardized form when the user does their own scaling in `transition`.
    Names are optional so a `Normal` can be passed as an inner distribution
    (e.g., `Jump.size_dist`) without being surfaced through `shock[...]`.
    """

    name: str | None = None
    sigma: float = 1.0

    def __post_init__(self):
        if not (self.sigma > 0):
            raise ValueError(f"Normal sigma must be positive, got {self.sigma}")

    def nodes_and_weights(
        self,
        n_quad: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Probabilists' Gauss-Hermite quadrature for ``E[f(X)]`` with ``X ~ N(0, sigma^2)``.

        Returns
        -------
        nodes : (n_quad,) tensor
            Quadrature abscissae, scaled by ``sigma``.
        weights : (n_quad,) tensor
            Normalized weights summing to 1.
        """
        if n_quad < 1:
            raise ValueError(f"n_quad must be >= 1, got {n_quad}")

        # numpy's hermegauss gives nodes/weights for the probabilists' Hermite
        # measure ∫ f(x) exp(-x^2/2) dx, so weights sum to sqrt(2*pi).
        # Divide by sqrt(2*pi) to make them a proper probability quadrature.
        raw_nodes, raw_weights = np.polynomial.hermite_e.hermegauss(n_quad)
        weights = raw_weights / math.sqrt(2.0 * math.pi)
        nodes = raw_nodes * self.sigma

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
        return z * self.sigma
