"""Continuous uniform innovation shock on ``[low, high]`` with Gauss-Legendre quadrature.

``np.polynomial.legendre.leggauss(n_quad)`` gives nodes and weights for the
integral ``∫_{-1}^{1} f(t) dt ≈ Σ w_i f(t_i)``. We map ``[-1, 1] → [low, high]``
via ``x = (low+high)/2 + (high-low)/2 · t`` and halve the weights so the
total integrates the uniform density ``1/(high-low)`` (i.e. expectations,
not Lebesgue integrals) and the weights sum to 1.

For polynomial integrands of degree ≤ ``2*n_quad - 1`` the quadrature is
**exact**, same as Gauss-Hermite for Normal. Most reward and transition
functions in finance / consumption-savings are smooth enough that
``n_quad = 7`` is plenty.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Uniform:
    """Continuous uniform on ``[low, high]``.

    Parameters
    ----------
    name : str
        Surfaces in ``shock[name]`` in user-supplied callables.
    low, high : float
        Support endpoints. ``high`` must be strictly greater than ``low``.
        Defaults to the standard uniform ``[0, 1]``.
    """

    name: str
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self):
        if not self.name:
            raise ValueError("Uniform requires a name")
        if not (self.high > self.low):
            raise ValueError(
                f"high must be > low, got low={self.low}, high={self.high}"
            )

    def nodes_and_weights(
        self,
        n_quad: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gauss-Legendre quadrature for ``E[f(X)]`` with ``X ~ Uniform(low, high)``.

        Returns
        -------
        nodes : (n_quad,) tensor — abscissae in ``[low, high]``.
        weights : (n_quad,) tensor — weights summing to 1.
        """
        if n_quad < 1:
            raise ValueError(f"n_quad must be >= 1, got {n_quad}")
        raw_nodes, raw_weights = np.polynomial.legendre.leggauss(n_quad)
        mid = 0.5 * (self.high + self.low)
        half = 0.5 * (self.high - self.low)
        nodes = mid + half * raw_nodes
        weights = raw_weights / 2.0  # so they sum to 1
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
        u = torch.rand(n, generator=generator, dtype=dtype, device=device)
        return self.low + (self.high - self.low) * u
