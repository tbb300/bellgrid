"""Jump / Poisson shock with a normally-distributed log-jump magnitude.

Discrete-time approximation: in each period at most one jump occurs (the
Bernoulli approximation to a Poisson process). The per-period jump
probability is ``p = 1 - exp(-intensity)``; for the small intensities
typical of jump-diffusion option pricing (≲ 0.1) this is within ~1%
of the exact Poisson and gives a clean quadrature: ``1 + n_quad`` joint
nodes (one no-jump node + Gauss-Hermite nodes for the log-magnitude).

``shock[name]`` is the **log jump multiplier** at each quadrature node
(or, in simulation, at each sample). Zero when no jump; a draw from
``Normal(jump_mu, jump_sigma)`` when a jump occurs. The user's
``transition`` callable adds it to the log-price update:

    next_S = S * exp(diffusion_drift + diffusion_sigma * shock["z"] + shock["jump"])

This pairs naturally with a `Normal` for the continuous diffusion
component — the canonical Merton 1976 jump-diffusion model.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Jump:
    """Bernoulli-approximated Poisson jump with Normal log-magnitudes.

    Parameters
    ----------
    name : str
        Surfaces in ``shock[name]`` in user-supplied ``transition`` /
        ``reward`` callables.
    intensity : float
        Expected jumps per period (Poisson rate × dt). The actual per-period
        jump probability used is ``1 - exp(-intensity)`` — the exact
        Bernoulli approximation of "at least one jump".
    jump_mu : float
        Mean of the log jump multiplier when a jump occurs (default 0.0).
    jump_sigma : float
        Std of the log jump multiplier when a jump occurs (default 1.0).
    """

    name: str
    intensity: float
    jump_mu: float = 0.0
    jump_sigma: float = 1.0

    def __post_init__(self):
        if not self.name:
            raise ValueError("Jump requires a name")
        if self.intensity < 0:
            raise ValueError(f"intensity must be non-negative, got {self.intensity}")
        if not (self.jump_sigma > 0):
            raise ValueError(
                f"jump_sigma must be positive, got {self.jump_sigma}"
            )

    @property
    def p_jump(self) -> float:
        return 1.0 - math.exp(-self.intensity)

    def nodes_and_weights(
        self,
        n_quad: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> tuple[dict, torch.Tensor]:
        """Joint quadrature for the Bernoulli-jump approximation.

        Returns ``({name: nodes}, weights)`` with ``nodes`` of length
        ``1 + n_quad``: a zero entry for the no-jump branch, plus
        ``n_quad`` Gauss-Hermite nodes for the log-jump magnitude.
        Weights sum to 1.
        """
        if n_quad < 1:
            raise ValueError(f"n_quad must be >= 1, got {n_quad}")
        p = self.p_jump

        raw_z, raw_w = np.polynomial.hermite_e.hermegauss(n_quad)
        w_unit = raw_w / math.sqrt(2.0 * math.pi)
        magnitudes = self.jump_mu + self.jump_sigma * raw_z

        nodes = torch.cat([
            torch.zeros(1, dtype=dtype, device=device),
            torch.as_tensor(magnitudes, dtype=dtype, device=device),
        ])
        weights = torch.cat([
            torch.tensor([1.0 - p], dtype=dtype, device=device),
            torch.as_tensor(p * w_unit, dtype=dtype, device=device),
        ])
        return {self.name: nodes}, weights

    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> dict:
        """Draw ``n`` jump samples. Returns ``{name: (n,) tensor}``.

        With probability ``p_jump`` the entry is ``Normal(jump_mu,
        jump_sigma)``; otherwise zero (no jump).
        """
        p = self.p_jump
        u = torch.rand(n, generator=generator, dtype=dtype, device=device)
        jumped = u < p
        z = torch.randn(n, generator=generator, dtype=dtype, device=device)
        magnitudes = self.jump_mu + self.jump_sigma * z
        return {
            self.name: torch.where(jumped, magnitudes, torch.zeros_like(magnitudes))
        }
