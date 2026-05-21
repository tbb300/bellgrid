"""Regular (uniformly spaced) tensor-product grid."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RegularGrid:
    """Uniformly spaced 1-D grid spec.

    The grid carries only its size `n`; the actual `[low, high]` interval is
    supplied at solve time by the caller (the state's `range` for state grids,
    or `[0, 1]` for action grids that get rescaled per state).
    """

    n: int

    def __post_init__(self):
        if self.n < 2:
            raise ValueError(f"RegularGrid requires n >= 2, got {self.n}")

    def points(
        self,
        low: float,
        high: float,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Return `n` linearly spaced points in `[low, high]` (inclusive)."""
        if not (high > low):
            raise ValueError(f"RegularGrid.points requires high > low, got [{low}, {high}]")
        return torch.linspace(low, high, self.n, dtype=dtype, device=device)
