"""Warped 1-D grid: points placed evenly in a transformed coordinate.

Built-in warps:

- ``"asinh"`` — ``x → asinh(x)``. Concentrates points near zero and handles
  ranges that include zero. Good for wealth/inventory grids.
- ``"log"`` — ``x → log(x)``. Concentrates points exponentially toward the
  lower end. Requires ``low > 0``.

The warp can be set on the WarpedGrid itself or inherited from the
ContinuousState declaration (the latter is the documented default — see
``docs/api.md``). The same warp also drives interpolation: queries are
transformed into warped space before linear interpolation, so a warp
matched to the value function's curvature gives a tighter approximation
(e.g. log-utility under a log warp is interp-exact up to floating point).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch


# Each entry is (scalar_fwd, tensor_fwd, tensor_inv).
_BUILTIN_WARPS = {
    "asinh": (math.asinh, torch.asinh, torch.sinh),
    "log":   (math.log,   torch.log,   torch.exp),
}


def _resolve_warp(warp):
    if isinstance(warp, str):
        if warp not in _BUILTIN_WARPS:
            raise ValueError(
                f"unknown warp {warp!r}; choose from {sorted(_BUILTIN_WARPS)}"
            )
        return _BUILTIN_WARPS[warp]
    raise NotImplementedError(
        f"callable warps not yet supported; got {type(warp).__name__}"
    )


@dataclass(frozen=True)
class WarpedGrid:
    """1-D grid spec with points placed evenly in a transformed coordinate.

    ``warp=None`` inherits the warp from the state declaration at solve time
    (passed through ``points(..., warp=...)``); a non-None value overrides
    the state's warp.
    """

    n: int
    warp: Optional[str] = None

    def __post_init__(self):
        if self.n < 2:
            raise ValueError(f"WarpedGrid requires n >= 2, got {self.n}")

    def points(
        self,
        low: float,
        high: float,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        warp: Optional[str] = None,
    ) -> torch.Tensor:
        if not (high > low):
            raise ValueError(
                f"WarpedGrid.points requires high > low, got [{low}, {high}]"
            )

        effective_warp = self._effective_warp(warp)
        if effective_warp == "log" and low <= 0:
            raise ValueError(f"log warp requires low > 0, got low={low}")

        scalar_fwd, _, tensor_inv = _resolve_warp(effective_warp)
        u_low = scalar_fwd(float(low))
        u_high = scalar_fwd(float(high))
        u = torch.linspace(u_low, u_high, self.n, dtype=dtype, device=device)
        return tensor_inv(u)

    def transform_for_interp(
        self,
        x: torch.Tensor,
        *,
        warp: Optional[str] = None,
    ) -> torch.Tensor:
        """Map physical-space values to the warped coordinate the solver
        interpolates in. Identity-equivalent under no warp; under ``log`` we
        clamp non-positive inputs to the smallest positive float so the
        forward transform is finite (the multilinear edge-clamp then handles
        the resulting out-of-range query)."""
        effective_warp = self._effective_warp(warp)
        _, tensor_fwd, _ = _resolve_warp(effective_warp)
        if effective_warp == "log":
            x = torch.clamp(x, min=torch.finfo(x.dtype).tiny)
        return tensor_fwd(x)

    def _effective_warp(self, warp: Optional[str]) -> str:
        effective = self.warp if self.warp is not None else warp
        if effective is None:
            raise ValueError(
                "WarpedGrid requires a warp — pass it at construction "
                "(`WarpedGrid(n=..., warp=...)`) or have it inherited from "
                "the ContinuousState declaration"
            )
        return effective
