"""Golden-section action search.

``GoldenSearch`` is an *action* grid spec (it appears in ``action_grid``,
not ``state_grid``). It replaces brute-force enumeration of a continuous
action with a small ``n_init`` seed grid + vectorized golden-section
refinement, run lock-step across every state and every joint
configuration of the other actions.

Cost per state per refined continuous action:

    n_init + 2 + n_iter   Bellman integrand evaluations
                          (2 initial bracket evals; 1 fresh eval per iter via
                           the standard "one of c, d carries over"
                           golden-section trick)

vs. ``n`` evaluations for a ``RegularGrid(n=n)``. ``n_init=4, n_iter=20``
gives ~26 evals to a final bracket of ``(1 / golden_ratio)^20 ≈ 6.6e-5``
of one seed cell width — much sharper than a 500-point grid (relative
precision ~2e-3) at ~1/20th the work.

When multiple continuous actions are golden, ``n_coord`` outer rounds of
coordinate descent refine each axis in turn. The default ``n_coord=2``
is sufficient for the weakly coupled case (most consumption / portfolio
problems). For strongly coupled actions, increase it.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GoldenSearch:
    """Action grid spec: coarse seed + vectorized golden-section refinement.

    Attributes
    ----------
    n_init : int
        Seed grid points along this action's normalised [0, 1] interval.
        These get cartesian-producted with every other action's grid to
        find the seed (state, joint-action) optimum, then bracket the
        golden-section refinement. ``n_init=4`` is usually enough; raise
        if the reward × value composition is non-concave on the action.
    n_iter : int
        Golden-section iterations per refinement pass. Each iteration
        contracts the bracket by the golden ratio ``φ ≈ 1.618``, so
        ``n_iter=20`` shrinks the bracket by ~7000x.
    n_coord : int
        Outer rounds of coordinate descent when multiple continuous
        actions use ``GoldenSearch``. Ignored when there is a single
        golden-search continuous action.
    """

    n_init: int = 4
    n_iter: int = 20
    n_coord: int = 2

    def __post_init__(self):
        if self.n_init < 2:
            raise ValueError(
                f"GoldenSearch requires n_init >= 2 (need a bracket); got {self.n_init}"
            )
        if self.n_iter < 1:
            raise ValueError(f"GoldenSearch requires n_iter >= 1; got {self.n_iter}")
        if self.n_coord < 1:
            raise ValueError(f"GoldenSearch requires n_coord >= 1; got {self.n_coord}")

    @property
    def n(self) -> int:
        """Seed grid size — matches the ``RegularGrid`` / ``WarpedGrid`` interface
        so ``setup_solve`` can build the joint enumeration axis uniformly."""
        return self.n_init

    def points(
        self,
        low: float,
        high: float,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        warp: object = None,
    ) -> torch.Tensor:
        """Seed grid points — uniform across [low, high]."""
        del warp
        if not (high > low):
            raise ValueError(
                f"GoldenSearch.points requires high > low, got [{low}, {high}]"
            )
        return torch.linspace(low, high, self.n_init, dtype=dtype, device=device)
