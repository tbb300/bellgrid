"""6-state economic regime model.

Each regime has its own (nominal) equity-return mean, bond yield, and
inflation rate. Regimes evolve under a tridiagonal mean-reverting
transition matrix — most periods stay in the same regime; occasional
moves to an adjacent regime; rare crossings of multiple regimes only
via multi-period chains.

Bond return is treated as the regime's yield (skipping the within-period
yield-drift effect that requires next-regime values — see
``docs/api.md`` on MarkovChain's next-value limitation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegimeParameters:
    """Parameters for the 6-state regime model. Indices 0..5 correspond
    to: deflationary bust, recession, normal, expansion, inflation,
    stagflation."""

    labels: tuple = (
        "deflationary_bust", "recession", "normal",
        "expansion", "inflation", "stagflation",
    )
    yields: tuple = (0.005, 0.015, 0.040, 0.055, 0.075, 0.090)
    inflation: tuple = (-0.010, 0.010, 0.025, 0.030, 0.050, 0.075)
    equity_means: tuple = (0.082, 0.092, 0.100, 0.102, 0.112, 0.125)
    equity_vol: float = 0.18

    @property
    def n_regimes(self) -> int:
        return len(self.labels)


def tridiagonal_matrix(n: int = 6, p_stay: float = 0.6) -> np.ndarray:
    """Row-stochastic tridiagonal mean-reverting matrix for ``n`` regimes.

    Each state stays with probability ``p_stay``; the remainder is split
    between adjacent states (interior) or piled onto the single adjacent
    state (endpoints). The matrix is symmetric in transition probabilities
    so the stationary distribution is approximately uniform over the
    middle states.
    """
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if i == 0:
            P[i, i] = p_stay
            P[i, i + 1] = 1.0 - p_stay
        elif i == n - 1:
            P[i, i - 1] = 1.0 - p_stay
            P[i, i] = p_stay
        else:
            P[i, i] = p_stay
            P[i, i - 1] = (1.0 - p_stay) / 2.0
            P[i, i + 1] = (1.0 - p_stay) / 2.0
    return P
