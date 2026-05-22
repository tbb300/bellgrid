"""Simplified mortality table for the lifecycle example.

Approximates the 2017 CSO non-smoker-male ultimate table by linearly
interpolating between hardcoded q_x ("probability of dying in the next
year") values at five-year intervals. Real applications would use a
proper actuarial table (SOA's ``pymort`` is the obvious dependency)
keyed to the planholder's sex, smoking status, and health; for an
example we just want the *shape* of human mortality — rising slowly
through middle age and steeply past 70.
"""

from __future__ import annotations

import torch


# Approximate q_x values for non-smoker males at five-year intervals.
# Interpolated linearly between knots for ages in between.
_Q_X_KNOTS = {
    25: 0.00050,
    30: 0.00060,
    35: 0.00080,
    40: 0.00120,
    45: 0.00200,
    50: 0.00350,
    55: 0.00580,
    60: 0.00950,
    65: 0.01480,
    70: 0.02250,
    75: 0.03450,
    80: 0.05550,
    85: 0.09000,
    90: 0.14500,
    95: 0.22000,
    100: 0.30000,
    105: 0.45000,
    110: 0.65000,
    115: 1.00000,
}


def _q_x(age: int) -> float:
    """Probability of dying between age and age + 1."""
    knots = sorted(_Q_X_KNOTS.keys())
    if age <= knots[0]:
        return _Q_X_KNOTS[knots[0]]
    if age >= knots[-1]:
        return 1.0
    # Bracket and linear-interpolate
    for i in range(len(knots) - 1):
        a_lo, a_hi = knots[i], knots[i + 1]
        if a_lo <= age <= a_hi:
            t = (age - a_lo) / (a_hi - a_lo)
            return (1.0 - t) * _Q_X_KNOTS[a_lo] + t * _Q_X_KNOTS[a_hi]
    return 1.0


def survival_table(
    min_age: int = 0,
    max_age: int = 120,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return ``p_survive[age]`` for ``age`` in ``[min_age, max_age]``.

    Index by integer age: ``table[age]`` is the probability of surviving
    from ``age`` to ``age + 1``. ``table[max_age]`` is 0 by definition.
    """
    out = torch.zeros(max_age + 1, dtype=dtype, device=device)
    for age in range(min_age, max_age + 1):
        out[age] = 1.0 - _q_x(age)
    return out


def life_expectancy(start_age: int, max_age: int = 120) -> float:
    """Remaining life expectancy from ``start_age`` under this table."""
    if start_age >= max_age:
        return 0.0
    p_survive_to = 1.0
    expectancy = 0.0
    for age in range(start_age, max_age):
        p_survive_to *= 1.0 - _q_x(age)
        expectancy += p_survive_to
    return expectancy
