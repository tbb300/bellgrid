"""Deterministic age-earnings profile.

Asymmetric Gaussian curve peaking at ``peak_age`` (typically 50) and
calibrated to hit specific income multipliers at age 20 (45% of peak)
and age 65 (80% of peak). Based on the Cocco-Gomes-Maenhout (2005)
lifecycle wage profile. Wages are zero above ``retirement_age`` (the
hard mandatory-retirement bound; the user's policy chooses when to
retire below that).
"""

from __future__ import annotations

import math

import torch


_VAL_AT_20 = 0.45
_VAL_AT_65 = 0.80


def _profile_sigma_below(peak_age: float) -> float:
    return math.sqrt(-(20 - peak_age) ** 2 / (2.0 * math.log(_VAL_AT_20)))


def _profile_sigma_above(peak_age: float) -> float:
    return math.sqrt(-(65 - peak_age) ** 2 / (2.0 * math.log(_VAL_AT_65)))


def wage_at_age(
    age: torch.Tensor | float,
    *,
    peak_wage: float = 150_000.0,
    peak_age: float = 50.0,
    retirement_age: float = 70.0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Wage at integer ``age`` from the deterministic age-earnings profile.

    ``age`` may be a scalar (returns a 0-D tensor) or a tensor of any shape
    (returns the same shape with per-age wages). Returns zero for ages
    at or above ``retirement_age``.
    """
    if not isinstance(age, torch.Tensor):
        age = torch.as_tensor(age, dtype=dtype, device=device)
    sigma_below = _profile_sigma_below(peak_age)
    sigma_above = _profile_sigma_above(peak_age)

    pa = torch.as_tensor(peak_age, dtype=age.dtype, device=age.device)
    sb = torch.as_tensor(sigma_below, dtype=age.dtype, device=age.device)
    sa = torch.as_tensor(sigma_above, dtype=age.dtype, device=age.device)

    val_below = torch.exp(-((age - pa) ** 2) / (2.0 * sb ** 2))
    val_above = torch.exp(-((age - pa) ** 2) / (2.0 * sa ** 2))
    mult = torch.where(age < pa, val_below, val_above)
    # Floor at the age-20 multiplier for very young ages, zero after retirement
    mult = torch.where(age < 20, torch.full_like(age, _VAL_AT_20), mult)
    mult = torch.where(age >= retirement_age, torch.zeros_like(age), mult)
    return peak_wage * mult
