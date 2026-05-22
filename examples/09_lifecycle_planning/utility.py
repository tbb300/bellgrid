"""CRRA utility for consumption and bequest.

Consumption utility is the standard CRRA form with a subsistence floor
(below which utility goes to ``-∞`` via a log barrier so the solver
never picks below-floor consumption when it can avoid it). Bequest
utility annuitises the bequest wealth over a fixed window and computes
the CRRA utility of that consumption stream relative to a baseline
income — the standard "warm-glow" bequest specification.

Both use a normalisation that puts ``U(floor) = -∞`` and ``U(preferred)
= 1``, which makes the magnitudes of consumption vs. bequest utility
directly comparable.
"""

from __future__ import annotations

import math

import torch


_EPS = 1e-12
_BELOW_FLOOR_SCALE = 50.0  # how steeply utility drops below the floor


def crra_normalised(
    c: torch.Tensor,
    *,
    floor: float,
    preferred: float,
    risk_aversion: float,
) -> torch.Tensor:
    """CRRA utility normalised so that ``U(floor) = 0`` and ``U(preferred) = 1``.

    Below the floor, switches to a log barrier ``scale·log(c/floor)`` that
    diverges to ``-∞`` as ``c → 0``.
    """
    c_safe = torch.clamp(c, min=_EPS * floor)
    u_below = _BELOW_FLOOR_SCALE * torch.log(c_safe / floor)

    if abs(risk_aversion - 1.0) < 1e-9:
        c_above = torch.clamp(c, min=floor)
        num = torch.log(c_above) - math.log(floor)
        den = math.log(preferred) - math.log(floor)
        u_above = num / den
    else:
        omk = 1.0 - risk_aversion
        c_above = torch.clamp(c, min=floor)
        num = c_above ** omk - floor ** omk
        den = preferred ** omk - floor ** omk
        u_above = num / den

    return torch.where(c < floor, u_below, u_above)


def consumption_utility(
    c: torch.Tensor,
    phase: torch.Tensor,
    *,
    floor: float = 10_000.0,
    preferred: float = 60_000.0,
    risk_aversion: float = 2.5,
    retirement_bonus: float = 0.1,
) -> torch.Tensor:
    """Per-period CRRA utility of consumption ``c`` with a small leisure
    bonus added in the retired phase.

    ``c`` and ``phase`` are tensors with matching broadcast shapes;
    ``phase == 1`` means retired.
    """
    u = crra_normalised(c, floor=floor, preferred=preferred, risk_aversion=risk_aversion)
    retired = (phase == 1).to(u.dtype)
    return u + retirement_bonus * retired


def bequest_utility(
    wealth: torch.Tensor,
    *,
    annuity_years: int = 20,
    annuity_discount: float = 0.0,
    baseline_income: float = 30_000.0,
    floor: float = 10_000.0,
    preferred: float = 50_000.0,
    risk_aversion: float = 2.5,
    bequest_strength: float = 1.0,
) -> torch.Tensor:
    """Warm-glow bequest utility.

    The bequest is annuitised over ``annuity_years`` years at a real
    discount ``annuity_discount`` (defaulting to 0 for a level annuity).
    The annuity payment is added to a ``baseline_income`` representing
    the recipient's own resources, and CRRA utility is taken on that
    sum. The result is multiplied by the (undiscounted) number of
    annuity years to give the total welfare gain from the bequest.

    Setting ``bequest_strength=0`` turns off the bequest motive entirely
    (useful as a sanity case).
    """
    wealth_pos = torch.clamp(wealth, min=0.0)
    if abs(annuity_discount) < 1e-12:
        annuity_payment = wealth_pos / annuity_years
    else:
        # Present-value annuity factor
        af = annuity_discount / (1.0 - (1.0 + annuity_discount) ** (-annuity_years))
        annuity_payment = wealth_pos * af

    u_with = crra_normalised(
        annuity_payment + baseline_income,
        floor=floor, preferred=preferred, risk_aversion=risk_aversion,
    )
    u_base = crra_normalised(
        torch.full_like(wealth_pos, baseline_income),
        floor=floor, preferred=preferred, risk_aversion=risk_aversion,
    )
    return bequest_strength * annuity_years * (u_with - u_base)
