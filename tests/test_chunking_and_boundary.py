"""Tests for the chunked Bellman update and the boundary-escape warning."""

import math
import warnings

import pytest
import torch

from bellgrid import ContinuousAction, ContinuousState, Problem, solve
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction


# --- chunked Bellman: same answer as unchunked ---------------------------


def _merton_problem(
    *, range_high: float = 200.0, horizon: int = 20, with_terminal: bool = True
):
    """Standard log-utility Merton, finite-horizon.

    ``with_terminal``: include the closed-form terminal V (the steady-state
    fixed point — makes the truncated DP reproduce V* exactly at every t).
    Drop it to actually exercise the boundary: without the closed-form
    terminal, multilinear clamps at the grid edge corrupt V backward
    through the sweep.
    """
    beta, mu, sigma = 0.96, 0.04, 0.15
    B = 1.0 / (1.0 - beta)
    A = (
        math.log(1.0 - beta) / (1.0 - beta)
        + (beta / (1.0 - beta) ** 2) * (math.log(beta) + mu)
    )

    def transition(state, action, shock, _t):
        return {"wealth": (state["wealth"] - action["consume"])
                * torch.exp(mu + sigma * shock["z"])}

    def reward(_s, action, _sh, _t):
        return torch.log(action["consume"])

    kwargs = dict(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, range_high))],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, horizon),
        discount=beta,
    )
    if with_terminal:
        kwargs["terminal_reward"] = lambda s: A + B * torch.log(s["wealth"])
    return Problem(**kwargs)


def test_chunking_matches_no_chunking_merton():
    """Tightly chunking the shock axis must produce the SAME V and policy
    as a single-shot bellman step."""
    problem = _merton_problem()

    # Big chunk_size: no chunking.
    _, value_unchunked = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=64)},
        action_grid={"consume": RegularGrid(n=200)},
        solver=BackwardInduction(n_quad=7),
        chunk_size=2**30,
    )
    # Tiny chunk_size: forces one shock node per chunk.
    _, value_chunked = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=64)},
        action_grid={"consume": RegularGrid(n=200)},
        solver=BackwardInduction(n_quad=7),
        chunk_size=1,  # one element per chunk → max chunking
    )

    w = torch.tensor([2.0, 10.0, 25.0, 50.0], dtype=torch.float64)
    v_un = value_unchunked({"wealth": w}, t=10).numpy()
    v_ch = value_chunked({"wealth": w}, t=10).numpy()
    # Floating-point identical (or very close) — chunked sums in a slightly
    # different order, so we allow rounding error around eps.
    for vu, vc in zip(v_un, v_ch):
        assert vu == pytest.approx(vc, abs=1e-10)


# --- boundary-escape warning --------------------------------------------


def test_boundary_warning_fires_when_optimal_policy_overshoots():
    """A deliberately escape-heavy problem: the transition multiplies the
    state by 5x every period, so any interior state guarantees an
    overshoot of the narrow grid. The diagnostic must fire."""

    def transition(state, _a, _sh, _t):
        return {"x": state["x"] * 5.0}

    def reward(_s, action, _sh, _t):
        return action["c"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[ContinuousAction("c", bounds=(0.0, 1.0))],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 3),
        discount=0.9,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        solve(
            problem,
            state_grid={"x": RegularGrid(n=16)},
            action_grid={"c": RegularGrid(n=8)},
            solver=BackwardInduction(n_quad=1),
        )
    msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    boundary_warnings = [m for m in msgs if "outside its grid range" in m]
    assert len(boundary_warnings) >= 1, (
        f"expected at least one boundary warning; got: {msgs}"
    )


def test_boundary_check_can_be_opted_out():
    """boundary_check=False suppresses the diagnostic entirely, even on a
    problem that would otherwise trigger it."""

    def transition(state, _a, _sh, _t):
        return {"x": state["x"] * 5.0}

    def reward(_s, action, _sh, _t):
        return action["c"]

    problem = Problem(
        states=[ContinuousState("x", range=(0.0, 1.0))],
        actions=[ContinuousAction("c", bounds=(0.0, 1.0))],
        transition=transition,
        reward=reward,
        shocks=[],
        horizon=range(0, 3),
        discount=0.9,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        solve(
            problem,
            state_grid={"x": RegularGrid(n=16)},
            action_grid={"c": RegularGrid(n=8)},
            solver=BackwardInduction(n_quad=1, boundary_check=False),
        )
    boundary_warnings = [
        str(w.message) for w in captured
        if issubclass(w.category, UserWarning) and "outside its grid range" in str(w.message)
    ]
    assert boundary_warnings == [], (
        f"expected no warnings with boundary_check=False; got: {boundary_warnings}"
    )


def test_boundary_warning_does_not_fire_on_well_configured_merton():
    """The widely-configured Merton (range up to 200) is fine in practice —
    the diagnostic should NOT fire on it."""
    problem = _merton_problem(range_high=200.0)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        solve(
            problem,
            state_grid={"wealth": WarpedGrid(n=128)},
            action_grid={"consume": RegularGrid(n=500)},
            solver=BackwardInduction(n_quad=7),
        )
    boundary_warnings = [
        str(w.message) for w in captured
        if issubclass(w.category, UserWarning) and "outside its grid range" in str(w.message)
    ]
    assert boundary_warnings == [], (
        f"expected no boundary warning on well-configured Merton; got: {boundary_warnings}"
    )
