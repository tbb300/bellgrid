"""Boundary-escape diagnostic with stochastic, regime-dependent dynamics.

Existing boundary tests use deterministic transitions (state × 5 each
step → guaranteed escape). This file tests two cases the diagnostic
hadn't been exercised on:

1. Stochastic transition where escape is regime-dependent — should warn
   on the bad regime, stay quiet on the good regime.
2. Well-configured MarkovChain problem — should NOT warn spuriously.
"""

import warnings

import numpy as np
import pytest
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    MarkovChain,
    Problem,
    solve,
)
from bellgrid.grids import RegularGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction


def test_boundary_warning_with_regime_dependent_escape():
    """Two-regime model: regime 0 has a "tame" transition (x stays in
    range); regime 1 has a "wild" transition (x ×= 5 each step,
    escapes range). The boundary diagnostic should fire because the
    overall escape mass is high under the optimal policy."""
    P = np.array([[0.5, 0.5], [0.5, 0.5]])

    def transition(state, _a, _sh, _t):
        x = state["x"]; regime = state["regime"]
        # Regime 0: small drift. Regime 1: amplification (escape!)
        return {
            "x": torch.where(regime == 0, x + 0.01, x * 5.0)
        }

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 5), discount=0.9,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve(problem, state_grid={"x": RegularGrid(n=8)},
              action_grid={}, solver=BackwardInduction(n_quad=1))
    boundary_warnings = [w for w in caught if "outside its grid range" in str(w.message)]
    assert len(boundary_warnings) > 0, (
        "Expected a boundary-escape warning under regime-dependent escape, "
        f"got none. All warnings: {[str(w.message) for w in caught]}"
    )


def test_boundary_no_spurious_warning_on_clean_markov_chain_problem():
    """A clean Merton-style problem with a MarkovChain regime affecting
    only the drift (not the range) should NOT trigger a boundary warning."""
    # Two regimes with slightly different drifts, both inside the range.
    P = np.array([[0.7, 0.3], [0.3, 0.7]])
    drifts = (0.02, 0.04)

    def transition(state, action, shock, _t):
        x = state["x"]
        regime = state["regime"]
        mu_t = torch.tensor(drifts, dtype=torch.float64, device=x.device)[regime]
        # Pure log-utility Merton-like: smooth, well-inside-range drift
        return {"x": torch.clamp(x * (1.0 + mu_t + 0.05 * shock["z"]),
                                  min=1e-3, max=100.0)}

    def reward(_s, action, _sh, _t):
        # Make consumption a no-op so we just exercise the dynamics path
        return torch.tensor(0.0, dtype=torch.float64)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.1, 100.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, 5), discount=0.95,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve(problem, state_grid={"x": RegularGrid(n=16)},
              action_grid={}, solver=BackwardInduction(n_quad=5))
    boundary_warnings = [w for w in caught if "outside its grid range" in str(w.message)]
    assert len(boundary_warnings) == 0, (
        "Expected no boundary warning on a clean problem, got: "
        f"{[str(w.message) for w in boundary_warnings]}"
    )


def test_boundary_opt_out_with_markov_chain():
    """When ``boundary_check=False``, no warning should fire even on an
    escape-heavy regime-dependent problem."""
    P = np.array([[0.5, 0.5], [0.5, 0.5]])

    def transition(state, _a, _sh, _t):
        x = state["x"]; regime = state["regime"]
        return {"x": torch.where(regime == 0, x + 0.01, x * 5.0)}

    def reward(_s, _a, _sh, _t):
        return torch.tensor(0.0, dtype=torch.float64)

    problem = Problem(
        states=[
            ContinuousState("x", range=(0.0, 1.0)),
            MarkovChain("regime", matrix=P),
        ],
        actions=[DiscreteAction("noop", n=1)],
        transition=transition, reward=reward, shocks=[],
        horizon=range(0, 5), discount=0.9,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve(problem, state_grid={"x": RegularGrid(n=8)},
              action_grid={},
              solver=BackwardInduction(n_quad=1, boundary_check=False))
    boundary_warnings = [w for w in caught if "outside its grid range" in str(w.message)]
    assert len(boundary_warnings) == 0, (
        f"opt-out failed: got {len(boundary_warnings)} boundary warnings"
    )
