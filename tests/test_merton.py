"""End-to-end validation: Merton consumption-savings with log utility.

For log utility u(c) = log(c), a single risky asset with gross return
``R = exp(mu + sigma Z)`` (``Z ~ N(0, 1)``), and discount factor ``beta``,
the infinite-horizon Bellman equation admits the closed form

    V(w) = A + B * log(w)        with  B = 1 / (1 - beta)
    c*   = w * (1 - beta)        — consumption rate is constant in wealth
    A    = log(1-beta)/(1-beta) + (beta/(1-beta)^2) * (log(beta) + mu)

We truncate to a finite horizon by setting the terminal reward to the
closed-form ``V``. With that terminal condition the truncated DP is in
steady state — every period of the backward sweep reproduces the
stationary policy and value function.
"""

import math

import torch

from bellgrid import ContinuousAction, ContinuousState, Problem, solve
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Lognormal, Normal
from bellgrid.solvers import BackwardInduction


def _closed_form_coefficients(beta: float, mu: float):
    B = 1.0 / (1.0 - beta)
    A = (
        math.log(1.0 - beta) / (1.0 - beta)
        + (beta / (1.0 - beta) ** 2) * (math.log(beta) + mu)
    )
    return A, B


def _build_problem(
    beta: float,
    mu: float,
    sigma: float,
    T: int,
    *,
    warp: str | None = None,
    wealth_range: tuple[float, float] = (0.5, 50.0),
    consume_low: float = 1e-3,
):
    A, B = _closed_form_coefficients(beta, mu)

    def transition(state, action, shock, _t):
        return {
            "wealth": (state["wealth"] - action["consume"])
            * torch.exp(mu + sigma * shock["z"])
        }

    def reward(_state, action, _shock, _t):
        return torch.log(action["consume"])

    def terminal_reward(state):
        return A + B * torch.log(state["wealth"])

    return Problem(
        states=[ContinuousState("wealth", warp=warp, range=wealth_range)],
        actions=[ContinuousAction("consume", bounds=(consume_low, "wealth"))],
        transition=transition,
        reward=reward,
        shocks=[Normal("z", sigma=1.0)],
        horizon=range(0, T),
        discount=beta,
        terminal_reward=terminal_reward,
    )


def test_merton_consumption_rate_matches_closed_form():
    beta, mu, sigma = 0.96, 0.04, 0.15
    problem = _build_problem(beta, mu, sigma, T=10)

    policy, _ = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)
    actions = policy({"wealth": test_w}, t=5)  # mid-horizon, steady state
    rates = actions["consume"] / test_w

    expected = torch.full_like(rates, 1.0 - beta)
    assert torch.allclose(rates, expected, rtol=0.03, atol=0.005), (
        f"consumption rates={rates.tolist()}, expected={expected.tolist()}"
    )


def test_merton_value_function_matches_closed_form():
    beta, mu, sigma = 0.96, 0.04, 0.15
    A, B = _closed_form_coefficients(beta, mu)
    problem = _build_problem(beta, mu, sigma, T=10)

    _, value = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)
    v = value({"wealth": test_w}, t=5)
    expected = A + B * torch.log(test_w)

    assert torch.allclose(v, expected, rtol=0.01, atol=0.05), (
        f"V={v.tolist()}, expected={expected.tolist()}"
    )


def test_merton_consumption_rate_constant_across_time():
    """In steady state the consumption rate doesn't drift period-to-period."""
    beta, mu, sigma = 0.96, 0.04, 0.15
    problem = _build_problem(beta, mu, sigma, T=10)

    policy, _ = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    test_w = torch.tensor([5.0], dtype=torch.float64)
    rates = [
        (policy({"wealth": test_w}, t=t)["consume"] / test_w).item()
        for t in (1, 3, 5, 7, 9)
    ]
    # All rates should be ~equal and ~(1-beta)
    assert max(rates) - min(rates) < 0.005, f"rates drift: {rates}"
    assert all(abs(r - (1.0 - beta)) < 0.01 for r in rates), f"rates={rates}"


# ---------------------------------------------------------------------------
# Diagnostic tests — print closed-form vs solver tables.
# Run with `pytest tests/test_merton.py -v -s` to see output.
# ---------------------------------------------------------------------------


def test_merton_print_comparison_table():
    """Print a side-by-side comparison of closed-form and solver policy/value."""
    beta, mu, sigma = 0.96, 0.04, 0.15
    A, B = _closed_form_coefficients(beta, mu)
    expected_rate = 1.0 - beta

    problem = _build_problem(beta, mu, sigma, T=10)
    policy, value = solve(
        problem,
        state_grid={"wealth": RegularGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    # Interior wealth values — the lower-edge of the regular wealth grid
    # produces a known boundary artifact (V at the leftmost grid point is
    # clamped, so consuming-all wins there). We test on the interior where
    # the closed form holds; a warped wealth grid would close this gap.
    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0, 30.0], dtype=torch.float64)
    v_cf = (A + B * torch.log(test_w)).tolist()

    print()
    print(f"Merton log utility  |  beta={beta}, mu={mu}, sigma={sigma}")
    print(f"Closed form:  c/w = {expected_rate:.4f}    "
          f"V(w) = {A:.4f} + {B:.4f}*log(w)")
    print()
    print(f"{'t':>3} {'w':>8} {'c/w solver':>12} {'c/w CF':>10}"
          f" {'Δ c/w':>11} {'V solver':>12} {'V CF':>12} {'Δ V':>11}")
    print("-" * 86)

    max_rate_err = 0.0
    max_v_err = 0.0
    for t in (1, 5, 9):
        actions = policy({"wealth": test_w}, t=t)
        v = value({"wealth": test_w}, t=t)
        rates = (actions["consume"] / test_w).tolist()
        v_list = v.tolist()

        for i, w in enumerate(test_w.tolist()):
            re = rates[i] - expected_rate
            ve = v_list[i] - v_cf[i]
            print(f"{t:>3d} {w:>8.2f} {rates[i]:>12.6f} {expected_rate:>10.6f}"
                  f" {re:>+11.2e} {v_list[i]:>12.4f} {v_cf[i]:>12.4f} {ve:>+11.2e}")
            max_rate_err = max(max_rate_err, abs(re))
            max_v_err = max(max_v_err, abs(ve))
        print()

    print(f"Max |Δ c/w| = {max_rate_err:.2e}    Max |Δ V| = {max_v_err:.2e}")

    assert max_rate_err < 0.01
    # V ranges over ~70 units across our test wealths; ~1% absolute is fine
    # and bounded by the same resolution dependence shown in the sweep test.
    assert max_v_err < 1.0


def test_merton_print_resolution_sweep():
    """Sweep grid resolution under RegularGrid and WarpedGrid; print error decay."""
    beta, mu, sigma = 0.96, 0.04, 0.15
    A, B = _closed_form_coefficients(beta, mu)
    expected_rate = 1.0 - beta

    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)
    v_cf = A + B * torch.log(test_w)

    print()
    print(f"Resolution sweep  |  beta={beta}, mu={mu}, sigma={sigma}, t=5")
    print()
    print(f"{'grid':>9} {'n_wealth':>10} {'n_consume':>11} {'n_quad':>8}"
          f" {'max |Δ c/w|':>14} {'max |Δ V|':>14} {'wall (s)':>10}")
    print("-" * 85)

    import time

    for grid_label, grid_factory in [
        ("Regular", lambda n: (RegularGrid(n=n), None, (0.5, 50.0))),
        ("Warped",  lambda n: (WarpedGrid(n=n),  "asinh", (1e-3, 50.0))),
    ]:
        prev_v_err = None
        for n_w, n_c, n_q in [(32, 100, 5), (64, 200, 5), (128, 500, 7), (256, 1000, 9)]:
            wealth_grid, warp, wealth_range = grid_factory(n_w)
            problem = _build_problem(
                beta, mu, sigma, T=10, warp=warp, wealth_range=wealth_range
            )
            t0 = time.perf_counter()
            policy, value = solve(
                problem,
                state_grid={"wealth": wealth_grid},
                action_grid={"consume": RegularGrid(n=n_c)},
                solver=BackwardInduction(n_quad=n_q),
            )
            wall = time.perf_counter() - t0

            rates = policy({"wealth": test_w}, t=5)["consume"] / test_w
            v = value({"wealth": test_w}, t=5)
            rate_err = (rates - expected_rate).abs().max().item()
            v_err = (v - v_cf).abs().max().item()

            print(f"{grid_label:>9} {n_w:>10d} {n_c:>11d} {n_q:>8d}"
                  f" {rate_err:>14.4e} {v_err:>14.4e} {wall:>10.3f}")

            if prev_v_err is not None:
                assert v_err <= prev_v_err * 1.5, (
                    f"{grid_label} V error grew with refinement: "
                    f"{prev_v_err:.4e} → {v_err:.4e}"
                )
            prev_v_err = v_err
        print()


def test_merton_lognormal_matches_normal_with_exp_in_transition():
    """The Lognormal shock and the equivalent Normal("z") + exp(mu+sigma*z)
    written into transition give the same solution. Lognormal is just
    sugar over the standard pattern; switching shouldn't change the math."""
    beta, mu, sigma = 0.96, 0.04, 0.15
    A, B = _closed_form_coefficients(beta, mu)

    def transition_lognormal(state, action, shock, _t):
        return {
            "wealth": (state["wealth"] - action["consume"]) * shock["ret"]
        }

    def reward(_state, action, _shock, _t):
        return torch.log(action["consume"])

    def terminal_reward(state):
        return A + B * torch.log(state["wealth"])

    problem_lognormal = Problem(
        states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 50.0))],
        actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
        transition=transition_lognormal,
        reward=reward,
        shocks=[Lognormal("ret", mu=mu, sigma=sigma)],
        horizon=range(0, 10),
        discount=beta,
        terminal_reward=terminal_reward,
    )

    policy_ln, value_ln = solve(
        problem_lognormal,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    # The Normal-with-exp formulation (using _build_problem) under the
    # same wealth range and consume bounds.
    problem_normal = _build_problem(
        beta, mu, sigma, T=10,
        warp="asinh", wealth_range=(1e-3, 50.0), consume_low=1e-6,
    )
    policy_no, value_no = solve(
        problem_normal,
        state_grid={"wealth": WarpedGrid(n=128)},
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)
    v_ln = value_ln({"wealth": test_w}, t=5)
    v_no = value_no({"wealth": test_w}, t=5)
    assert torch.allclose(v_ln, v_no, atol=1e-12), (
        f"Lognormal V: {v_ln.tolist()}\nNormal+exp V: {v_no.tolist()}"
    )

    c_ln = policy_ln({"wealth": test_w}, t=5)["consume"]
    c_no = policy_no({"wealth": test_w}, t=5)["consume"]
    assert torch.allclose(c_ln, c_no, atol=1e-12), (
        f"Lognormal c: {c_ln.tolist()}\nNormal+exp c: {c_no.tolist()}"
    )


def test_merton_warped_grid_reclaims_low_wealth():
    """With an asinh-warped wealth grid starting near 0, the closed-form
    policy is recovered at low wealth (where the regular grid showed a
    boundary artifact)."""
    beta, mu, sigma = 0.96, 0.04, 0.15
    A, B = _closed_form_coefficients(beta, mu)
    expected_rate = 1.0 - beta

    problem = _build_problem(
        beta, mu, sigma, T=10,
        warp="asinh", wealth_range=(1e-3, 50.0), consume_low=1e-6,
    )
    policy, value = solve(
        problem,
        state_grid={"wealth": WarpedGrid(n=128)},     # asinh inherited
        action_grid={"consume": RegularGrid(n=500)},
        solver=BackwardInduction(n_quad=7),
    )

    # w=1.0 is the value that failed under the regular grid setup
    test_w = torch.tensor([1.0, 2.0, 5.0, 10.0, 20.0], dtype=torch.float64)
    actions = policy({"wealth": test_w}, t=5)
    rates = actions["consume"] / test_w
    expected = torch.full_like(rates, expected_rate)
    assert torch.allclose(rates, expected, rtol=0.05, atol=0.005), (
        f"rates={rates.tolist()}"
    )

    # V should match closed form across the same range
    v = value({"wealth": test_w}, t=5)
    v_cf = A + B * torch.log(test_w)
    assert torch.allclose(v, v_cf, rtol=0.02, atol=1.0), (
        f"V={v.tolist()}, expected={v_cf.tolist()}"
    )


def test_merton_print_parameter_sweep():
    """Show solver vs closed-form across several (beta, mu) parameterizations."""
    test_w = torch.tensor([2.0, 5.0, 10.0, 20.0], dtype=torch.float64)

    print()
    print("Parameter sweep at t=5, w=10  |  sigma=0.15 fixed")
    print()
    print(f"{'beta':>6} {'mu':>6} {'1-beta':>10} {'c/w solver':>12}"
          f" {'Δ c/w':>11} {'A':>10} {'B':>10}")
    print("-" * 70)

    for beta in (0.92, 0.95, 0.97):
        for mu in (0.02, 0.04, 0.06):
            sigma = 0.15
            A, B = _closed_form_coefficients(beta, mu)
            expected_rate = 1.0 - beta
            problem = _build_problem(beta, mu, sigma, T=10)
            policy, _ = solve(
                problem,
                state_grid={"wealth": RegularGrid(n=128)},
                action_grid={"consume": RegularGrid(n=500)},
                solver=BackwardInduction(n_quad=7),
            )
            actions = policy({"wealth": test_w}, t=5)
            rate = (actions["consume"][2] / test_w[2]).item()  # w=10
            err = rate - expected_rate
            print(f"{beta:>6.2f} {mu:>6.2f} {expected_rate:>10.4f}"
                  f" {rate:>12.6f} {err:>+11.2e} {A:>10.3f} {B:>10.3f}")

            assert abs(err) < 0.02, f"beta={beta}, mu={mu}: rate err = {err}"

    print()
