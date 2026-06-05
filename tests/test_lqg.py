"""2-D Linear-Quadratic-Gaussian control — closed-form benchmark.

Discrete-time LQR with two-dim state, scalar action, scalar Normal noise:

    x_{t+1} = A x_t + B u_t + C w_t            w_t ~ N(0, 1)
    r_t(x, u) = -(x^T Q x + u^T R u)

Bellgrid maximises, so the closed-form value is
``V_t(x) = -(x^T P_t x + c_t)`` with ``P_t`` from the backward Riccati
recursion and ``c_t`` the noise-term accumulator. Optimal policy:
``u_t*(x) = -K_t x``.

This is the canonical analytical benchmark for multi-dim continuous-state
DP. The solver is invoked once via a module-scoped fixture so the
parametrised tests stay fast.
"""

import numpy as np
import pytest
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    Problem,
    solve,
)
from bellgrid.grids import RegularGrid
from bellgrid.rl import PolicyGradient
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction, iLQG


# --- Riccati reference --------------------------------------------------


def _riccati_lqg(A, B, C, Q, R, gamma: float, T: int):
    """Backward Riccati for the discrete-time finite-horizon LQR + noise."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    P_list = [None] * (T + 1)
    K_list = [None] * T
    c_list = [0.0] * (T + 1)
    P_list[T] = Q

    for t in range(T - 1, -1, -1):
        Pn = P_list[t + 1]
        Rg = R + gamma * B.T @ Pn @ B
        K_list[t] = np.linalg.solve(Rg, gamma * B.T @ Pn @ A)
        P_list[t] = Q + gamma * A.T @ Pn @ A - gamma * A.T @ Pn @ B @ K_list[t]
        # c_t = gamma * (E[(Cw)^T P_{t+1} (Cw)] + c_{t+1})
        c_list[t] = gamma * (float((C.T @ Pn @ C).item()) + c_list[t + 1])

    return P_list, K_list, c_list


def _lqg_value(P, c, x):
    return -(x.T @ P @ x + c)


def _lqg_action(K, x):
    return -(K @ x).item()


# --- LQG problem (constants) --------------------------------------------


A_MAT = np.array([[0.9, 0.1], [0.05, 0.85]])
B_MAT = np.array([[1.0], [0.5]])
C_MAT = np.array([[0.1], [0.05]])
Q_MAT = np.eye(2)
R_MAT = np.array([[0.1]])
GAMMA = 0.95
T_HORIZON = 15


def _build_lqg_problem():
    A = torch.as_tensor(A_MAT, dtype=torch.float64)
    B = torch.as_tensor(B_MAT.flatten(), dtype=torch.float64)
    C = torch.as_tensor(C_MAT.flatten(), dtype=torch.float64)

    def transition(state, action, shock, t):
        x1 = state["x1"]
        x2 = state["x2"]
        u = action["u"]
        w = shock["w"]
        next_x1 = A[0, 0] * x1 + A[0, 1] * x2 + B[0] * u + C[0] * w
        next_x2 = A[1, 0] * x1 + A[1, 1] * x2 + B[1] * u + C[1] * w
        return {"x1": next_x1, "x2": next_x2}

    def reward(state, action, shock, t):
        return -(state["x1"]**2 + state["x2"]**2 + 0.1 * action["u"]**2)

    def terminal_reward(state):
        return -(state["x1"]**2 + state["x2"]**2)

    return Problem(
        states=[
            ContinuousState("x1", range=(-5.0, 5.0)),
            ContinuousState("x2", range=(-5.0, 5.0)),
        ],
        actions=[ContinuousAction("u", bounds=(-5.0, 5.0))],
        transition=transition,
        reward=reward,
        shocks=[Normal("w", sigma=1.0)],
        horizon=range(0, T_HORIZON),
        discount=GAMMA,
        terminal_reward=terminal_reward,
    )


@pytest.fixture(scope="module")
def lqg_solved():
    """Solve once per test module."""
    problem = _build_lqg_problem()
    policy, value = solve(
        problem,
        state_grid={
            "x1": RegularGrid(n=129),
            "x2": RegularGrid(n=129),
        },
        action_grid={"u": RegularGrid(n=101)},
        solver=BackwardInduction(n_quad=7),
    )
    P_list, K_list, c_list = _riccati_lqg(
        A_MAT, B_MAT, C_MAT, Q_MAT, R_MAT, GAMMA, T_HORIZON
    )
    return {"policy": policy, "value": value, "P": P_list, "K": K_list, "c": c_list}


# --- tests --------------------------------------------------------------


@pytest.mark.parametrize(
    "x_pair",
    [
        (0.5, 0.5),
        (-0.5, 1.0),
        (1.5, -1.0),
        (-2.0, -2.0),
        (0.0, 0.0),
        # Opposite-sign corners — exposes the upper-edge boundary artifact
        # when the state range is too narrow (the optimal next-state pushes
        # past the grid edge under positive shocks).
        (2.5, -2.5),
        (-2.5, 2.5),
    ],
)
def test_lqg_value_matches_riccati(lqg_solved, x_pair):
    x = np.array(x_pair, dtype=np.float64)
    v_closed = _lqg_value(lqg_solved["P"][0], lqg_solved["c"][0], x)

    state_q = {
        "x1": torch.tensor([x[0]], dtype=torch.float64),
        "x2": torch.tensor([x[1]], dtype=torch.float64),
    }
    v_bg = lqg_solved["value"](state_q, t=0).item()

    # Quadratic V means multilinear interp has an O(h^2) bias that
    # accumulates across the T backward sweeps. At range=(-5, 5) with
    # n=129 the grid spacing h ≈ 0.078, so per-step bias is roughly
    # 1.5x what it was at range=(-3, 3) with the same n.
    assert v_bg == pytest.approx(v_closed, rel=0.05, abs=0.1), (
        f"x={x_pair}: bellgrid V = {v_bg:.4f}, riccati V = {v_closed:.4f}"
    )


@pytest.mark.parametrize(
    "x_pair",
    [
        (0.5, 0.5),
        (-0.5, 1.0),
        (1.5, -1.0),
        (-1.5, 1.0),
    ],
)
def test_lqg_policy_matches_riccati(lqg_solved, x_pair):
    x = np.array(x_pair, dtype=np.float64)
    u_closed = _lqg_action(lqg_solved["K"][0], x)

    state_q = {
        "x1": torch.tensor([x[0]], dtype=torch.float64),
        "x2": torch.tensor([x[1]], dtype=torch.float64),
    }
    u_bg = lqg_solved["policy"](state_q, t=0)["u"].item()

    assert u_bg == pytest.approx(u_closed, abs=0.1), (
        f"x={x_pair}: bellgrid u = {u_bg:.4f}, riccati u = {u_closed:.4f}"
    )


def test_lqg_print_convergence_sweep():
    """Resolution sweep: confirm bellgrid V converges to Riccati as the state
    grid refines. Multilinear-on-quadratic bias per step is O(h^2); over T
    backward sweeps we expect the accumulated error to track the same rate.
    """
    P_list, K_list, c_list = _riccati_lqg(
        A_MAT, B_MAT, C_MAT, Q_MAT, R_MAT, GAMMA, T_HORIZON
    )
    P0, K0, c0 = P_list[0], K_list[0], c_list[0]

    test_states = [(0.0, 0.0), (-1.0, 1.0), (1.5, -1.5)]

    print()
    print("LQG convergence in state-grid resolution  |  action_grid n=101, n_quad=7")
    print()
    print(f"{'n_state':>10} {'h':>8} {'max |Δ V|':>14} {'max |Δ u|':>14}"
          f" {'V ratio':>10}")
    print("-" * 65)

    prev_v_err = None
    for n in (33, 65, 129):
        problem = _build_lqg_problem()
        policy, value = solve(
            problem,
            state_grid={
                "x1": RegularGrid(n=n),
                "x2": RegularGrid(n=n),
            },
            action_grid={"u": RegularGrid(n=101)},
            solver=BackwardInduction(n_quad=7),
        )

        max_v_err = 0.0
        max_u_err = 0.0
        for (x1, x2) in test_states:
            x = np.array([x1, x2])
            v_closed = _lqg_value(P0, c0, x)
            u_closed = _lqg_action(K0, x)
            state_q = {
                "x1": torch.tensor([x1], dtype=torch.float64),
                "x2": torch.tensor([x2], dtype=torch.float64),
            }
            v_bg = value(state_q, t=0).item()
            u_bg = policy(state_q, t=0)["u"].item()
            max_v_err = max(max_v_err, abs(v_bg - v_closed))
            max_u_err = max(max_u_err, abs(u_bg - u_closed))

        h = 10.0 / (n - 1)
        ratio = f"{prev_v_err / max_v_err:.2f}x" if prev_v_err is not None else "—"
        print(f"{n:>10d} {h:>8.4f} {max_v_err:>14.4e} {max_u_err:>14.4e}"
              f" {ratio:>10}")

        # Each refinement should strictly improve V error.
        if prev_v_err is not None:
            assert max_v_err < prev_v_err, (
                f"V error didn't decrease: {prev_v_err:.4e} -> {max_v_err:.4e}"
            )
        prev_v_err = max_v_err

    print()


def test_lqg_print_comparison_table(lqg_solved):
    P0 = lqg_solved["P"][0]
    K0 = lqg_solved["K"][0]
    c0 = lqg_solved["c"][0]
    policy = lqg_solved["policy"]
    value = lqg_solved["value"]

    test_states = [
        (-2.0, -2.0), (-1.0, -1.0), (-0.5, 0.5), (0.0, 0.0),
        (0.5, -0.5), (1.0, 1.0), (1.5, -1.5), (2.0, 2.0),
    ]

    print()
    print("LQG  |  A=[[0.9,0.1],[0.05,0.85]], B=[1,0.5], C=[0.1,0.05], "
          "gamma=0.95, T=15")
    print()
    print(f"{'x1':>6} {'x2':>6} {'V bellgrid':>12} {'V riccati':>12}"
          f" {'Δ V':>11} {'u bellgrid':>12} {'u riccati':>12} {'Δ u':>11}")
    print("-" * 86)

    max_v_err = 0.0
    max_u_err = 0.0
    for (x1, x2) in test_states:
        x = np.array([x1, x2])
        v_closed = _lqg_value(P0, c0, x)
        u_closed = _lqg_action(K0, x)
        state_q = {
            "x1": torch.tensor([x1], dtype=torch.float64),
            "x2": torch.tensor([x2], dtype=torch.float64),
        }
        v_bg = value(state_q, t=0).item()
        u_bg = policy(state_q, t=0)["u"].item()
        v_err = v_bg - v_closed
        u_err = u_bg - u_closed
        print(f"{x1:>6.2f} {x2:>6.2f} {v_bg:>12.5f} {v_closed:>12.5f}"
              f" {v_err:>+11.2e} {u_bg:>12.5f} {u_closed:>12.5f} {u_err:>+11.2e}")
        max_v_err = max(max_v_err, abs(v_err))
        max_u_err = max(max_u_err, abs(u_err))

    print()
    print(f"Max |Δ V| = {max_v_err:.4e}   Max |Δ u| = {max_u_err:.4e}")
    print()

    assert max_v_err < 0.2
    assert max_u_err < 0.15


# --- iLQG: exact on LQ, no grid -----------------------------------------
#
# On an exactly linear-quadratic Problem, iLQG is a single Newton step on an
# exact quadratic model, so it reproduces the closed-form Riccati value AND
# policy to machine precision — a far tighter bar than the grid solver's
# O(h^2) interpolation bias (5% above), and at any state dimension.


@pytest.fixture(scope="module")
def lqg_ilqg_solved():
    problem = _build_lqg_problem()
    policy, value = solve(problem, solver=iLQG(), device="cpu", dtype=torch.float64)
    P_list, K_list, c_list = _riccati_lqg(
        A_MAT, B_MAT, C_MAT, Q_MAT, R_MAT, GAMMA, T_HORIZON
    )
    return {"policy": policy, "value": value, "P": P_list, "K": K_list, "c": c_list}


@pytest.mark.parametrize("t", [0, 7, 14])
@pytest.mark.parametrize("x_pair", [(0.5, 0.5), (-0.5, 1.0), (1.5, -1.0), (-2.0, -2.0)])
def test_ilqg_value_matches_riccati_to_machine_precision(lqg_ilqg_solved, x_pair, t):
    x = np.array(x_pair, dtype=np.float64)
    v_closed = _lqg_value(lqg_ilqg_solved["P"][t], lqg_ilqg_solved["c"][t], x)
    state_q = {
        "x1": torch.tensor([x[0]], dtype=torch.float64),
        "x2": torch.tensor([x[1]], dtype=torch.float64),
    }
    v_il = lqg_ilqg_solved["value"](state_q, t=t).item()
    # Riccati value (incl. the noise constant c_t) reproduced to ~1e-9.
    assert v_il == pytest.approx(v_closed, rel=1e-8, abs=1e-8), (
        f"x={x_pair}, t={t}: iLQG V={v_il:.10f}, riccati V={v_closed:.10f}"
    )


@pytest.mark.parametrize("t", [0, 7, 14])
@pytest.mark.parametrize("x_pair", [(0.5, 0.5), (-0.5, 1.0), (1.5, -1.0), (-2.0, -2.0)])
def test_ilqg_policy_matches_riccati(lqg_ilqg_solved, x_pair, t):
    x = np.array(x_pair, dtype=np.float64)
    u_closed = _lqg_action(lqg_ilqg_solved["K"][t], x)
    state_q = {
        "x1": torch.tensor([x[0]], dtype=torch.float64),
        "x2": torch.tensor([x[1]], dtype=torch.float64),
    }
    u_il = lqg_ilqg_solved["policy"](state_q, t=t)["u"].item()
    # The feedback gain is exact; the residual is the sqrt(eps) nominal-offset
    # convergence floor.
    assert u_il == pytest.approx(u_closed, rel=1e-6, abs=1e-6), (
        f"x={x_pair}, t={t}: iLQG u={u_il:.8f}, riccati u={u_closed:.8f}"
    )


def test_ilqg_rejects_out_of_scope():
    """iLQG declines non-continuous / infinite-horizon problems with a pointer
    to the grid solver, rather than silently mis-solving."""
    from bellgrid import DiscreteState

    base = _build_lqg_problem()
    # infinite horizon
    inf = Problem(
        states=base.states, actions=base.actions, transition=base.transition,
        reward=base.reward, shocks=base.shocks, horizon=None, discount=GAMMA,
    )
    with pytest.raises(NotImplementedError, match="finite-horizon"):
        solve(inf, solver=iLQG(), device="cpu")

    # discrete state
    disc = Problem(
        states=[*base.states, DiscreteState("regime", n=2)],
        actions=base.actions, transition=base.transition, reward=base.reward,
        shocks=base.shocks, horizon=range(0, T_HORIZON), discount=GAMMA,
    )
    with pytest.raises(NotImplementedError, match="ContinuousState"):
        solve(disc, solver=iLQG(), device="cpu")


# --- PolicyGradient: pathwise gradient, near-optimal, no critic ----------
#
# The pathwise / analytic policy-gradient solver backpropagates the return
# through the differentiable model — no learned critic, no bootstrap, none of the
# overestimation machinery. It converges to a near-optimal policy (a trained net,
# so within a few percent of the closed-form Riccati, not machine precision).


@pytest.fixture(scope="module")
def lqg_pgrad_solved():
    problem = _build_lqg_problem()
    policy, value = solve(
        problem,
        solver=PolicyGradient(steps=300, batch=2048, hidden=(64, 64),
                              value_samples=2048, seed=0),
        device="cpu", dtype=torch.float64,
    )
    P_list, K_list, c_list = _riccati_lqg(
        A_MAT, B_MAT, C_MAT, Q_MAT, R_MAT, GAMMA, T_HORIZON
    )
    return {"policy": policy, "value": value, "P": P_list, "K": K_list, "c": c_list}


@pytest.mark.parametrize("x_pair", [(0.5, 0.5), (-0.5, 1.0), (1.5, -1.0), (-2.0, -2.0)])
def test_pgrad_converges_near_riccati(lqg_pgrad_solved, x_pair):
    x = np.array(x_pair, dtype=np.float64)
    state_q = {
        "x1": torch.tensor([x[0]], dtype=torch.float64),
        "x2": torch.tensor([x[1]], dtype=torch.float64),
    }
    # value (an honest MC rollout of the trained policy) within ~10% of Riccati
    v_pg = lqg_pgrad_solved["value"](state_q, t=0).item()
    v_ric = _lqg_value(lqg_pgrad_solved["P"][0], lqg_pgrad_solved["c"][0], x)
    assert v_pg == pytest.approx(v_ric, rel=0.12, abs=0.5), (
        f"x={x_pair}: pgrad V={v_pg:.4f}, riccati V={v_ric:.4f}"
    )
    # policy near the closed-form optimum
    u_pg = lqg_pgrad_solved["policy"](state_q, t=0)["u"].item()
    u_ric = _lqg_action(lqg_pgrad_solved["K"][0], x)
    assert abs(u_pg - u_ric) < 0.4, (
        f"x={x_pair}: pgrad u={u_pg:.4f}, riccati u={u_ric:.4f}"
    )


def test_pgrad_rejects_out_of_scope():
    """PolicyGradient needs a differentiable model: discrete states break the
    pathwise gradient, so it declines them with a pointer to the alternatives."""
    from bellgrid import DiscreteState

    base = _build_lqg_problem()
    disc = Problem(
        states=[*base.states, DiscreteState("regime", n=2)],
        actions=base.actions, transition=base.transition, reward=base.reward,
        shocks=base.shocks, horizon=range(0, T_HORIZON), discount=GAMMA,
    )
    with pytest.raises(NotImplementedError, match="ContinuousState"):
        solve(disc, solver=PolicyGradient(steps=1), device="cpu")
