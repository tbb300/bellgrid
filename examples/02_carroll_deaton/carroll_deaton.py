# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Carroll/Deaton lifecycle consumption-savings
#
# A finite-lived household with deterministic risk-free returns, stochastic
# labour income, CRRA utility, and a **borrowing constraint**. There's no
# closed form; the qualitative signature is a *kinked* consumption function
# at low cash-on-hand.
#
# ## Problem
#
# State: cash-on-hand $m_t$ (wealth + current income). Action:
# consumption $c_t \in [0, m_t]$ (the binding upper bound is the no-borrowing
# constraint). Dynamics:
#
# $$ m_{t+1} = R\,(m_t - c_t) + y_{t+1}, \qquad y_{t+1} = \mu_y + \sigma_y\,\varepsilon_{t+1}, \quad \varepsilon \sim \mathcal{N}(0,1). $$
#
# Reward: CRRA utility with risk aversion $\gamma$:
#
# $$ u(c) = \frac{c^{1-\gamma}}{1-\gamma}. $$
#
# Objective: maximise $\sum_{t=0}^{T-1} \beta^t\,u(c_t)$ with $R\beta < 1$
# (impatient household). The expected behaviour:
#
# - **Below the buffer-stock target**, the borrowing constraint binds:
#   $c_t \approx m_t$ and MPC $\approx 1$.
# - **Above the target**, the household saves: $c_t < m_t$, MPC $< 1$
#   (typically 0.1 – 0.5 in standard calibrations).
# - Forward-simulated wealth accumulates from low initial cash toward a
#   stationary buffer-stock distribution.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

from bellgrid import (
    ContinuousAction,
    ContinuousState,
    Problem,
    simulate,
    solve,
)
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction


# %% [markdown]
# ## Parameters

# %%
gamma = 2.0    # CRRA risk aversion
R = 1.04       # gross risk-free return
beta = 0.94    # discount factor (R*beta = 0.978 < 1 → impatient → buffer-stock saver)
mu_y = 1.0     # mean labor income
sigma_y = 0.1  # income innovation std
T = 25         # life-cycle horizon


# %% [markdown]
# ## Bellgrid problem

# %%
def transition(state, action, shock, _t):
    savings = state["cash"] - action["consume"]
    next_income = mu_y + sigma_y * shock["z"]
    return {"cash": R * savings + next_income}


def reward(_state, action, _shock, _t):
    c = action["consume"]
    if gamma == 1.0:
        return torch.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def terminal_reward(state):
    # Natural finite-horizon convention: the agent consumes everything
    # remaining at expiry, so V_T(m) = u(m). Without this the agent has
    # no reason to save in the final period, which warps the entire
    # backward sweep and prevents an apples-to-apples comparison with EGM.
    c = state["cash"]
    if gamma == 1.0:
        return torch.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


problem = Problem(
    states=[ContinuousState("cash", warp="asinh", range=(0.5, 20.0))],
    actions=[ContinuousAction("consume", bounds=(1e-4, "cash"))],
    transition=transition,
    reward=reward,
    shocks=[Normal("z", sigma=1.0)],
    horizon=range(0, T),
    discount=beta,
    terminal_reward=terminal_reward,
)

policy, value = solve(
    problem,
    state_grid={"cash": WarpedGrid(n=128)},
    action_grid={"consume": RegularGrid(n=500)},
    solver=BackwardInduction(n_quad=7),
)


# %% [markdown]
# ## Reference: Endogenous Grid Method (Carroll 2006)
#
# Carroll/Deaton has no closed form. The canonical numerical benchmark is
# the **endogenous grid method**: instead of gridding cash-on-hand and
# searching for the optimal action at each point, you grid the
# **post-decision wealth** $a = m - c$ and use the Euler equation to
# back out the implied consumption at each $a$.
#
# At each $t$, for every post-decision $a$:
#
# $$ c_t^*(a) = \Bigl(\,\beta R\,\mathbb{E}\bigl[u'(c_{t+1}(R a + y))\bigr]\,\Bigr)^{-1/\gamma}, $$
#
# $$ m_t^*(a) = c_t^*(a) + a. $$
#
# That gives an *endogenous* grid of $(m_t, c_t)$ pairs from which the
# consumption function is interpolated. The borrowing constraint is baked
# in: $a = 0$ produces $m_t = c_t$, and below that threshold the agent
# consumes all of cash. EGM is the de facto reference for this problem
# class — independent of bellgrid's grid-on-cash approach, so the two
# should overlap tightly if both are right.

# %%
def egm_carroll_deaton(gamma, R, beta, mu_y, sigma_y, T, n_a=400, a_max=30.0, n_quad=7):
    """Endogenous Grid Method for Carroll/Deaton with i.i.d. Normal income."""
    raw_z, raw_w = np.polynomial.hermite_e.hermegauss(n_quad)
    w_quad = raw_w / np.sqrt(2 * np.pi)
    y_nodes = mu_y + sigma_y * raw_z

    a_grid = np.geomspace(1e-4, a_max, n_a)

    m_endo = np.linspace(0.0, 50.0, 500)
    c_endo = m_endo.copy()

    policies = [None] * (T + 1)
    policies[T] = (m_endo, c_endo)

    for t in range(T - 1, -1, -1):
        m_next, c_next = policies[t + 1]
        m_implied = np.zeros(n_a)
        c_implied = np.zeros(n_a)
        for i, a in enumerate(a_grid):
            m_realizations = R * a + y_nodes
            c_realizations = np.interp(m_realizations, m_next, c_next)
            E_up_next = (w_quad * c_realizations ** (-gamma)).sum()
            c_t = (beta * R * E_up_next) ** (-1.0 / gamma)
            m_implied[i] = c_t + a
            c_implied[i] = c_t

        # Prepend (0, 0) so linear interp recovers c = m in the constraint region.
        m_full = np.concatenate(([0.0], m_implied))
        c_full = np.concatenate(([0.0], np.minimum(c_implied, m_implied)))
        policies[t] = (m_full, c_full)

    return policies


def egm_consumption(policies, t, m_query):
    m_endo, c_endo = policies[t]
    c = np.interp(m_query, m_endo, c_endo)
    return np.minimum(c, m_query)


egm_policies = egm_carroll_deaton(gamma, R, beta, mu_y, sigma_y, T)


# %% [markdown]
# ## Consumption function
#
# At low cash the no-borrowing constraint $c \le m$ binds and the
# household consumes essentially all of it (45° line). Above the
# buffer-stock threshold the household saves and the curve bends
# away. Bellgrid and EGM overlap visually; the residual plot shows
# the actual agreement.

# %%
cash_np = np.linspace(0.5, 18.0, 200)
consume_bg = policy(
    {"cash": torch.tensor(cash_np, dtype=torch.float64)}, t=T // 2
)["consume"].numpy()
consume_eg = egm_consumption(egm_policies, T // 2, cash_np)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(cash_np, consume_bg, lw=2.5, label="bellgrid")
ax1.plot(cash_np, consume_eg, ls="--", lw=2, color="C1", label="EGM (Carroll 2006)")
ax1.plot(cash_np, cash_np, color="C3", lw=1, alpha=0.6, label="constraint $c = m$")
ax1.set_xlabel("cash-on-hand $m$")
ax1.set_ylabel("consumption $c^*$")
ax1.set_title("Consumption function (mid-horizon)")
ax1.legend()
ax1.grid(alpha=0.3)

residual = consume_bg - consume_eg
ax2.plot(cash_np, residual, lw=2, color="C2")
ax2.axhline(0.0, color="black", lw=0.5)
ax2.set_xlabel("cash-on-hand $m$")
ax2.set_ylabel("$c_{bellgrid} - c_{EGM}$")
ax2.set_title(f"Residual  (max |Δ| = {np.abs(residual).max():.2e})")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Marginal propensity to consume
#
# $\mathrm{MPC}(m) = \partial c^*(m)/\partial m$. We compute the
# numerical derivative on a moderately coarse sample (≈0.3 spacing) so
# the underlying piecewise-linear grid segments average out cleanly.
# Approaches 1 at low cash (constraint binding) and drops below 0.5
# quickly. Standard calibrations in the literature report MPC in the
# 0.2 – 0.5 range at typical wealth levels.

# %%
cash_mpc = np.linspace(0.5, 18.0, 60)
c_bg_mpc = policy(
    {"cash": torch.tensor(cash_mpc, dtype=torch.float64)}, t=T // 2
)["consume"].numpy()
c_eg_mpc = egm_consumption(egm_policies, T // 2, cash_mpc)
mpc_bg = np.gradient(c_bg_mpc, cash_mpc)
mpc_eg = np.gradient(c_eg_mpc, cash_mpc)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(cash_mpc, mpc_bg, lw=2.5, label="bellgrid")
ax.plot(cash_mpc, mpc_eg, ls="--", lw=2, color="C1", label="EGM")
ax.axhline(1.0, color="C3", ls="--", lw=1.0, alpha=0.6, label="MPC = 1 (constrained)")
ax.set_xlabel("cash-on-hand $m$")
ax.set_ylabel("MPC")
ax.set_ylim(-0.05, 1.1)
ax.set_title("Marginal propensity to consume")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Forward simulation: buffer-stock dynamics
#
# Starting from cash near the constraint, the household accumulates a
# buffer over time. Mean cash rises and stabilizes around a stationary
# level — this is the Carroll target.

# %%
T_long = 50
problem_long = Problem(
    states=problem.states,
    actions=problem.actions,
    transition=transition,
    reward=reward,
    shocks=problem.shocks,
    horizon=range(0, T_long),
    discount=beta,
    terminal_reward=terminal_reward,
)
policy_long, _ = solve(
    problem_long,
    state_grid={"cash": WarpedGrid(n=128)},
    action_grid={"consume": RegularGrid(n=500)},
    solver=BackwardInduction(n_quad=7),
)

paths = simulate(
    policy=policy_long,
    problem=problem_long,
    n=2000,
    initial_state={"cash": 0.8},
    seed=0,
)
paths_cpu = {k: v.cpu() for k, v in paths.items()}

cash_paths = paths_cpu["cash"].numpy()
t_axis = np.arange(T_long)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(t_axis, cash_paths[:50].T, color="C0", alpha=0.2)
ax1.plot(t_axis, cash_paths.mean(axis=0), color="C1", lw=2.5, label="mean")
ax1.set_xlabel("period $t$")
ax1.set_ylabel("cash-on-hand $m_t$")
ax1.set_title(f"Cash trajectories (showing 50 of {len(cash_paths)} paths)")
ax1.legend()
ax1.grid(alpha=0.3)

# Distribution of cash at end of horizon
ax2.hist(cash_paths[:, -1], bins=40, density=True, alpha=0.7, color="C0")
ax2.axvline(cash_paths[:, -1].mean(), color="C1", lw=2, label=f"mean = {cash_paths[:, -1].mean():.2f}")
ax2.set_xlabel("cash $m_T$ at end of horizon")
ax2.set_ylabel("density")
ax2.set_title("Stationary buffer distribution")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()
