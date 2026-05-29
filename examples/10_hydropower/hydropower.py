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
# # Multi-reservoir hydropower under a stochastic price — the neural solver
#
# This is the example the grid solver **cannot** do, and the one the neural
# `ActorCritic` solver was built for.
#
# A cascade of $N$ reservoirs sits on a river. Each period you choose how much
# water to **release** through each dam's turbines — selling power at the
# *current* electricity price — versus **store** for later. Released water
# cascades into the reservoir below; storage is capped, so holding too much
# spills, holding too little risks running dry. Two things are random:
#
# - **inflows** — a common hydrological "wetness" factor (a wet year fills the
#   whole basin), and
# - **the price** — a mean-reverting (Ornstein–Uhlenbeck / AR(1)) log-price.
#   You see today's price and must bet on an *uncertain* future: sell now, or
#   hold expecting reversion toward the mean and the occasional spike?
#
# The stochastic price is what makes this genuinely interesting: the optimal
# release is no longer a fixed seasonal schedule but a **policy that reacts to
# the realized price** — so different sample paths release at different times.
#
# Each reservoir is a continuous **state** and each release a continuous
# **action**; the price adds one more continuous state. A grid solver needs
# $(\text{pts})^{N+1}\times(\text{pts})^{N}$ cells — ~$10^{14}$ at $N=4$,
# physically impossible. The neural solver represents $V$ and $\pi$ as networks
# over *sampled* states, so its cost is essentially flat in dimension.
#
# ## The correctness contract
#
# The neural solution is *approximate*, so we don't just trust it. We:
#
# 1. **Certify against the exact grid solver at $N=1$** (a 2-D state — reservoir
#    level × price — still griddable): same `Problem`, two solvers, compare.
# 2. **Run at $N=4$ over a 5-year horizon, where no grid can exist** (a 5-D
#    state), and self-validate by Monte-Carlo consistency: because the critic is
#    trained *on-policy* and *on-distribution* (`ergodic=True`), the value it
#    reports must equal the expected discounted return earned by *simulating its
#    own learned policy* — exactly what `bellgrid.simulate()` returns as
#    `discounted_total`. (`V_solve(s_0) ≈ E[∑ β^t r_t]`.) This agreement holds at
#    1% and **does not degrade with the horizon** — see the note in §2.

# %%
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from bellgrid import ContinuousAction, ContinuousState, Problem, simulate, solve
from bellgrid.grids import RegularGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction
from bellgrid.rl import ActorCritic

torch.manual_seed(0)

# %% [markdown]
# ## Parameters
#
# The log-price follows a mean-reverting AR(1) process,
#
# $$\log p_{t+1} = \theta + \phi\,(\log p_t - \theta) + \sigma_p\,\varepsilon,$$
#
# with $\theta = 0$ (mean price 1), persistence $\phi = 0.6$, and innovation
# $\sigma_p = 0.4$ — a stationary price spread of roughly $[0.4, 2.5]$. Inflows
# share a single lognormal wetness factor. Two shocks total, so the expectation
# is a $4 \times 4 = 16$-node tensor-product quadrature (well under the
# shock-dimension limit; many *independent* shocks would need a Monte-Carlo
# expectation instead).

# %%
CAP, SIG = 1.0, 0.4                 # reservoir capacity, inflow wetness vol
THETA, PHI, SIG_P = 0.0, 0.6, 0.4   # log-price OU: mean, AR(1) persistence, innovation std
LOGP_LO, LOGP_HI = -2.2, 2.2        # ~ ±4 stationary std of log-price
BETA = 0.99
T = 60                              # planning horizon: 5 years, monthly


def build(N):
    """An N-reservoir cascade with a stochastic OU price.

    States: `lvl_i` ∈ [0, CAP] storage in reservoir i, plus `logp` the
    log-price. Action `rel_i` is bounded by current storage (state-dependent —
    you cannot release water you do not hold). Released water cascades down;
    storage above CAP spills. The price evolves exogenously (a price-taker).
    """
    base = [0.20] + [0.08] * (N - 1)   # top reservoir gets the natural inflow

    def transition(s, a, sh, t):
        wet = torch.exp(SIG * sh["z"] - 0.5 * SIG ** 2)   # mean-1 lognormal wetness
        nxt, upstream = {}, 0.0
        for i in range(N):
            raw = s[f"lvl{i}"] - a[f"rel{i}"] + base[i] * wet + upstream
            nxt[f"lvl{i}"] = torch.clamp(raw, max=CAP)     # spill above capacity
            upstream = a[f"rel{i}"]                        # cascades downstream
        nxt["logp"] = THETA + PHI * (s["logp"] - THETA) + SIG_P * sh["eps"]
        return nxt

    def reward(s, a, sh, t):
        generation = sum(a[f"rel{i}"] for i in range(N))
        return torch.exp(s["logp"]) * generation           # sell at the current price

    # No terminal reward: a finite operating season where leftover water has no
    # salvage value. Then V_0 is exactly the expected discounted operating
    # revenue, Σ β^t r_t — which is precisely what `simulate()` reports, so the
    # consistency check below is a one-liner with the library's own simulator.
    states = [ContinuousState(f"lvl{i}", range=(0.0, CAP)) for i in range(N)]
    states.append(ContinuousState("logp", range=(LOGP_LO, LOGP_HI)))
    return Problem(
        states=states,
        actions=[ContinuousAction(f"rel{i}", bounds=(0.0, f"lvl{i}")) for i in range(N)],
        transition=transition, reward=reward,
        shocks=[Normal("z", sigma=1.0), Normal("eps", sigma=1.0)],
        horizon=range(0, T), discount=BETA,
    )


# Shared neural-solver configuration for every N below. `ergodic=True` (the
# default) is what keeps the value accurate over this 5-year horizon: it trains
# the critic on the states the policy actually visits (a second pass over a
# simulated state buffer), so the 1-step bootstrap doesn't compound
# off-distribution error down the 60-step backward sweep. With uniform-only
# sampling the self-consistency gap would balloon to ~15% at this horizon;
# on-distribution it stays ~1%.
AC = ActorCritic(
    n_quad=4, hidden=(128, 128), state_samples=1024, steps=150,
    lr=3e-3, n_global=10, n_local=10, ergodic=True, seed=0,
)


# %% [markdown]
# ## Why a grid can't do this
#
# A grid stores $V$ on a mesh and enumerates a tensor product of action grids,
# so both its memory and its per-step work grow as $(\text{points})^{\text{dim}}$
# — and the OU price adds a whole state dimension on top of the $N$ reservoirs.
# The network's size is essentially flat in dimension.

# %%
Ns = np.arange(1, 9)
state_pts, action_pts = 56, 28
v_storage = state_pts ** (Ns + 1).astype(float)              # cells to store V (N reservoirs + price)
compute = state_pts ** (Ns + 1) * action_pts ** Ns.astype(float)   # state × action work
neural = np.full_like(Ns, 6e4, dtype=float)                  # ~ net params × 2 nets × T

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.semilogy(Ns, v_storage, "C0o-", lw=2, label="grid: cells to store V  (= pts$^{N+1}$)")
ax.semilogy(Ns, compute, "C3s-", lw=2, label="grid: state×action Bellman work")
ax.semilogy(Ns, neural, "C2^-", lw=2, label="neural: parameters (≈ flat in N)")
ax.axhline(2e9, color="gray", ls=":", lw=1.5)
ax.text(1, 2.6e9, "16 GB GPU (fp64)", fontsize=8, color="gray")
ax.axhline(1e15, color="gray", ls=":", lw=1.5)
ax.text(1, 1.3e15, "a large datacenter", fontsize=8, color="gray")
ax.scatter([4], [compute[3]], s=120, color="C3", zorder=5)
ax.annotate(f"N=4: {compute[3]:.0e} cells\n→ physically impossible", (4, compute[3]),
            (1.5, 1e22), fontsize=9, color="C3", arrowprops=dict(arrowstyle="->", color="C3"))
ax.scatter([4], [neural[3]], s=120, color="C2", zorder=5)
ax.annotate("neural: ~10⁴ params,\nsolved in ~minutes", (4, neural[3]), (4.5, 5e6),
            fontsize=9, color="C2", arrowprops=dict(arrowstyle="->", color="C2"))
ax.set_xlabel("number of reservoirs  N   (state dimension = N reservoirs + 1 price)")
ax.set_ylabel("count (log scale)")
ax.set_title("Why a neural solver: the grid explodes exponentially in dimension,\nthe network does not")
ax.legend(loc="center right", fontsize=9)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1. Certify at N = 1 against the exact grid solver
#
# One reservoir + the price is a 2-D state — still griddable. We solve it both
# ways and confirm the neural value and the *price-responsive* release policy
# match the grid.

# %%
problem1 = build(1)
pol_g, val_g = solve(
    problem1,
    state_grid={"lvl0": RegularGrid(n=96), "logp": RegularGrid(n=96)},
    action_grid={"rel0": RegularGrid(n=96)},
    solver=BackwardInduction(n_quad=4, boundary_check=False),
)
t0 = time.time()
pol_n, val_n = solve(problem1, solver=AC)
print(f"N=1 neural solve: {time.time() - t0:.0f}s")

lv = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
for logp in (-0.8, 0.0, 0.8):
    q = {"lvl0": lv, "logp": torch.full_like(lv, logp)}
    vg, vn = val_g(q, 30), val_n(q, 30)
    rg, rn = pol_g(q, 30)["rel0"], pol_n(q, 30)["rel0"]
    print(f" price={math.exp(logp):.2f}  V grid {[f'{v:.3f}' for v in vg.tolist()]}"
          f"  neural {[f'{v:.3f}' for v in vn.tolist()]}  | rel grid {[f'{r:.2f}' for r in rg.tolist()]}"
          f"  neural {[f'{r:.2f}' for r in rn.tolist()]}")

# %% [markdown]
# The policy is the stochastic arbitrage: at a **low price** hold the water
# (wait for reversion / a spike); at a **high price** release it. The right
# panel below shows that price-response directly — release as a function of the
# current price — with neural (dashed) overlaying the exact grid (solid).

# %%
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))
# Value vs storage at low / high price.
dense_lvl = torch.linspace(0.01, 1.0, 120, dtype=torch.float64)
for logp, c, lab in [(0.8, "C3", "high price (p=2.2)"), (-0.8, "C0", "low price (p=0.45)")]:
    qd = {"lvl0": dense_lvl, "logp": torch.full_like(dense_lvl, logp)}
    axA.plot(dense_lvl, val_g(qd, 30), color=c, lw=4, alpha=0.35)
    axA.plot(dense_lvl, val_n(qd, 30), color=c, ls="--", lw=1.8, label=lab)
axA.plot([], [], color="gray", lw=4, alpha=0.35, label="grid (exact)")
axA.plot([], [], color="gray", ls="--", lw=1.8, label="neural")
axA.set(title="Value of stored water (N=1)", xlabel="storage", ylabel="value V")
axA.legend(fontsize=8); axA.grid(alpha=0.3)
# Release vs current price at fixed storage levels — the price-responsive policy.
dense_p = torch.linspace(LOGP_LO + 0.2, LOGP_HI - 0.2, 120, dtype=torch.float64)
for lvl0, c in [(0.3, "C1"), (0.6, "C4"), (0.9, "C2")]:
    qd = {"lvl0": torch.full_like(dense_p, lvl0), "logp": dense_p}
    axB.plot(torch.exp(dense_p), pol_g(qd, 30)["rel0"], color=c, lw=4, alpha=0.35)
    axB.plot(torch.exp(dense_p), pol_n(qd, 30)["rel0"], color=c, ls="--", lw=1.8,
             label=f"storage = {lvl0}")
axB.set(title="Release vs current price — hold when cheap, sell when dear\n(neural dashed overlays grid solid)",
        xlabel="current price", ylabel="optimal release")
axB.legend(fontsize=8); axB.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. N = 4 — where the grid cannot exist
#
# Four reservoirs plus the price: a 5-D state, over a 5-year (60-month) horizon.
# An equivalently-resolved grid needs the product below — hundreds of trillions
# of cells. The neural solver handles it in a few minutes.

# %%
N = 4
problem4 = build(N)
equiv_cells = state_pts ** (N + 1) * action_pts ** N
print(f"Equivalent grid cells at N={N}: {equiv_cells:.2e}  (physically impossible)")

t0 = time.time()
pol4, val4 = solve(problem4, solver=AC)
print(f"N={N} neural solve: {time.time() - t0:.0f}s")
print("per-period critic RMSE:",
      {k: round(v, 3) for k, v in sorted(val4.residual_by_t.items())})

# %% [markdown]
# ### Self-validation by Monte-Carlo consistency
#
# With no grid to compare against, we check the solver against itself *through
# the model*, using `bellgrid.simulate()` — the same simulator any user would
# call. Because the critic is trained on-policy *and on-distribution*
# (`ergodic=True`), the value it reports at the initial state should equal the
# expected discounted return of simulating its learned policy. They agree to
# ~1% — and, crucially, that gap **does not grow with the horizon**: it's ~1%
# here at 60 months just as at 10. (With uniform-only sampling the critic
# mis-fits the operating region and the 1-step bootstrap compounds that error
# down the 60-step sweep, ballooning the gap to ~15%; on-distribution sampling
# is what removes it. Try `ergodic=False` to see the difference.)

# %%
s0 = 0.0   # start with all reservoirs empty
init = {**{f"lvl{i}": s0 for i in range(N)}, "logp": THETA}
v_solve = val4({k: torch.tensor([v]) for k, v in init.items()}, 0).item()

# `simulate()` rolls the learned policy forward under the same transition,
# reward, and discount; `discounted_total` is exactly the Σ β^t r_t that V_0
# equals here. No hand-rolled rollout — the library's own simulator IS the check.
paths = simulate(policy=pol4, problem=problem4, n=40_000, initial_state=init, seed=0)
ret = paths["discounted_total"]
v_mc = ret.mean().item()
half = 1.96 * ret.std().item() / math.sqrt(len(ret))
print(f"V_solve(s0={s0}) = {v_solve:.3f}")
print(f"V_mc   (s0={s0}) = {v_mc:.3f}  ±{half:.3f}  (95% CI, 40k simulated paths)")
print(f"gap = {abs(v_solve - v_mc):.3f}  ({abs(v_solve - v_mc) / v_mc * 100:.1f}%)")

# %% [markdown]
# ### The learned V and π are clean
#
# Without a grid to compare against in 5-D, we also want to *see* that the
# solved functions are well-behaved, not noisy fits. Slice the N=4 solution
# over (reservoir-0 storage × price), holding the other three reservoirs at 0.5.
# The value is a smooth surface increasing in both storage and price; the
# release policy is a smooth price-threshold capped by available storage —
# exactly the shape the economics demands, with no speckle or wiggle.

# %%
n = 70
lv = torch.linspace(0.01, 1.0, n, dtype=torch.float64)
lp = torch.linspace(LOGP_LO, LOGP_HI, n, dtype=torch.float64)
LV, LP = torch.meshgrid(lv, lp, indexing="ij")
qh = {f"lvl{i}": torch.full((n * n,), 0.5, dtype=torch.float64) for i in range(N)}
qh["lvl0"] = LV.reshape(-1)
qh["logp"] = LP.reshape(-1)
ts = 30   # a mid-horizon period
Vh = val4(qh, ts).reshape(n, n).cpu().numpy()
Rh = pol4(qh, ts)["rel0"].reshape(n, n).cpu().numpy()
pr = torch.exp(lp).numpy()

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
im = axes[0, 0].pcolormesh(pr, lv.numpy(), Vh, shading="auto", cmap="viridis")
axes[0, 0].set(title="Value V(storage₀, price)   [other reservoirs = 0.5]",
               xlabel="current price", ylabel="reservoir-0 storage")
fig.colorbar(im, ax=axes[0, 0])
im = axes[0, 1].pcolormesh(pr, lv.numpy(), Rh, shading="auto", cmap="magma")
axes[0, 1].set(title="Policy release₀(storage₀, price)",
               xlabel="current price", ylabel="reservoir-0 storage")
fig.colorbar(im, ax=axes[0, 1])
for s0, c in [(0.2, "C0"), (0.5, "C1"), (0.8, "C2")]:
    j = int(abs(lv - s0).argmin())
    axes[1, 0].plot(pr, Vh[j, :], color=c, lw=2, label=f"storage₀={s0}")
axes[1, 0].set(title="V vs price (slices) — smooth, increasing, convex",
               xlabel="current price", ylabel="V")
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
for lpv, c in [(-0.8, "C0"), (0.0, "C1"), (0.8, "C2")]:
    k = int(abs(lp - lpv).argmin())
    axes[1, 1].plot(lv.numpy(), Rh[:, k], color=c, lw=2, label=f"price={math.exp(lpv):.2f}")
axes[1, 1].set(title="release₀ vs own storage (slices) — monotone",
               xlabel="reservoir-0 storage", ylabel="release₀")
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### The payoff: the policy *adapts to the realized price*
#
# This is what the stochastic price buys us. We plot a single simulated path:
# the price wanders (mean-reverting, with spikes), and **total release tracks
# it** — the policy stores through cheap stretches and sells into expensive
# ones, *as they happen*, not on a fixed schedule. Different random paths
# therefore release at different times. The right panel shows the four
# reservoirs drawing down in a coordinated cascade along that path.

# %%
# Again just `simulate()` — draw a batch of paths and pull one out to plot.
# Start empty: the reservoirs fill from inflow, then operate.
sim = simulate(policy=pol4, problem=problem4, n=256,
               initial_state={**{f"lvl{i}": 0.0 for i in range(N)}, "logp": THETA}, seed=7)
sim = {key: v.cpu() for key, v in sim.items()}   # solve/simulate may run on GPU
k = 0   # one representative 5-year path
months = np.arange(T)
path_price = torch.exp(sim["logp"][k]).numpy()
path_release = sum(sim[f"rel{i}"][k] for i in range(N)).numpy()
path_per = np.stack([sim[f"lvl{i}"][k].numpy() for i in range(N)], axis=1)   # [T, N]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))
ax2 = axA.twinx()
# 60 monthly points — lines/fill read better than bars over a 5-year path.
axA.fill_between(months, path_release, color="C1", alpha=0.4, step="mid",
                 label="total release (generate)")
axA.plot(months, path_release, color="C1", lw=1.0, alpha=0.8)
ax2.plot(months, path_price, "k-", lw=1.8, label="realized price (stochastic)")
ax2.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
axA.set_xlabel("month"); axA.set_ylabel("total release"); ax2.set_ylabel("price")
axA.set_title("One 5-year sample path: release tracks the *realized* price\n"
              "(empty → fill → sell into the spikes, hold through the lulls)")
l1, lab1 = axA.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
axA.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=8)
for i in range(N):
    axB.plot(months, path_per[:, i], lw=1.4, label=f"reservoir {i}")
axB.set_xlabel("month"); axB.set_ylabel("storage")
axB.set_title("Same path: 4 coupled reservoirs (empty → fill → cascade)")
axB.legend(fontsize=8); axB.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Takeaway
#
# Same `Problem`, same `solve()` call — one swap of `solver=BackwardInduction(...)`
# for `solver=ActorCritic(...)`. The grid solver certifies the neural one at
# $N=1$; at $N=4$ the grid is physically impossible and the neural solver
# self-validates against its own forward simulation to ~1% — over a full
# 5-year horizon — thanks to the on-policy, on-distribution (`ergodic`) critic.
# The stochastic OU price is what makes it a real control problem: the optimal
# release is a policy that *reacts* to the price you observe, not a schedule you
# could write down in advance.
#
# Caveats, repeated from the docs: the neural solution is approximate and
# optimisation-trained, so check it against the grid wherever both run and
# against forward simulation where it can't, watch `value.residual_by_t`, and
# prefer the grid solver outright on low-dimensional problems (it is exact and
# far faster there). The neural solver earns its keep in the dimensions the grid
# cannot reach.
