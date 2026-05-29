# bellgrid

**Solve stochastic dynamic programs on the GPU.** Mixed continuous and discrete state, mixed continuous and discrete action. Two solvers behind one `Problem` spec: **exact** GPU backward induction, and a **model-based neural solver** for the high dimensions a grid can't reach — with the exact solver doubling as a built-in correctness oracle for the neural one.

A bellgrid `Problem` specifies:

- a **state** `s` and **action** `a`, each a tuple of continuous-real components and/or discrete-categorical components (including Markov-chain indices) in any combination;
- a **transition** `s' = f(s, a, w, t)` mapping a current state, action, shock realisation, and time index to a next state;
- a **per-period reward** `r(s, a, w, t)` (optionally also a function of `s'`);
- a **shock distribution** `w ~ p` — Normal, Lognormal, MultivariateNormal, Uniform, Categorical, or Jump, individually or jointly via tensor product;
- a **discount** `β(s, t)` (scalar or callable);
- a **planning horizon** `t ∈ {0, 1, …, T}` (finite) or `t = ∞` (stationary).

bellgrid solves the [Bellman recursion](https://en.wikipedia.org/wiki/Bellman_equation)

```
V_t(s) = max_a  E_w[ r(s, a, w, t) + β(s, t) · V_{t+1}( f(s, a, w, t) ) ]
```

over a user-chosen state mesh, evaluating the shock expectation by quadrature appropriate to the shock type — Gauss-Hermite for `Normal` and `Lognormal`, Cholesky-rotated tensor-product Gauss-Hermite for `MultivariateNormal`, Gauss-Legendre for `Uniform`, exact for `Categorical`, Bernoulli arrival + Gauss-Hermite over magnitude for `Jump` — and interpolating `V_{t+1}` multilinearly at next-state coordinates. Finite-horizon problems sweep backward from a user-supplied terminal `V_T(s)`; infinite-horizon problems iterate to convergence under a stationary policy.

Every example in this repo is validated against an analytical or numerical reference: log-utility Merton matches the closed form to machine precision, the LQG case bit-for-bit matches the Riccati recursion, the American put matches a high-resolution binomial tree to ~1e-4, Merton (1976) jump-diffusion matches the series expansion to ~1e-3.

## Why the GPU matters

A realistic lifecycle problem — wealth × employment phase × regime, joint consumption + retire + asset-allocation decision, mortality with bequest, 75 periods, ~1.2 billion Bellman cell-action-shock evaluations end to end — solves in **1.8 seconds** on a single GPU. The same workload takes **80 seconds** on a 32-core CPU.

That gap is what turns calibration from an overnight job into something interactive. Sweep five parameters on a 10×10×10×10×10 grid? 50 hours on CPU, 25 minutes on GPU.

| Problem | grid × actions × shocks × horizon | CPU (Threadripper 5975WX) | GPU (RTX 6000 Ada) | speedup |
|---|---|---|---|---|
| Toy Merton | 128 × 256 × 5 × 20T = 3.3M ops | 77 ms | 6 ms | 14× |
| Medium Merton | 512 × 1k × 7 × 20T = 72M ops | 1.1 s | 21 ms | 52× |
| Large Merton | 1k × 2k × 11 × 20T = 451M ops | 6.4 s | 123 ms | 52× |
| Big Merton | 2k × 4k × 15 × 20T = 2.5B ops | 38 s | 0.84 s | 45× |
| **Full lifecycle DP** | **960 × 2400 × 7 × 75T = 1.2B ops** | **80 s** | **1.8 s** | **45×** |

CPU times use `torch` with 32 threads. Both backends share the exact same `Problem` definition — `device='cuda'` is the only line that changes.

## Quick start

```bash
pip install bellgrid                                                # CPU-only torch from PyPI
pip install bellgrid --extra-index-url https://download.pytorch.org/whl/cu126   # GPU
```

A minimal Merton consumption-portfolio (log utility, lognormal returns):

```python
import math, torch
from bellgrid import Problem, ContinuousState, ContinuousAction, solve
from bellgrid.grids import WarpedGrid, RegularGrid
from bellgrid.shocks import Normal
from bellgrid.solvers import BackwardInduction

beta, mu, sigma = 0.96, 0.04, 0.15
# closed-form coefficients for V(w) = A + B log(w)
B = 1.0 / (1.0 - beta)
A = math.log(1 - beta) / (1 - beta) + (beta / (1 - beta) ** 2) * (math.log(beta) + mu)

def transition(state, action, shock, t):
    return {"wealth": (state["wealth"] - action["consume"]) * torch.exp(mu + sigma * shock["z"])}

def reward(state, action, shock, t):
    return torch.log(action["consume"])

problem = Problem(
    states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 200.0))],
    actions=[ContinuousAction("consume", bounds=(1e-6, "wealth"))],
    transition=transition,
    reward=reward,
    shocks=[Normal("z", sigma=1.0)],
    horizon=range(0, 20),
    discount=beta,
    terminal_reward=lambda state: A + B * torch.log(state["wealth"]),
)

policy, value = solve(
    problem,
    state_grid={"wealth": WarpedGrid(n=128)},
    action_grid={"consume": RegularGrid(n=500)},
    solver=BackwardInduction(n_quad=7),
)

# Optimal consumption rate at any wealth ≈ 1 - β = 0.040
w = torch.tensor([2.0, 10.0, 25.0, 50.0])
policy({"wealth": w}, t=10)["consume"] / w
# → tensor([0.0401, 0.0401, 0.0401, 0.0401])  (closed-form: 0.04)
```

Reward is any scalar callable that matches your problem: utility maximisation, cost minimisation, profit, option payoff, regret. bellgrid maximises — negate costs.

## Why you'd reach for bellgrid

- **Correctness across the full support.** The grid solver's backward induction gives the same answer at the tail of the distribution as in the middle — exact everywhere, not just where a sampler happened to explore. (When the dimension outgrows the grid, the neural solver takes over — and the grid certifies it on the overlap, so you're never just trusting an approximation.)
- **Cheap counterfactuals.** Change a parameter, re-solve in seconds. No retraining. Perfect for sensitivity analysis, calibration, and "what if the equity premium were 5%" sweeps.
- **Constraints are first-class.** Borrowing constraints, irreversible state transitions, state-dependent action bounds, mortality-driven discount factors, and warm-glow bequest rewards all fit into the `Problem` interface without RL-style penalty shaping.
- **The solver and simulator share the same `transition` and `reward`.** You can't have a "the simulator was wrong" bug because the simulator literally calls the same callables the solver did.

## Breaking the curse of dimensionality: the neural solver

The grid is exact, but its cost is `∏ (points per dimension)` — hopeless past ~6 continuous state/action dimensions. For those, swap `BackwardInduction` for **`ActorCritic`**, a neural solver behind the **same `Problem`**: it represents `V` and `π` as networks over *sampled* states instead of a mesh, so cost scales with network size, not grid volume.

```python
from bellgrid.rl import ActorCritic
policy, value = solve(problem, solver=ActorCritic(), device="cuda")   # no state/action grid needed
```

It's **model-based**, not model-free RL: it uses your known, differentiable `transition`/`reward` and the *same exact shock quadrature* as the grid solver, so the actor improves against genuine Bellman targets (not a black-box environment it has to explore) and the critic is trained **on-distribution** — on the states the policy actually visits. That on-distribution training is what keeps the reported value consistent with forward simulation *at any horizon*, rather than letting bootstrap error compound down a long backward sweep.

**The correctness contract.** Because both solvers share the `Problem` spec, the grid solver *certifies* the neural one wherever both can run. The [`10_hydropower`](https://github.com/tbb300/bellgrid/blob/main/examples/10_hydropower/hydropower.ipynb) example does exactly this — a multi-reservoir cascade under a stochastic, mean-reverting price — **certified against the exact grid at one reservoir**, then run at a scale where no grid can exist (5-D state, 5-year horizon, an equivalent grid of ~10¹⁴ cells) and **self-validated to ~1%** against `simulate()`. Prove it where you can; trust it where you must.

Scope (v1): `ContinuousState` / `DiscreteState`, `ContinuousAction`, any shock, finite horizon. `MarkovChain`, `DiscreteAction`, and infinite horizon use the grid solvers.

## Examples

Ten canonical problems, each side-by-side with an analytical or numerical reference. Open the notebooks in JupyterLab or [view them on GitHub](https://github.com/tbb300/bellgrid/tree/main/examples).

| Notebook | Problem | Validates against |
|---|---|---|
| [`01_merton`](https://github.com/tbb300/bellgrid/blob/main/examples/01_merton/merton.ipynb) | Log-utility Merton consumption-portfolio | Closed form `V = A + B log w`, `c/w = 1 − β` |
| [`02_carroll_deaton`](https://github.com/tbb300/bellgrid/blob/main/examples/02_carroll_deaton/carroll_deaton.ipynb) | CRRA lifecycle savings with a borrowing constraint | Endogenous Grid Method (Carroll 2006) |
| [`03_american_option`](https://github.com/tbb300/bellgrid/blob/main/examples/03_american_option/american_option.ipynb) | American put on GBM | CRR binomial tree (n=2000), agreement within ~1e-4 |
| [`04_lqg`](https://github.com/tbb300/bellgrid/blob/main/examples/04_lqg/lqg.ipynb) | 2-D linear-quadratic-Gaussian control | Discrete-time Riccati recursion |
| [`05_two_asset_merton`](https://github.com/tbb300/bellgrid/blob/main/examples/05_two_asset_merton/two_asset_merton.ipynb) | 2-asset Merton with correlated returns (`MultivariateNormal`) | Numerical FOC for the optimal portfolio share |
| [`06_regime_switching_option`](https://github.com/tbb300/bellgrid/blob/main/examples/06_regime_switching_option/regime_switching_option.ipynb) | American put under regime-switching vol (`MarkovChain`) | Bracketed by constant-vol references at σ_low, σ_high, σ_stationary |
| [`07_retirement_decision`](https://github.com/tbb300/bellgrid/blob/main/examples/07_retirement_decision/retirement_decision.ipynb) | Lifecycle work vs retire decision (`DiscreteState`, irreversible) | Qualitative — boundary falls with age, accumulate → retire → decumulate dynamics |
| [`08_jump_diffusion_option`](https://github.com/tbb300/bellgrid/blob/main/examples/08_jump_diffusion_option/jump_diffusion_option.ipynb) | American put under Merton (1976) jump-diffusion (`Jump` + `Normal`, multi-shock) | Merton 1976 European series expansion to ~1e-3 |
| [`09_lifecycle_planning`](https://github.com/tbb300/bellgrid/blob/main/examples/09_lifecycle_planning/lifecycle_planning.ipynb) | Full lifecycle: consumption + retirement + asset allocation under mortality, regime-switching markets, warm-glow bequest | The motivating problem. Exercises every primitive at once. |
| [`10_hydropower`](https://github.com/tbb300/bellgrid/blob/main/examples/10_hydropower/hydropower.ipynb) | Multi-reservoir hydropower under a stochastic OU price — **the neural-solver showcase** (`ActorCritic`) | Exact grid at N=1; forward-simulation consistency (~1%) at 5-D / 5-year, where no grid exists |

## What's built

- **States**: `ContinuousState`, `DiscreteState`, `MarkovChain` (any number per problem; cost is additive in chains).
- **Actions**: `ContinuousAction` (with optional state-dependent bounds), `DiscreteAction`.
- **Shocks**: `Normal`, `Lognormal`, `MultivariateNormal` (Cholesky-rotated Gauss-Hermite), `Uniform` (Gauss-Legendre), `Categorical` (exact), `Jump` (Bernoulli-approximated Poisson with Normal log-magnitudes). Multiple independent shocks per problem combine via tensor-product quadrature.
- **Solvers**: `BackwardInduction` (finite-horizon) and `PolicyIteration` (infinite-horizon stationary) — exact GPU grid sweeps with JIT-compiled multilinear interpolation and memory-chunked Bellman updates; **`ActorCritic`** — a model-based neural solver for high-dimensional finite-horizon problems (samples states, learns `V`/`π` as networks, trained on-distribution, certifiable against the grid). CPU or CUDA.
- **Diagnostics**: post-solve check that the optimal policy's next-state distribution stays inside the declared state range; warns if it doesn't (you set your grid too tight).
- **Discount**: scalar, or a callable `(state, t) → tensor` for mortality / hazard-style problems.
- **Reward signature**: 4-arg `(state, action, shock, t)` or 5-arg `(state, action, shock, t, next_state)` — for per-period bequests, terminal-style payoffs computed per period, etc.
- **Simulator**: `simulate()` shares the user's `transition` and `reward` with the solver, so they can't drift apart. Supports the same callable discount and 5-arg reward as the solver.

## Choosing a solver

Both solvers consume the same `Problem` and return the same `(policy, value)` callables — pick by dimension:

| | `BackwardInduction` / `PolicyIteration` (grid) | `ActorCritic` (neural) |
|---|---|---|
| State-dim sweet spot | 1–6 continuous (+ discrete) | high-dimensional, where a grid is infeasible |
| Solution | Exact across the full grid | Approximate; trained on-distribution |
| Tail / edge-case behavior | Correct by construction | Accurate where the policy visits — certify against the grid |
| Speed at low dim | Milliseconds–seconds, and exact | Slower and approximate — prefer the grid here |
| Validation | Analytical / numerical references | Grid solver (low dim) **+** forward-simulation consistency |

Both are **model-based**: they need your `transition` and `reward`. That's also how they differ from **model-free RL** — which is for when you *don't* have a model at all. If you do have one, the neural solver exploits it (differentiable Bellman targets, exact shock quadrature) and is far more sample-efficient than exploring a black box — and, uniquely, it's *certifiable* against the exact grid solver on shared ground.

Full API surface in [`docs/api.md`](https://github.com/tbb300/bellgrid/blob/main/docs/api.md).

## License

MIT.
