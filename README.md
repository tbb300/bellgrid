# bellgrid

**Solve stochastic dynamic programs on the GPU.** Mixed continuous and discrete state, mixed continuous and discrete action. One `Problem` spec, a **portfolio of solvers matched to its structure**: **exact** GPU grid sweeps at low dimension, and — where a grid can't reach — **analytic trajectory optimization** (`iLQG`), **pathwise policy gradients** (`PolicyGradient`), and a **model-based neural actor–critic** (`ActorCritic`). Because every solver shares the spec, each certifies the others — or an exact analytical oracle — wherever they overlap.

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

- **Correctness across the full support.** The grid solver's backward induction gives the same answer at the tail of the distribution as in the middle — exact everywhere, not just where a sampler happened to explore. (When the dimension outgrows the grid, a structure-matched solver takes over — `iLQG`, `PolicyGradient`, or `ActorCritic` — and the grid or an exact oracle certifies it on the overlap, so you're never just trusting an approximation.)
- **Cheap counterfactuals.** Change a parameter, re-solve in seconds. No retraining. Perfect for sensitivity analysis, calibration, and "what if the equity premium were 5%" sweeps.
- **Constraints are first-class.** Borrowing constraints, irreversible state transitions, state-dependent action bounds, mortality-driven discount factors, and warm-glow bequest rewards all fit into the `Problem` interface without RL-style penalty shaping.
- **The solver and simulator share the same `transition` and `reward`.** You can't have a "the simulator was wrong" bug because the simulator literally calls the same callables the solver did.

## Beyond the grid: matching the solver to the structure

The grid is exact, but its cost is `∏ (points per dimension)` — hopeless past ~6 continuous state/action dimensions. Past that you don't reach for one high-D solver; you reach for the one that fits your problem's **structure**, all behind the same `Problem`:

- **`iLQG`** — *smooth continuous control.* Builds the local quadratic model of the value from autograd derivatives of your `transition`/`reward` and solves it Newton-style. On a linear-quadratic problem that model is exact, so one step *is* the matrix-Riccati solution — to **machine precision, in seconds, at any dimension**. Returns a time-varying affine feedback law (globally optimal for LQ; local around the optimized trajectory otherwise).
- **`PolicyGradient`** — *high-D continuous, the clean default.* Trains a policy by backpropagating the return straight through your differentiable model (the pathwise / "stochastic value gradient" estimator — *not* score-function/REINFORCE). **No learned critic, no bootstrap**, so none of the overestimation pathologies of value-based RL: the model itself supplies the policy gradient.
- **`ActorCritic`** — *high-D with discrete states or non-differentiable dynamics.* The one option that tolerates a regime you can't differentiate through: it learns `V` as networks and **bootstraps** (sampling next states rather than differentiating them), with a truncated critic ensemble to control the resulting overestimation, trained on-distribution.

```python
from bellgrid.solvers import iLQG
from bellgrid.rl import PolicyGradient, ActorCritic

policy, value = solve(problem, solver=PolicyGradient(), device="cuda")   # no state/action grid needed
```

All four are **model-based** — they use your known, differentiable `transition`/`reward` and the *same exact shock quadrature* as the grid, not a black-box environment they must explore. The split among them is one tradeoff seen from two sides: backpropagating *through* the model (`iLQG`, `PolicyGradient`) is exact and overestimation-free but **needs** a differentiable model; bootstrapping a *learned* value (`ActorCritic`) tolerates discrete / non-smooth dynamics but pays for it in approximation bias. Pick the one whose assumption your problem actually satisfies.

**The correctness contract.** Because every solver shares the `Problem` spec, they certify each other — and where the problem is linear-quadratic, a matrix-Riccati closed form is an exact oracle at *any* dimension. [`11_liquidation`](https://github.com/tbb300/bellgrid/blob/main/examples/11_liquidation/liquidation.ipynb) certifies `iLQG` against it to **machine precision** and the neural solvers to ~1–2% at **80 state dimensions** (~10¹⁷⁶ equivalent grid cells). Where no oracle exists ([`10_hydropower`](https://github.com/tbb300/bellgrid/blob/main/examples/10_hydropower/hydropower.ipynb)), the grid certifies the high-D solvers on the low-D overlap and `simulate()` checks forward-consistency at scale. Prove it where you can; trust it where you must.

Scope (v1): `iLQG` and `PolicyGradient` need an all-`ContinuousState`, `ContinuousAction`, scalar-discount, finite-horizon problem (the model must be differentiable). `ActorCritic` additionally handles `DiscreteState`. `MarkovChain`, `DiscreteAction`, callable discount, and infinite horizon use the grid solvers.

## Examples

Eleven canonical problems, each side-by-side with an analytical or numerical reference. Open the notebooks in JupyterLab or [view them on GitHub](https://github.com/tbb300/bellgrid/tree/main/examples).

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
| [`11_liquidation`](https://github.com/tbb300/bellgrid/blob/main/examples/11_liquidation/liquidation.ipynb) | Optimal execution of N correlated assets (Almgren–Chriss) — **the right tool vs. the general tool, against an exact high-D oracle** | Matrix-Riccati closed form at any dimension. At **N=40 (80-D, ~1e176 grid cells)** `iLQG` matches it to **machine precision in seconds**; the neural solvers approximate it to ~1–2%, certified against the same oracle. The structure-routing lesson, made concrete. |

## What's built

- **States**: `ContinuousState`, `DiscreteState`, `MarkovChain` (any number per problem; cost is additive in chains).
- **Actions**: `ContinuousAction` (with optional state-dependent bounds), `DiscreteAction`.
- **Shocks**: `Normal`, `Lognormal`, `MultivariateNormal` (Cholesky-rotated Gauss-Hermite), `Uniform` (Gauss-Legendre), `Categorical` (exact), `Jump` (Bernoulli-approximated Poisson with Normal log-magnitudes). Multiple independent shocks per problem combine via tensor-product quadrature.
- **Solvers**: `BackwardInduction` (finite-horizon) and `PolicyIteration` (infinite-horizon stationary) — exact GPU grid sweeps with JIT-compiled multilinear interpolation and memory-chunked Bellman updates; **`iLQG`** — analytic trajectory optimization (DDP) that backprops autograd derivatives of the model into a Newton step, exact on LQ; **`PolicyGradient`** — a pathwise/analytic policy gradient that backprops the return through the differentiable model (no critic, no bootstrap); **`ActorCritic`** — a model-based neural actor–critic with a truncated critic ensemble, for high-D problems with discrete states or non-differentiable dynamics. All behind one `Problem`, CPU or CUDA, cross-certifiable.
- **Diagnostics**: post-solve check that the optimal policy's next-state distribution stays inside the declared state range; warns if it doesn't (you set your grid too tight).
- **Discount**: scalar, or a callable `(state, t) → tensor` for mortality / hazard-style problems.
- **Reward signature**: 4-arg `(state, action, shock, t)` or 5-arg `(state, action, shock, t, next_state)` — for per-period bequests, terminal-style payoffs computed per period, etc.
- **Simulator**: `simulate()` shares the user's `transition` and `reward` with the solver, so they can't drift apart. Supports the same callable discount and 5-arg reward as the solver.

## Choosing a solver

Every solver consumes the same `Problem` and returns the same `(policy, value)` callables — route by **structure**, not just dimension:

| solver | reach for it when… | states | actions | the answer it gives |
|---|---|---|---|---|
| **grid** — `BackwardInduction` / `PolicyIteration` | ≤ ~6 dimensions, **or any discrete action**, or infinite horizon | continuous + discrete + Markov | continuous + **discrete** | **exact** across the full support |
| **`iLQG`** | smooth continuous control; you want the exact/near-exact answer, fast | continuous | continuous | exact on LQ; local optimum otherwise |
| **`PolicyGradient`** | high-D continuous with a differentiable model — **the default past the grid** | continuous | continuous | near-optimal, no overestimation |
| **`ActorCritic`** | high-D **with discrete states** or non-smooth dynamics you can't differentiate | continuous + **discrete** | continuous | approximate (bootstrapped, certified) |

A quick decision: **discrete actions or low dimension → grid.** Otherwise, high-D and continuous — **smooth/LQ → `iLQG`**, **a differentiable model → `PolicyGradient`** (the clean default), and **discrete states or non-differentiable dynamics → `ActorCritic`** (the only one that can bootstrap past a step it can't differentiate).

All four are **model-based** — they need your `transition` and `reward`, which is how they differ from **model-free RL** (for when you have no model at all). Given a model, three of the four *exploit its gradient* directly; the actor–critic is the fallback for exactly the cases where that gradient doesn't exist.

→ The **math, scope, and literature** for each solver — the Bellman backup, the iLQG backward recursion, the pathwise gradient, the critic ensemble, and references — are in [`docs/solvers.md`](https://github.com/tbb300/bellgrid/blob/main/docs/solvers.md).

Full API surface in [`docs/api.md`](https://github.com/tbb300/bellgrid/blob/main/docs/api.md); the solvers' math and literature in [`docs/solvers.md`](https://github.com/tbb300/bellgrid/blob/main/docs/solvers.md).

## License

MIT.
