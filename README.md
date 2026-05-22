# bellgrid

**Solve stochastic dynamic programs on the GPU.** Mixed continuous and discrete state, mixed continuous and discrete action, exact via backward induction.

bellgrid is for the case where you *have* a model — a transition, a reward, a discount factor — and want the optimal policy across the entire state space. Not approximate. Not on-distribution. Not after a week of RL tuning. Declare a `Problem`, pick a state mesh, and get back a policy you can query at any point in the support.

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

- **Correctness across the full support.** Backward induction over a state mesh gives the same answer at the tail of the distribution as in the middle. RL is approximate and *on-distribution* — its policy on bankruptcy paths, regime-change scenarios, or mortality boundaries is whatever happened to be explored during training.
- **Cheap counterfactuals.** Change a parameter, re-solve in seconds. No retraining. Perfect for sensitivity analysis, calibration, and "what if the equity premium were 5%" sweeps.
- **Constraints are first-class.** Borrowing constraints, irreversible state transitions, state-dependent action bounds, mortality-driven discount factors, and warm-glow bequest rewards all fit into the `Problem` interface without RL-style penalty shaping.
- **The solver and simulator share the same `transition` and `reward`.** You can't have a "the simulator was wrong" bug because the simulator literally calls the same callables the solver did.

## Examples

Nine canonical problems, each side-by-side with an analytical or numerical reference. Open the notebooks in JupyterLab or [view them on GitHub](https://github.com/tbb300/bellgrid/tree/main/examples).

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

## What's built

- **States**: `ContinuousState`, `DiscreteState`, `MarkovChain` (any number per problem; cost is additive in chains).
- **Actions**: `ContinuousAction` (with optional state-dependent bounds), `DiscreteAction`.
- **Shocks**: `Normal`, `Lognormal`, `MultivariateNormal` (Cholesky-rotated Gauss-Hermite), `Uniform` (Gauss-Legendre), `Categorical` (exact), `Jump` (Bernoulli-approximated Poisson with Normal log-magnitudes). Multiple independent shocks per problem combine via tensor-product quadrature.
- **Solvers**: `BackwardInduction` for finite-horizon problems, `PolicyIteration` for infinite-horizon stationary problems. CPU or CUDA, JIT-compiled multilinear interpolation, memory-chunked Bellman update for big state × action × shock tensors.
- **Diagnostics**: post-solve check that the optimal policy's next-state distribution stays inside the declared state range; warns if it doesn't (you set your grid too tight).
- **Discount**: scalar, or a callable `(state, t) → tensor` for mortality / hazard-style problems.
- **Reward signature**: 4-arg `(state, action, shock, t)` or 5-arg `(state, action, shock, t, next_state)` — for per-period bequests, terminal-style payoffs computed per period, etc.
- **Simulator**: `simulate()` shares the user's `transition` and `reward` with the solver, so they can't drift apart. Supports the same callable discount and 5-arg reward as the solver.

## When to use bellgrid (vs. RL)

| | bellgrid | RL |
|---|---|---|
| State dim sweet spot | 1–6 continuous + discrete | thousands |
| Correctness | Exact across the full grid | Approximate, on-distribution |
| Tail / edge-case behavior | By construction | Only if explored in training |
| Constraints / kinks | First-class | Hard to encode |
| Off-policy what-ifs | Cheap recompute | Full retrain |
| You don't have a model | Doesn't apply | Where RL wins |

Full API surface in [`docs/api.md`](https://github.com/tbb300/bellgrid/blob/main/docs/api.md).

## License

MIT.
