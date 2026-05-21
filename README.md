# bellgrid

**GPU-native backward-induction for continuous-state stochastic dynamic programs.**

`bellgrid` solves Bellman equations *exactly* (up to interpolation error) across the entire state space. It's opinionated about backward induction, vectorization, and constraints; unopinionated about your application domain. Composes K continuous states with discrete-state primitives (`DiscreteState`, `MarkovChain`) and any mix of continuous and discrete actions. Supports asinh/log-warped grids, scalar / multivariate Gauss-Hermite shock quadrature, and a JIT-compiled multilinear kernel that runs on CPU or CUDA.

## Quick start

```bash
git clone https://github.com/tbb300/bellgrid && cd bellgrid
uv sync --extra examples
```

A minimal Merton consumption-portfolio (log utility, single risky asset, lognormal returns):

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
# → tensor([0.0401, 0.0401, 0.0401, 0.0401])  (closed-form: 0.04 at every point)
```

Reward is whatever scalar callable matches your problem: utility maximization, cost minimization, profit, payoff. Sign convention: bellgrid maximizes — negate costs.

## Examples

Eight canonical problems, each validated against an analytical or numerical reference. Open the `.ipynb` files in JupyterLab or [view them on GitHub](examples/).

| Notebook | Problem | Validates against |
|---|---|---|
| [`01_merton`](examples/01_merton/merton.ipynb) | Log-utility Merton consumption-portfolio | Closed form `V = A + B log w`, `c/w = 1 − β` |
| [`02_carroll_deaton`](examples/02_carroll_deaton/carroll_deaton.ipynb) | CRRA lifecycle savings with a borrowing constraint | Endogenous Grid Method (Carroll 2006) |
| [`03_american_option`](examples/03_american_option/american_option.ipynb) | American put on GBM | CRR binomial tree (n=2000), agreement within ~1e-4 |
| [`04_lqg`](examples/04_lqg/lqg.ipynb) | 2-D linear-quadratic-Gaussian control | Discrete-time Riccati recursion |
| [`05_two_asset_merton`](examples/05_two_asset_merton/two_asset_merton.ipynb) | 2-asset Merton with correlated returns (`MultivariateNormal`) | Numerical FOC for the optimal portfolio share |
| [`06_regime_switching_option`](examples/06_regime_switching_option/regime_switching_option.ipynb) | American put under regime-switching vol (`MarkovChain`) | Bracketed by constant-vol references at σ_low, σ_high, σ_stationary |
| [`07_retirement_decision`](examples/07_retirement_decision/retirement_decision.ipynb) | Lifecycle work vs retire decision (`DiscreteState`, irreversible) | Qualitative — boundary falls with age, accumulate → retire → decumulate dynamics |
| [`08_jump_diffusion_option`](examples/08_jump_diffusion_option/jump_diffusion_option.ipynb) | American put under Merton (1976) jump-diffusion (`Jump` + `Normal`, multi-shock) | European case validated against Merton 1976 series expansion (agreement within ~1e-3); American case shows the jump premium and lower exercise boundary |

Each notebook opens with the problem statement and equations, then runs bellgrid against the reference side-by-side.

## What's built

- **States**: `ContinuousState` (with optional `asinh` / `log` warp), `DiscreteState`, `MarkovChain`.
- **Actions**: `ContinuousAction` (with optional state-dependent bounds), `DiscreteAction`.
- **Shocks**: `Normal`, `Lognormal`, `MultivariateNormal`, `Jump` (Bernoulli-approximated Poisson with Normal log-magnitudes) — all with Gauss-Hermite quadrature. Multiple independent shocks per problem are supported via tensor-product quadrature.
- **Grids**: `RegularGrid`, `WarpedGrid`.
- **Solver**: `BackwardInduction` for finite horizon, CPU or CUDA, JIT-compiled K-D multilinear interpolation.
- **Simulator**: `simulate()` shares the user's `transition` and `reward` with the solver, so they can't drift apart.

## Planned

- `PolicyIteration` solver — infinite-horizon problems (removes the truncate-with-closed-form-terminal hack).
- Implicit differentiation of `policy` / `value` wrt model parameters.
- Local action search instead of grid enumeration (for problems with many continuous actions).

## When to use bellgrid (vs. RL)

You have (or can write) a transition model, state is roughly 1–6 continuous dims plus discrete, and you need a policy that is **correct across the entire support** — including tails the agent rarely visits.

| | bellgrid | RL |
|---|---|---|
| State dim sweet spot | 1–6 continuous + discrete | thousands |
| Correctness | Exact across the full grid | Approximate, on-distribution |
| Tail / edge-case behavior | By construction | Only if explored in training |
| Constraints / kinks | First-class | Hard to encode |
| Off-policy what-ifs | Cheap recompute | Full retrain |
| You don't have a model | Doesn't apply | Where RL wins |

Longer version with concrete borderline cases in [`docs/when_to_use.md`](docs/when_to_use.md); full API surface in [`docs/api.md`](docs/api.md).

## License

MIT.
