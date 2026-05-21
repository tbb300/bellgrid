# bellgrid

**GPU-native backward-induction for continuous-state stochastic dynamic programs.**

`bellgrid` solves Bellman equations *exactly* (up to interpolation error) across the entire state space. It is opinionated about backward induction, vectorization, and constraints; it is unopinionated about your application domain. Composes K continuous states with arbitrary continuous and discrete actions; supports asinh/log-warped grids, Gauss-Hermite shock quadrature, and a JIT-compiled multilinear kernel that runs on CPU or CUDA.

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
    states=[ContinuousState("wealth", warp="asinh", range=(1e-3, 50.0))],
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
w = torch.tensor([2.0, 10.0, 25.0])
policy({"wealth": w}, t=10)["consume"] / w
# → tensor([0.0401, 0.0401, 0.0421])  (closed-form: 0.04 at every point)
```

Reward is whatever scalar callable matches your problem: utility maximization, cost minimization, profit, payoff. Sign convention: bellgrid maximizes — negate costs.

## Examples

Four canonical problems, each validated against an analytical or numerical reference. Open the `.ipynb` files in JupyterLab or [view them on GitHub](examples/).

| Notebook | Problem | Reference |
|---|---|---|
| [`01_merton`](examples/01_merton/merton.ipynb) | Log-utility Merton consumption-portfolio | Closed-form `V = A + B log w`, `c/w = 1 − β` |
| [`02_carroll_deaton`](examples/02_carroll_deaton/carroll_deaton.ipynb) | CRRA lifecycle savings, borrowing constraint | Qualitative (kinked consumption, MPC → 1 at the constraint, buffer-stock target) |
| [`03_american_option`](examples/03_american_option/american_option.ipynb) | American put on GBM | CRR binomial tree (n=2000), agreement within ~1e-4 absolute |
| [`04_lqg`](examples/04_lqg/lqg.ipynb) | 2-D linear-quadratic-Gaussian control | Discrete-time Riccati recursion |

Each notebook leads with the problem statement and equations before the code, and plots bellgrid against the reference side-by-side.

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

See [`docs/when_to_use.md`](docs/when_to_use.md) for the full positioning piece.

## License

MIT.
