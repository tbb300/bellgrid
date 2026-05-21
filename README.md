# bellgrid

**GPU-native backward-induction for continuous-state stochastic dynamic programs.**

`bellgrid` solves Bellman equations *exactly* (up to interpolation error) across the entire state space. It is opinionated about backward induction, vectorization, and constraints; it is unopinionated about your application domain.

```python
from bellgrid import Problem, ContinuousState, MarkovChain, ContinuousAction, solve
from bellgrid.shocks import Normal
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.solvers import BackwardInduction

def transition(state, action, shock, t):
    ...

def reward(state, action, shock, t):
    ...

problem = Problem(
    states=[ContinuousState("wealth", warp="asinh", range=(0, 1e7)),
            MarkovChain("regime", matrix=P)],
    actions=[ContinuousAction("consumption", bounds=(0, "wealth")),
             ContinuousAction("equity_share", bounds=(0, 1))],
    transition=transition,
    reward=reward,
    shocks=[Normal("equity_return", sigma=0.18)],
    horizon=range(25, 120),
    discount=0.96,
)

policy, value = solve(
    problem,
    state_grid={"wealth": WarpedGrid(n=128, warp="asinh")},
    action_grid={"consumption": RegularGrid(n=64),
                 "equity_share": RegularGrid(n=33)},
    solver=BackwardInduction(),
    device="cuda",
)
```

Reward is whatever scalar callable matches your problem: utility maximization, cost minimization, profit, payoff. Sign convention: bellgrid maximizes — negate costs.

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

See [`docs/when_to_use.md`](docs/when_to_use.md) for the full comparison.

## Status

Pre-alpha. Targeting first PyPI release around the Carroll/Deaton example landing (~month 6 on the roadmap).

## License

MIT.
