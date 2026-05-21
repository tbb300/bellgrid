# bellgrid API

The current public API. Items not yet implemented are marked **(planned)**.

## Design principles

1. **A `Problem` is a value.** No subclassing, no framework lock-in. You build a `Problem`, you pass it to `solve()`, you get back a policy.
2. **Vectorization is mandatory, not optional.** Every transition and reward call is evaluated in batch on GPU by default.
3. **Domain-agnostic core.** The library knows about states, actions, transitions, rewards, shocks, discounts. It does not know about utility theory, options theory, or inventory theory. Those live in user code and example notebooks.
4. **Common constraints are first-class.** Linear action bounds (including state-linear), `MarkovChain` transitions, and terminal/absorbing states are expressed declaratively. Arbitrary feasibility predicates fall back to `-inf` reward.

## Core objects

### State variables

```python
from bellgrid import ContinuousState, DiscreteState, MarkovChain

wealth   = ContinuousState("wealth", warp="asinh", range=(0, 1e7))
basis    = ContinuousState("basis_fraction", range=(0, 1))
regime   = MarkovChain("regime", matrix=P, labels=["bull", "neutral", "bear"])
phase    = DiscreteState("phase", n=2, labels=["accumulation", "decumulation"])
```

`warp` accepts `None` (default), `"asinh"`, `"log"`, or a callable. Warped grids concentrate points where the value function has curvature. The state declaration owns the warp; a `WarpedGrid(n=128)` entry in `state_grid` inherits it automatically. Pass `WarpedGrid(n=128, warp=...)` only to override (e.g., for warp-sweep experiments).

`MarkovChain` is a discrete state with built-in transition dynamics — declare the matrix once and the solver handles the expectation over next states. The convention is row-stochastic: `matrix[i, j] = Pr(next = j | current = i)`, so rows sum to 1. The number of categories is inferred from `matrix.shape[0]`; if `labels` is provided it must have matching length. For state-dependent or otherwise non-stationary discrete dynamics, use `DiscreteState` and write the transition yourself in `transition`. The library doesn't ship a built-in categorical innovation shock; the canonical workaround for stochastic plain-`DiscreteState` transitions is to threshold a `Normal` innovation inside `transition`.

#### Mixed discrete + continuous states

The value function is stored as a tensor with one axis per state dimension. Continuous axes use `state_grid` points; discrete axes (`DiscreteState`, `MarkovChain`) use one point per category. Interpolation runs over the continuous axes only — at each discrete slice the value function is a smooth function over continuous coordinates, and queries inside `policy(state)` / `value(state)` interpolate within the slice corresponding to the discrete state's current value.

Discrete states arrive in user code as `torch.long` tensors under their declared name: `state["regime"]` is a tensor of category indices, not a one-hot. `labels` are display-only (printing, diagnostics) — user code always sees indices.

### Actions

```python
from bellgrid import ContinuousAction, DiscreteAction

draw         = ContinuousAction("draw", bounds=(0, "wealth"))
equity_share = ContinuousAction("equity_share", bounds=(0, 1))
switch_phase = DiscreteAction("switch_phase", n=2)
```

Bounds can reference state variables. In `policy(state, t)` outputs, continuous actions arrive as float tensors and discrete actions as `torch.long` index tensors — mirroring the state dtype convention.

### Shocks

Shocks in bellgrid are *iid innovations*. Persistent processes (AR(1), GARCH, ...) belong in state — see [Persistent shocks](#persistent-shocks) below. Markov-chain dynamics belong in state too — see `MarkovChain` above.

```python
from bellgrid.shocks import Normal, MultivariateNormal, Lognormal
# from bellgrid.shocks import Jump   # planned

equity_shock = Normal("equity", sigma=0.18)         # sigma defaults to 1.0 (standard normal)
yield_shock  = Lognormal("yield", mu=0.03, sigma=0.02)
correlated   = MultivariateNormal(names=["equity", "bonds"],
                                  mean=[0.07, 0.03], cov=...)
# jumps      = Jump("rare_event", intensity=0.02,            # planned
#                   size_dist=Lognormal(mu=-0.1, sigma=0.2))
```

Quadrature: Gauss-Hermite for `Normal`/`Lognormal`, Cholesky-rotated tensor-product Gauss-Hermite for `MultivariateNormal`. `Jump` will use mixed quadrature when it lands. The solver supports at most one shock object per problem (a `MultivariateNormal` counts as one even though it carries K named dimensions).

Shock names are optional — nameless shocks work as `size_dist` for `Jump` or any inner-distribution slot where they're not surfaced through `shock[...]`. Defaults: `Normal()` is standard normal; `Lognormal()` is standard lognormal, with `mu` and `sigma` as parameters of the underlying normal (`log(X) ~ N(mu, sigma)`). `MultivariateNormal` has no no-args default — pass `names=[...]` (or `dim=N`) to fix the dimensionality, and `names[i]` indexes both `mean[i]` and row/column `i` of `cov`.

#### Time-varying shock parameters

A shock declaration carries fixed parameters. For age- or state-varying shock magnitudes (e.g., labor-income variance that grows with age), declare the shock as a standardized innovation and scale in `transition`:

```python
labor_innovation = Normal("z_labor")            # sigma=1 by default

def transition(state, action, shock, t):
    sigma_t = labor_sigma_schedule(t)
    new_income = base + sigma_t * shock["z_labor"]
    ...
```

Gauss-Hermite nodes for `Normal("z")` are the standard-normal nodes — scaling them by a constant inside `transition` is mathematically equivalent to varying the shock's variance at declaration time.

#### Persistent shocks

AR(1), AR(p), GARCH, and similar processes carry memory, so their level has to live in the Markov state regardless. Rather than hide that inside a shock object, bellgrid keeps it explicit: declare a `ContinuousState` for the persistent process and write the recurrence into your `transition`.

```python
equity_return_lag = ContinuousState("equity_return_lag", range=(-0.5, 0.5))
equity_innovation = Normal("equity_innovation", sigma=0.18)

def transition(state, action, shock, t):
    new_return = 0.6 * state["equity_return_lag"] + shock["equity_innovation"]
    ...
    return {"equity_return_lag": new_return, ...}
```

This makes the `N^d` cost visible: each persistent process adds one continuous state dimension. A `bellgrid.processes.AR1` builder that emits the `(state, shock, transition fragment)` triple may land later as sugar over this pattern.

### Transition

A user-supplied callable mapping `(state, action, shock, t) -> next_state`:

```python
def transition(state, action, shock, t):
    new_wealth = (state["wealth"] - action["draw"]) * (
        action["equity_share"] * (1 + shock["equity"])
        + (1 - action["equity_share"]) * (1 + shock["yield"])
    )
    return {
        "wealth": new_wealth,
        "basis_fraction": ...,
    }
```

Operates on batched tensors. bellgrid handles broadcasting across the state grid and shock quadrature.

`transition` returns a single sampled next-state per shock quadrature node, not a distribution. The shock objects own the quadrature weights; the solver applies them externally when computing the Bellman expectation.

The return dict only needs entries for states whose dynamics live in user code (continuous states and plain `DiscreteState`s). `MarkovChain` states are advanced internally by the solver from their declared transition matrix.

### Reward

A user-supplied callable mapping `(state, action, shock, t) -> scalar`. bellgrid maximizes the expected discounted sum.

```python
def reward(state, action, shock, t):
    return some_per_step_payoff(state, action, shock, t)
```

`shock` and `t` are always passed; rewards that ignore either can use `_shock` / `_t`. This is intentionally a plain callable, not a class hierarchy. CRRA utility, quadratic cost, option payoff, inventory holding-cost — all the same primitive. If you are minimizing cost, negate.

For finite-horizon problems, an optional terminal reward `(state) -> scalar` evaluated at the end of the horizon (e.g., bequest motive, residual value):

```python
def terminal_reward(state):
    return ...
```

For option-style problems where exercise is a per-period action choice, the exercise payoff goes in `reward` keyed on the exercise action — `terminal_reward` covers only the residual value if the horizon ends without that choice ever firing.

### Discount

```python
discount = 0.96
```

A scalar is the common case. For problems with state- or age-dependent termination (mortality being one example), `discount` may also be a callable `(state, t) -> scalar`:

```python
def discount(state, t):
    return 0.96 * survival_probability(state, t)
```

This is the general mechanism for stochastic termination. Mortality tables, equipment-failure hazards, and bankruptcy probabilities all fit through this slot — none are built-in.

### Problem

```python
from bellgrid import Problem

problem = Problem(
    states=[wealth, basis, regime, phase],
    actions=[draw, equity_share, switch_phase],
    transition=transition,
    reward=reward,
    terminal_reward=terminal_reward,    # optional
    shocks=[equity_shock, yield_shock],
    horizon=range(25, 120),             # finite; `t` ranges over these values (25..119)
    discount=discount,
)
```

`t` is passed to `transition`, `reward`, and `discount` and takes the values in `horizon` — `range(25, 120)` yields `t ∈ {25, 26, …, 119}`, swept in reverse by backward induction. For `horizon=None` (infinite-horizon), `t=None`.

Deterministic problems pass `shocks=[]`; the `shock` argument is still passed to `transition` and `reward` as an empty dict.

`Problem` validates at construction time: state, action, and shock names must not collide; `MarkovChain.matrix` must be square and match its number of categories; action `bounds` that reference state variables must reference declared state names; transition return dicts must cover every non-`MarkovChain` state. Errors raise eagerly so they're easy to fix.

## Solving

```python
from bellgrid import solve
from bellgrid.grids import RegularGrid, WarpedGrid
from bellgrid.solvers import BackwardInduction, PolicyIteration

policy, value = solve(
    problem,
    state_grid={"wealth": WarpedGrid(n=128, warp="asinh"),
                "basis_fraction": RegularGrid(n=16)},
    action_grid={"draw": RegularGrid(n=64),
                 "equity_share": RegularGrid(n=33)},
    solver=BackwardInduction(),         # PolicyIteration(tol=1e-7) for horizon=None is planned
    device="cuda",
    dtype="float64",                    # float32 is faster but risky for wide-range value funcs or -inf rewards (NaN risk)
    chunk_size=2**20,                   # batch size for Bellman expectation; lower if OOM
)
```

`state_grid` is required for continuous states; discrete states (`DiscreteState`, `MarkovChain`) need no entry. `action_grid` is required for continuous actions; discrete actions enumerate their `n` values. Action bounds that reference a state name (`bounds=(0, "wealth")`) are interpreted on a normalized `[0, 1]` grid and rescaled to `[lower, upper]` at each grid point — the simple linear case is built in. More complex constraints (e.g., `consumption ≤ wealth + borrowing_limit(credit_state)`) are user-side: return `-inf` from `reward` for infeasible `(state, action)` combinations.

`chunk_size` controls the batch size for evaluating the Bellman expectation — the joint grid of states × shock quadrature nodes is processed in chunks of this size. Lower it if you OOM; raise it for throughput on large devices.

Defaults and required arguments: `state_grid` is required when there are continuous states; `action_grid` when there are continuous actions; `solver` has no default (pass `BackwardInduction()` for finite horizon; `PolicyIteration(tol=...)` for `horizon=None` is planned). The remaining arguments default to `device="cuda" if available else "cpu"`, `dtype="float64"`, `chunk_size=2**20` (the latter is currently accepted but unused — it'll matter once memory-chunked Bellman updates land).

`policy(state, t)` and `value(state, t)` are time-indexed for finite-horizon problems — the solver stores a separate V and π slice at each `t` in `horizon`. Pass `t=None` for infinite-horizon problems, where V and π are stationary. Both accept batched state dicts (equal-shaped tensors) and return batched actions and values; scalar dicts work too for one-off queries.

## Simulating

```python
from bellgrid import simulate

paths = simulate(
    policy=policy,
    problem=problem,
    n=10_000,
    initial_state={"wealth": 500_000, "regime": 2},  # MarkovChain default: stationary
    seed=0,
)
# paths["wealth"], paths["regime"], paths["draw"], ...    # state + action realizations
# paths["reward"]                                          # per-step realized reward
# paths["discounted_total"]                                # sum of discounted rewards per path
```

`paths` is a dict of tensors. State and action keys carry realized values shaped `(n, len(horizon))`; axis 1 indexes the values in `horizon` (so for `horizon=range(25, 120)` the columns are `t = 25, 26, …, 119`). `paths["reward"]` is the per-step realized reward at each `t`; `paths["discounted_total"]` is the per-path scalar sum of discounted rewards.

`initial_state` must specify every state variable, including `MarkovChain` states (e.g., `"regime": 2`). (Defaulting `MarkovChain` initial state to the stationary distribution is planned but not currently implemented — pass an explicit category index for now.)

The simulator uses the *same* transition and reward functions as the solver. There is no opportunity for the simulator and solver to drift apart.

