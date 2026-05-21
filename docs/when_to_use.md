# When to use bellgrid (vs. RL, vs. analytical methods)

This page exists to keep us honest about positioning. bellgrid is not a general-purpose RL framework, and it is not a replacement for closed-form solutions where those exist. It occupies a specific niche.

## The short version

Use **bellgrid** when:

1. You have, or can write, a transition model — a `(state, action, shock, t) -> next_state` function plus distributions for the shocks.
2. Your continuous state is roughly 1–6 dimensions (plus any number of discrete states).
3. You need a policy that is correct *across the entire support of the state space*, not just along the trajectories an agent happens to visit.
4. Your problem has kinks, constraints, or discrete events (action bounds, state-dependent feasibility, regime switches, absorbing states, brackets).
5. You want to differentiate the solution with respect to model parameters, run sensitivity analyses, or recompute under counterfactual specifications cheaply.

Use **RL** (PPO, SAC, DQN, etc.) when:

1. State is high-dimensional (images, robot proprioception, large feature vectors).
2. You don't have a transition model, only the ability to sample.
3. On-distribution performance is what matters; tail behavior is a secondary concern.
4. Representational capacity matters more than exactness.

Use **closed-form** methods when one exists. (bellgrid's Merton example exists precisely to validate against the closed-form solution.)

## The longer version

Backward induction and RL are not competitors — they are tools for different ends of two axes: model availability and dimensionality.

### Axis 1: Do you have a model?

bellgrid requires a transition function. If you can write `next_state = transition(state, action, shock, t)` and `shock ~ distribution`, bellgrid can solve the problem. If you cannot — if your only access to the dynamics is through a simulator or the real world — you need RL.

In finance, operations research, energy, and many engineering domains, models are usually available (you wrote them; they are the point). In robotics and games, models are often unavailable or too inaccurate to be useful.

### Axis 2: How high-dimensional is the state?

bellgrid stores a value function on a grid. The cost scales roughly as `N^d` where `d` is the continuous-state dimensionality. With multilinear interpolation on regular grids:

- `d = 1`: trivial, milliseconds.
- `d = 3`: comfortable, seconds.
- `d = 5`: feasible on GPU, minutes.
- `d = 6–7`: the boundary. Warped grids, factorization tricks, or function-approximation methods become necessary.
- `d ≥ 8`: bellgrid is the wrong tool. Reach for RL or a function-approximation method.

RL trades the curse of dimensionality for the curse of distribution: a neural-network policy can represent a value function over a 1000-dimensional state, but it is only accurate where it has been trained.

### Axis 3: Where do you need to be correct?

This is the axis that gets least attention and matters most for the audiences bellgrid targets.

A bellgrid solution is exact across the entire grid, up to interpolation error. The policy at a state nobody ever visits in simulation is still correct. This is essential when:

- You are writing software that has to behave well in tail scenarios (financial planning under catastrophic markets, energy storage during grid events, options pricing in volatile regimes, inventory under demand shocks).
- You need to *certify* a policy — regulators, auditors, model-risk committees, or your own conscience demand that the answer be defensible at every state, not just on the modal trajectory.
- You want to study the structure of the optimal policy — its kinks, its boundaries — not just its expected reward.

RL gives you a policy that is good on the training distribution. If your training distribution covers the tails, the policy is good in the tails. If it does not, the policy is undefined behavior in the tails. There is no easy way to know which without testing exhaustively.

### Axis 4: Constraints and kinks

Real problems are full of them. Action sets depend on state. Continuation is forbidden in some regions. Reward functions have brackets, floors, or singularities. Discrete events kick in at thresholds.

bellgrid expresses these as first-class objects: bounded actions, state-dependent feasibility, finite-state Markov chains, terminal/absorbing states. The backward-induction algorithm respects them by construction — there is no "soft constraint penalty" hyperparameter to tune.

RL handles constraints poorly. Soft penalties are sensitive and frequently violated; constrained-RL is an active research area but not a solved problem.

### Axis 5: Recomputability and sensitivity

Once you have a bellgrid solution, perturbing a parameter (discount rate, shock volatility, reward coefficient) and re-solving is cheap relative to retraining an RL agent. For workflows that involve calibration, sensitivity analysis, or scenario analysis, this matters a lot.

## Borderline cases

Some problems live on the boundary. Practical guidance:

- **Inventory control with 2–3 items.** bellgrid wins. With 50+ SKUs, RL wins.
- **Option pricing.** bellgrid wins for American options on 1–4 underlyings; for high-dim baskets, Monte Carlo or RL.
- **Portfolio choice.** bellgrid wins for academic problems (1–2 assets, 1 wealth state). For institutional portfolios (50 assets, factor exposures), RL or convex optimization.
- **Robotics.** RL wins. Don't try bellgrid.
- **Energy storage with weather forecasts.** bellgrid wins if the forecast is compressible to a few state variables; RL otherwise.
- **Lifecycle / retirement planning.** bellgrid wins — 1–6 continuous states is the canonical range, and tail correctness matters because the stakes of being wrong on the policy are real.

## What bellgrid is *not* trying to be

- A general-purpose RL framework. There are excellent ones; we link to them.
- A symbolic DP solver. Dolo/Dolang serve that audience; we are numerical.
- A grid-math library. Tasmanian is the gold standard for grid math; we benchmark against a hand-rolled Tasmanian DP loop but don't depend on it.
- A domain-specific library for any one application. Utility theory, options theory, inventory theory, control theory — bellgrid is the common substrate. Domain primitives live in the example notebooks where they belong.
- An advice / planning product. Solvers compute optimal policies given a reward and constraints. Applying that to a real person's life is a separate, regulated activity.
