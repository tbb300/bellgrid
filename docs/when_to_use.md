# When to use bellgrid

bellgrid is not a general RL framework and not a substitute for closed-form solutions where they exist. It occupies a specific niche between the two.

## Use bellgrid when

1. You have, or can write, a transition function `(state, action, shock, t) -> next_state`.
2. Continuous state is roughly **1–6 dimensions** (plus any number of discrete states).
3. You need a policy that is **correct across the full state space**, not just along the modal trajectory — for tail scenarios, certification, or because you care about the structure of the policy (kinks, boundaries) and not just its expected return.
4. Your problem has constraints, kinks, or discrete events (action bounds, regime switches, absorbing states, brackets).

The canonical examples in this repo cover most of the territory:

- **Merton, Carroll/Deaton, retirement-decision**: lifecycle consumption-savings — wealth state, CRRA / log utility, optional borrowing constraint, optional retirement-phase switch. 1–2 continuous states + optional discrete state.
- **American option, regime-switching option**: exercise-vs-hold decisions under risk-neutral GBM with optional `MarkovChain` regime. 1 continuous state + optional discrete state.
- **LQG**: 2-D linear-quadratic-Gaussian control with closed-form Riccati validation. 2 continuous states.
- **Two-asset Merton**: portfolio choice with correlated returns via `MultivariateNormal`. 1 continuous state, 2 continuous actions.

## Use RL instead when

- State is high-dimensional (images, large feature vectors, hundreds of dimensions).
- You don't have a transition model, only the ability to sample.
- On-distribution performance matters more than tail correctness.

## Use closed-form when one exists

Don't burn cycles on a numerical solver for a problem that has a clean Hamilton-Jacobi-Bellman solution. The Merton example exists in this repo to validate bellgrid against the closed form, not to compete with it.

## Borderline cases

| Problem | Verdict |
|---|---|
| Inventory control, 2–3 items | bellgrid wins |
| Inventory control, 50+ SKUs | RL wins |
| Option pricing on 1–4 underlyings (incl. early exercise) | bellgrid wins |
| High-dim basket options | Monte Carlo / RL |
| Academic portfolio choice (1–2 assets, 1 wealth state) | bellgrid wins |
| Institutional portfolios (50 assets, factor exposures) | Convex optimization or RL |
| Lifecycle / retirement planning | bellgrid wins (1–6 continuous states; tail correctness matters) |
| Energy storage with weather forecasts | bellgrid if the forecast compresses to a few state variables; RL otherwise |
| Robotics | RL — don't try bellgrid |

## What bellgrid is not trying to be

- A general RL framework. There are excellent ones; this is not one.
- A symbolic DP solver. Dolo/Dolang serve that audience; bellgrid is numerical.
- A grid-math library. Tasmanian is the gold standard for that; bellgrid is a higher-level DP solver sitting on top of multilinear interpolation on regular and warped grids.
- A domain-specific library for any one application. Utility theory, options theory, control theory — bellgrid is the common substrate. Domain primitives live in the example notebooks.
- An advice / planning product. Solvers compute optimal policies given a reward and constraints. Applying that to a real person's life is a separate, regulated activity.
