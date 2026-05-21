# bellgrid documentation

- [When to use bellgrid (vs. RL)](when_to_use.md) — start here; this is the positioning piece.
- [API draft](api.md) — public surface, subject to change.
- [Examples](../examples/) — Merton, Carroll/Deaton, American option, LQG (all four canonical examples).

## Status

The four canonical examples (Merton, Carroll/Deaton, American option, LQG) all run end-to-end and validate against analytical or numerical references. Core primitives in place: `ContinuousState`, `ContinuousAction`, `DiscreteAction`, `Normal` / `Lognormal` shocks with Gauss-Hermite quadrature, `RegularGrid` / `WarpedGrid` (asinh, log), JIT-compiled K-D multilinear interpolation, finite-horizon `BackwardInduction` solver on CPU or CUDA.

Planned but not yet implemented: `DiscreteState` / `MarkovChain` (regime-switching states), `MultivariateNormal` / `Jump` shocks, `PolicyIteration` (infinite-horizon), implicit differentiation, warm-starting.
