# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/) — the leading `0.`
indicates the API may still change in non-additive ways before a `1.0` release.

## [0.1.0a1] — 2026-05-22

PyPI-rendered README fix: relative links to `examples/` and `docs/` in the
README don't resolve on the PyPI project page (they only work on GitHub's
file-tree rendering). Rewrote all of them as absolute GitHub URLs so the
PyPI page lands correctly. Also added a `pip install bellgrid` quick-start
since the alpha is now on PyPI. No code changes.

## [0.1.0a0] — 2026-05-21

First alpha release. The library has eight validated example notebooks, ~200
tests, and every primitive listed in the original design doc except implicit
differentiation. The API may still change before `0.1.0` proper, but the
public surface is now stable enough to start using on real problems and
collecting feedback.

### State primitives
- `ContinuousState` with optional `asinh` / `log` warp (or a user callable).
- `DiscreteState` — finite-state variable whose dynamics the user supplies
  in `transition`.
- `MarkovChain` — discrete state with a built-in row-stochastic transition
  matrix advanced internally by the solver.

### Action primitives
- `ContinuousAction` with optional state-dependent bounds (e.g.,
  `bounds=(0, "wealth")`).
- `DiscreteAction`.

### Shocks (all via Gauss-Hermite quadrature)
- `Normal` (univariate Gaussian).
- `Lognormal`.
- `MultivariateNormal` — K-dim correlated Gaussian via Cholesky-rotated
  tensor-product Gauss-Hermite.
- `Jump` — Bernoulli-approximated Poisson with Normal log-magnitudes.
- Multiple independent shocks per problem are combined via tensor-product
  quadrature; at most one of any kind is the comfortable territory.

### Grids
- `RegularGrid` (uniform spacing).
- `WarpedGrid` (inherits the warp from the corresponding `ContinuousState`).

### Solvers
- `BackwardInduction` for finite-horizon problems — `T` sweeps of the
  Bellman operator from an optional `terminal_reward`.
- `PolicyIteration` for infinite-horizon stationary problems — value
  iteration to a `tol` convergence threshold on `||V_new − V||_∞`.

### Interpolation
- JIT-compiled K-dimensional multilinear with mixed-axis support
  (continuous axes interpolated, discrete / markov axes exact-gathered;
  detection by query dtype).
- Auto-dispatch to a `torch.compile`d kernel above a query-size threshold
  (~10× speedup at K=2 / 12 M queries).

### Engine
- CPU or CUDA, picked automatically (`device="cuda"` if available).
- All transition / reward evaluation vectorised across the joint
  state × action × shock grid in a single pass.

### Examples (`examples/0?_*`)
| Notebook | Problem | Validates against |
|---|---|---|
| 01 Merton | Log-utility consumption-portfolio | Closed form `V = A + B log w`, validated via both `BackwardInduction` and `PolicyIteration` |
| 02 Carroll/Deaton | CRRA lifecycle savings with borrowing constraint | Endogenous Grid Method (Carroll 2006) |
| 03 American option | American put on GBM | CRR binomial tree (n=2000), agreement within ~1e-4 |
| 04 LQG | 2-D linear-quadratic-Gaussian control | Discrete-time Riccati recursion |
| 05 Two-asset Merton | Correlated returns (`MultivariateNormal`) | Numerical FOC for the optimal portfolio share |
| 06 Regime-switching option | American put under regime-switching vol (`MarkovChain`) | Bracketed by constant-vol references |
| 07 Retirement decision | Lifecycle work vs retire (`DiscreteState`, irreversible) | Qualitative — boundary falls with age |
| 08 Jump-diffusion option | American put under Merton (1976) jump-diffusion (`Jump` + `Normal`) | Merton 1976 series-expansion European reference (within ~1e-3) |

### Known limitations
- At most one `MarkovChain` per problem.
- State-dependent action bounds only when there's exactly one `ContinuousState`.
- The `chunk_size` parameter on `solve()` is currently accepted but unused
  (memory-chunked Bellman updates haven't landed yet).
- No implicit differentiation through the solver.
- No infinite-horizon `MarkovChain` initial-state defaulting to the stationary
  distribution in `simulate()` (users pass an explicit category index for now).

[0.1.0a1]: https://github.com/tbb300/bellgrid/releases/tag/v0.1.0a1
[0.1.0a0]: https://github.com/tbb300/bellgrid/releases/tag/v0.1.0a0
