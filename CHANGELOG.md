# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/) — the leading `0.`
indicates the API may still change in non-additive ways before a `1.0` release.

## [0.1.0a6] — 2026-06-05

Debuts the neural dynamic-programming solver: a model-based actor–critic for
high-dimensional problems no grid can reach, with a worked example that certifies
it against an exact analytical oracle at 80 state dimensions.

### Added
- **`ActorCritic` solver** — a model-based actor–critic that runs behind the same
  `Problem` spec as the grid solvers, so a grid solve can *certify* it wherever
  both run. Finite-horizon backward sweep with exact-quadrature Bellman targets
  (the model and the shock expectation are the `Problem`'s own, not learned).
- **Overestimation / compounding controls on `ActorCritic`** (all composable):
  - `twin_critic` — clipped double-Q (TD3, Fujimoto et al. 2018).
  - `n_critics` + `drop_top_atoms` — truncated critic ensemble (REDQ/TQC): pool
    the critics and drop the most optimistic atoms, removing the over-estimation
    the actor's argmax exploits before it compounds backward over the horizon.
    At 80-D this pulls the reported value from a seed-dependent ~6% off to ~1%.
  - `value_expansion` + `search_expansion` — model-based value expansion: roll the
    *exact* model forward `k` steps before bootstrapping, shrinking the bootstrap's
    (critic-error) share of the target. `search_expansion` decouples the dearer
    candidate-search horizon from the critic-target horizon so deeper, low-bias
    targets stay affordable at high dimension.
  - `critic_quantiles` — distributional (TQC-style) critic atoms.
- **`examples/11_liquidation`** — optimal execution (stochastic Almgren–Chriss).
  Linear impact ⇒ exact LQ ⇒ a matrix-Riccati closed form is ground truth at any
  dimension, so the example certifies `ActorCritic` at 40 assets / 80-D — where no
  grid can exist — three ways: policy match where the no-short constraint is slack,
  value-vs-Monte-Carlo consistency (~0.3%), and optimality vs the Riccati bound
  (~1.8%). Includes a "trade-rate error ≠ value error" analysis: the objective is
  flat in the trade near the optimum (envelope theorem), so a ~23%-median trade
  deviation forfeits only ~0.005% of the total return.

### Changed
- CI release actions bumped to Node 24 (`checkout@v6`, `setup-uv@v8`).

## [0.1.0a5] — 2026-05-29

A correctness fix plus internal memory/performance work. No public API
changes; existing solves are unaffected except where they previously crashed.

### Fixed
- **State-dependent callable discount no longer crashes `PolicyIteration`
  or `GoldenSearch`.** A callable discount that depends on state (returns a
  tensor, e.g. `torch.where(state["regime"] == 0, b0, b1)`) is evaluated on
  the joint state mesh, so it arrives with size `N_a` on the action axis. The
  two solver paths that re-use the Bellman machinery at a *different*
  action-axis size — `PolicyIteration`'s Howard / modified-policy-iteration
  inner loop (`M = 1`, hit whenever the default `k_howard = 10` is used) and
  `GoldenSearch` refinement (`M = N_enum`) — broadcast the `N_a` discount
  against the `M`-action value and raised a shape error. So *any*
  infinite-horizon problem with a state-dependent callable discount failed out
  of the box (only `k_howard = 1` worked), as did golden search with one.
  Fixed centrally in `_bellman_at_action_values` by collapsing the discount's
  action axis to 1 so it broadcasts to any `M`.

### Changed
- **Chunk-size budget accounts for the Markov kept-axis fan-out.** The
  per-chunk working tensor is the pre-contraction lookup, which is
  `prod(n_ms)`× larger than `state × action × shock`; the cap now divides by
  that, so problems with sizable `MarkovChain`s chunk correctly instead of
  silently under-counting and risking OOM. The accumulator-fits check stays
  fan-out-free (the accumulator is post-contraction).
- **Multilinear bracketing is no longer recomputed across the Markov axes.**
  Non-Markov queries carry trailing size-1 markov dims rather than being
  expanded to `lookup_shape`, so `searchsorted` + interp weights run at chunk
  size and only the final gather broadcasts up — removing `prod(n_ms)`×
  redundant work and the matching materialisation. `multilinear()` routes to
  the compiled core on the broadcast output size.
- **Per-chunk accumulation is in-place**, and **state-independent action
  tensors** (fixed-bound continuous, all discrete) are kept as broadcast views
  rather than materialised to the full `state × N_a`. Transition return-key
  validation now runs once per Bellman step rather than once per chunk.

### Test count
- 291 tests pass (up from 289 in 0.1.0a4; 2 new regression tests covering the
  callable-discount fix on the `PolicyIteration` and `GoldenSearch` paths).

## [0.1.0a4] — 2026-05-27

Two performance features, both opt-in and non-breaking: vectorized
golden-section action search, and modified policy iteration for the
infinite-horizon solver.

### Added
- **`GoldenSearch` action grid spec** — drop-in replacement for
  `RegularGrid` in `action_grid` for continuous actions. Replaces
  brute-force enumeration with a small seed grid (`n_init`, default 4)
  plus vectorized golden-section refinement (`n_iter`, default 20) — run
  lock-step across every state and every joint configuration of the
  other actions. Single-fresh-eval-per-iter via the standard
  golden-ratio trick, so the GPU work is one Bellman call per iter.
  Multi-continuous problems get `n_coord` (default 2) outer rounds of
  coordinate descent. Mixes naturally with `DiscreteAction` (enumerated
  alongside) and with non-golden `RegularGrid` continuous actions. On a
  warped-wealth log-utility Merton, golden with `n_init=4, n_iter=20`
  saturates the wealth-grid precision floor at ~1/23rd the wall time of
  a 500-point `RegularGrid` (same final precision).
- **`k_howard` knob on `PolicyIteration`** — promotes the historical
  value-iteration loop to **modified policy iteration**. Each outer
  iteration is now one Bellman improvement (the expensive `max_a` step)
  followed by `k_howard − 1` cheap Bellman evaluations at the
  freshly-improved policy. Eval steps cost `~1 / N_a` of an improvement,
  so even a moderate `k_howard` is nearly free per outer iter but cuts
  the outer-iter count dramatically. Defaults to `k_howard=10`;
  `k_howard=1` reverts to the previous value-iteration behaviour.
  On infinite-horizon Merton (β=0.96, 64×200 grid, tol=1e-7), outer
  iterations drop from 444 (`k=1`) → 51 (`k=10`, default) → 9 (`k=200`);
  wall-clock goes from 5.95s → 0.52s → 0.24s on CPU.

### Changed
- **`SolveContext` carries the raw shock-node tensors, resolved action
  bounds, and joint-axis strides** — needed by the new refinement paths
  to rebuild Bellman broadcasts at arbitrary action values without
  re-running shock setup. Additive on the dataclass; no public API
  change.
- **Internal: `_bellman_core` extracted from `_bellman_partial`**, so the
  Bellman integrand can be evaluated at pre-broadcast tensors that come
  from either the existing chunked grid path (`_bellman_partial`) or the
  new per-state-value paths (`_bellman_at_action_values`). Shared by the
  golden-search refinement and Howard's policy-evaluation step. No
  behavioural change to the grid path.

### Known limitations (carried over)
- `GoldenSearch` combined with a non-golden `RegularGrid` large enough
  to force action chunking will raise rather than silently regressing.
  The refinement needs the full `state × N_a` accumulator; the
  combination is also wasteful (full grid cost on the non-golden axis,
  no speedup from golden) so an explicit error is preferable.
- No implicit differentiation through the solver. Still planned.

### Test count
- 289 tests pass (up from 272 in 0.1.0a3; 9 new for `GoldenSearch`, 8
  new for Howard).

## [0.1.0a3] — 2026-05-22

No API changes. Ships the **full lifecycle planning example**, four
test files filling coverage gaps surfaced by an internal review, and
a tokenless release workflow.

### Added
- **`examples/09_lifecycle_planning/`** — the canonical lifecycle DP
  problem that motivated bellgrid: an agent from age 25 to 100
  choosing consumption, retirement age, and equity share under
  mortality, regime-switching markets, a deterministic age-earnings
  profile, and a warm-glow bequest motive. Exercises callable
  discount + next-state-aware reward + MarkovChain + DiscreteState +
  state-dependent bounds in one problem. Helpers in `mortality.py`,
  `utility.py`, `wages.py`, `regimes.py` — adapted from the rl-inv2
  project.
- **Coverage-gap tests** (16 new) addressing the post-0.1.0a2 review:
  - `test_simulate_with_categorical_uniform.py` — Categorical /
    Uniform shocks in `simulate()`, with 5σ empirical-distribution
    checks.
  - `test_three_markov_chains.py` — 3 and 4 independent
    MarkovChains; 4-chain Kronecker-product equivalence vs a single
    16-state MarkovChain.
  - `test_markov_callable_combo.py` — callable discount sees the
    current regime; 5-arg reward's `next_state` excludes MC keys;
    end-to-end combo with hand-derived analytic V (V_r1 = 2.5x).
  - `test_boundary_with_markov_chain.py` — boundary diagnostic with
    stochastic + regime-dependent dynamics; no spurious warning on
    clean MC problems; opt-out works.
- **`.github/workflows/release.yml`** — GitHub Actions workflow that
  publishes to PyPI via Trusted Publishing (OIDC) on `v*` tag push,
  with the GitHub Release body auto-extracted from the matching
  CHANGELOG section. No API tokens required.

### Test count
- 271 tests pass (up from 255 in 0.1.0a2).

## [0.1.0a2] — 2026-05-22

Substantial alpha-2: two new shocks, two solver-side capabilities, three
API loosenings, and a memory-management feature. All non-breaking.

### Added
- **`Categorical` shock** — finite-support discrete iid shock. Quadrature
  is exact (K nodes = values, K weights = probabilities; `n_quad` is
  ignored).
- **`Uniform` shock** — continuous uniform on `[low, high]` with
  Gauss-Legendre quadrature.
- **Callable discount** — `discount(state, t)` may now return a scalar
  or any tensor broadcastable to the state mesh. Use for mortality,
  equipment-failure hazards, or any other state/age-dependent
  termination factor.
- **Next-state-aware reward** — `reward` may declare a 5th positional
  argument and receive the dict of next-state values returned by
  `transition`. Detected via `inspect.signature` (4-arg form
  unchanged). Combined with callable discount, the two cleanly
  express mortality + bequest as
  `V_t = u(c) + β·E[p_survive·V_{t+1} + (1-p_survive)·Bequest(s')]`.
- **Multiple `MarkovChain`s per problem** — previously capped at one;
  now any number. Cost is additive in chains (one extra matrix-
  contraction per chain) rather than multiplicative (which is what
  baking them into a single product chain would have cost).
- **Memory-chunked Bellman update** — the `chunk_size` parameter on
  `solve()` was previously accepted-but-ignored; it now caps the
  per-Bellman-step memory by splitting the shock axis into chunks.
  Verified by a regression test that `chunk_size=1` and
  `chunk_size=2**30` produce the same V on a Merton problem.
- **Boundary-escape diagnostic** — after each solve, a single
  `problem.transition` call with the optimal policy is used to
  measure the weighted fraction of next-states that fall outside
  each `ContinuousState`'s range. Emits a `UserWarning` per state
  whose interior-mean escape exceeds 10%. Cheap (<5% overhead on the
  smallest solves, near-zero on big ones). Opt-out via
  `BackwardInduction(boundary_check=False)` / `PolicyIteration(...)`.
- **`simulate()` parity with the solver** — now accepts callable
  discount, infinite-horizon `Problem`s (via a new `n_periods`
  parameter), and 5-arg next-state-aware reward.
- **State-dependent action bounds with K_cont > 1** — previously
  restricted to exactly one `ContinuousState`; now references any
  declared continuous state regardless of how many there are.

### Changed
- **Multilinear interpolation now supports mixed continuous and discrete
  axes** in a single call. Detected per-query by `dtype`: floating →
  continuous (interpolated), integer → discrete (exact gather).
  Discrete axes don't contribute corners, so cost is `2 ** K_cont`
  corner gathers rather than `2 ** K_total`.
- **Transition return-dict validation** moved upfront and aggregated:
  one `ValueError` listing all missing / forbidden keys instead of
  per-axis errors raised mid-loop.
- **Style sweep**: `Optional[T]` → `T | None` and `Union[A, B]` → `A | B`
  across the source; dropped now-unused `typing` imports. Pure cleanup.
- **Docstring pass**: filled in docstrings for `ContinuousState`,
  `ContinuousAction`, and `Problem` (the three primitives that were
  bare while their siblings had rich docstrings).

### Fixed
- `torch.searchsorted` non-contiguous warning was firing on every
  Bellman step with a markov chain. The expanded/strided query is
  now materialised at the call site, silencing the warning at no
  additional cost (the kernel was doing the copy internally anyway).

### Known limitations (unchanged from 0.1.0a1)
- At most one `MarkovChain` per problem → **lifted in 0.1.0a2**.
- The user's `transition` and `reward` only see the **current**
  value of any `MarkovChain` state; the next value is integrated
  internally via the matrix and isn't exposed. For dynamics that
  depend on the next markov value (e.g. a bond return tied to yield
  drift between regimes), model the state as a `ContinuousState`
  AR-process or as a `DiscreteState` with hand-rolled stochastic
  dynamics.
- Single-axis (shock-only) chunking. For problems where state ×
  action dominates memory, chunk_size on the shock axis alone may
  not be enough — a future release will extend chunking to action
  and/or state axes.
- No implicit differentiation through the solver. Still planned.

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

[0.1.0a4]: https://github.com/tbb300/bellgrid/releases/tag/v0.1.0a4
[0.1.0a3]: https://github.com/tbb300/bellgrid/releases/tag/v0.1.0a3
[0.1.0a2]: https://github.com/tbb300/bellgrid/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/tbb300/bellgrid/releases/tag/v0.1.0a1
[0.1.0a0]: https://github.com/tbb300/bellgrid/releases/tag/v0.1.0a0
