# bellgrid solvers

Every solver consumes the same [`Problem`](api.md) and returns the same
`(policy, value)` callables. They differ only in **how** they solve the Bellman
recursion — and which structure of your problem each one exploits to do it. This
page is the deep dive: the math, the scope, and the literature behind each. For a
quick when-to-use-which, see the table in the
[README](../README.md#choosing-a-solver).

## The shared target

Every solver computes (an approximation of) the value function and its greedy
policy for

```
V_t(s) = max_a  E_w[ r(s, a, w, t) + β(s, t) · V_{t+1}( f(s, a, w, t) ) ]
```

with `V_T` fixed by a terminal reward (finite horizon) or a stationary fixed point
(infinite horizon). The shock expectation `E_w` is taken by the **exact quadrature**
attached to each shock — Gauss–Hermite for `Normal`/`Lognormal`, Cholesky-rotated
tensor-product for `MultivariateNormal`, Gauss–Legendre for `Uniform`, exact for
`Categorical`, Bernoulli-arrival × Gauss–Hermite for `Jump`.

The one distinction that organizes the whole portfolio is **how each solver handles
the continuation `V_{t+1}∘f`**:

| solver | continuation handled by | requires | overestimation |
|---|---|---|---|
| **grid** | exact interpolated lookup on a mesh | a mesh (`∏ points`) | none |
| **iLQG** | local 2nd-order model, backprop through `f` | differentiable `f`, `r` | none |
| **PolicyGradient** | full differentiable rollout | differentiable `f`, `r` | none |
| **ActorCritic** | a *learned* critic, bootstrapped | nothing extra | yes — controlled by an ensemble |

The first three **differentiate through the model**; the last **learns a value and
bootstraps it**. That single choice is the entire tradeoff: differentiating is exact
and bias-free but needs a smooth, continuous, differentiable model; bootstrapping a
learned value tolerates discrete / non-smooth dynamics but pays for it in
approximation bias (the overestimation the actor's `argmax` exploits). Route to the
one whose assumption your problem actually satisfies.

---

## Grid — `BackwardInduction` / `PolicyIteration`

**Exact dynamic programming on a discretized mesh.** Lay a grid over the continuous
state, evaluate the Bellman operator at every node, and store `V`/`π`. The maximum
over actions is by enumeration over an action grid; the continuation `V_{t+1}(s')`
is read back by **multilinear interpolation** at the (off-grid) next-state
coordinates; the shock expectation is the exact quadrature above.

```
V_t(s_i) = max_{a_j}  Σ_k w_k · [ r(s_i, a_j, w_k, t) + β · interp(V_{t+1}, f(s_i, a_j, w_k, t)) ]
```

`BackwardInduction` sweeps this backward from `V_T` over a finite horizon.
`PolicyIteration` solves the infinite-horizon stationary problem by alternating
policy evaluation and greedy improvement, with **modified policy iteration**
(`k_howard` inner Bellman sweeps per improvement; Puterman) to trade evaluation
accuracy against iteration count.

- **Exactness:** no learned approximation — the only error is the `O(h²)`
  interpolation bias of multilinear-on-curved-`V`, which vanishes as the grid
  refines, *everywhere*, including the tails.
- **Cost:** the curse of dimensionality — `∏ (points per dimension)`, hopeless past
  ~6 continuous state/action dimensions. This is the wall the other three solvers
  exist to get past.
- **Scope:** anything the `Problem` spec expresses — continuous **and** discrete
  states (incl. `MarkovChain`), continuous **and** discrete actions, finite and
  infinite horizon, callable discount, optimal stopping. The most general solver.

*References:* Bellman, *Dynamic Programming* (1957); Howard, *Dynamic Programming and
Markov Processes* (1960, policy iteration); Puterman, *Markov Decision Processes*
(1994, modified policy iteration); Judd, *Numerical Methods in Economics* (1998);
Carroll, "The method of endogenous gridpoints" (2006), for the EGM variant the
`02_carroll_deaton` example checks against.

---

## iLQG — Differential Dynamic Programming

**Newton's method on a trajectory.** For a continuous, finite-horizon problem with a
differentiable model, iLQG builds the *local quadratic model* of the cost-to-go from
autograd derivatives of your `transition`/`reward` and solves that LQ subproblem in
closed form. (We optimise in cost units, `cost = −reward`, since bellgrid maximises;
the value returned is `−cost`.)

**Forward pass.** Roll the current controls forward to a nominal trajectory
`(x̄_t, ū_t)`.

**Backward pass.** With `f` the dynamics, `l` the stage cost, and `V_x, V_xx` the
gradient/Hessian of the cost-to-go (seeded at the terminal), form the Q-function's
quadratic expansion and the affine control law at each `t` (the Gauss–Newton / iLQR
variant — the `f_xx` term of full DDP is dropped; it is exactly zero on an LQ
problem):

```
Q_x  = l_x  + β·fₓᵀ V_x
Q_u  = l_u  + β·f_uᵀ V_x
Q_xx = l_xx + β·fₓᵀ V_xx fₓ
Q_uu = l_uu + β·f_uᵀ V_xx f_u            (+ μ·I  Levenberg–Marquardt regularisation)
Q_ux = l_ux + β·f_uᵀ V_xx fₓ

k = −Q_uu⁻¹ Q_u            (feed-forward)
K = −Q_uu⁻¹ Q_ux           (feedback gain)

V_x  ← Q_x  + Kᵀ Q_uu k + Kᵀ Q_u + Q_uxᵀ k
V_xx ← Q_xx + Kᵀ Q_uu K + Kᵀ Q_ux + Q_uxᵀ K
```

**Forward line search.** Apply `u_t = ū_t + α·k_t + K_t·(x_t − x̄_t)`, accept the
step size `α` that reduces cost. Additive (certainty-equivalent) noise contributes a
constant `½·tr(Cᵀ V_xx C Σ)` to the value at each step (`C = ∂f/∂w`, `Σ` the shock
covariance) — the only place the shock enters in v1.

The returned policy is the **time-varying affine feedback law**
`u_t(x) = ū_t + K_t·(x − x̄_t)`.

- **Exactness:** on a linear-quadratic problem `l` is exactly quadratic and `f`
  exactly linear, so the model is exact and **one Newton step is the matrix-Riccati
  solution** — gains, trajectory, and value to machine precision, at any dimension.
  Off LQ it converges to a *local* optimum around the trajectory.
- **Scope (v1):** all-`ContinuousState`, `ContinuousAction`, scalar discount, finite
  horizon, additive shocks; **unconstrained** (returns the unconstrained optimum,
  exact where bounds are slack). Box bounds are control-limited DDP — a planned
  follow-up. Discrete/Markov states, discrete actions, callable discount, and
  infinite horizon raise with a pointer to the grid solver.

*References:* Mayne, "A second-order gradient method…" (1966); Jacobson & Mayne,
*Differential Dynamic Programming* (1970); Li & Todorov, "Iterative linear quadratic
regulator design…" (2004, iLQR); Todorov & Li, "A generalized iterative LQG method…"
(2005, iLQG); Tassa, Erez & Todorov, "Synthesis and stabilization of complex
behaviors through online trajectory optimization" (2012); Tassa, Mansard & Todorov,
[Control-Limited Differential Dynamic Programming](https://roboti.us/lab/papers/TassaICRA14.pdf)
(2014).

---

## PolicyGradient — pathwise (analytic) policy gradient

**Backpropagate the return through the differentiable model.** Parameterise a policy
`π_θ` (per-period nets), roll it forward under reparameterized (sampled) shocks using
your own `transition`/`reward`, and take the gradient of the expected return directly
w.r.t. `θ`:

```
maximise  J(θ) = E_{s₀, w}[ Σ_t β^t r(s_t, π_θ(s_t), w_t) + β^T g(s_T) ],
                  with  s_{t+1} = f(s_t, π_θ(s_t), w_t)
```

Because each shock is **reparameterized** (sampled as fixed noise per path, e.g.
`w = σ·ε`), the gradient flows through the deterministic `f`/`r` w.r.t. the action —
this is the **pathwise / "stochastic value gradient"** estimator (the
reparameterization trick + backprop-through-time), *not* the score-function /
REINFORCE estimator `∇_θ J = E[Σ ∇_θ log π_θ · R]`. The difference matters: with a
differentiable model the pathwise estimator is **low-variance** and, crucially,
**uses no learned critic and no bootstrap** — so the entire overestimation drawer of
value-based RL (twin critics, ensembles, value expansion) is simply absent. The model
itself supplies the value gradient. `value(s, t)` is an honest Monte-Carlo rollout of
the trained policy — the value *is* what the policy earns.

- **Behaviour:** on the LQ liquidation example it reaches a ~1% optimality gap with
  no critic, beating the ActorCritic's tuned stack — because there is no
  bootstrap-overestimation bias to fight in the first place.
- **Scope (v1):** all-`ContinuousState`, `ContinuousAction`, scalar discount, finite
  horizon — the model must be differentiable, so a discrete-state transition or
  discrete action (non-differentiable) breaks the pathwise gradient and is rejected.
  Full-horizon backprop suits short horizons; very long or chaotic rollouts can have
  exploding gradients, fixed by **truncating** the rollout and bootstrapping a learned
  terminal value (SHAC) — a planned follow-up that meets ActorCritic in the middle.

*References:* Heess et al.,
[Learning Continuous Control Policies by Stochastic Value Gradients](https://arxiv.org/abs/1510.09142)
(2015, SVG); Xu et al.,
[Accelerated Policy Learning with Parallel Differentiable Simulation](https://cdfg.mit.edu/assets/files/shac_iclr_2022.pdf)
(2022, SHAC); Suh et al.,
[Do Differentiable Simulators Give Better Policy Gradients?](https://arxiv.org/abs/2202.00817)
(2022, on when first-order gradients beat zeroth-order); Mohamed et al., "Monte Carlo
Gradient Estimation in Machine Learning" (2020, the pathwise-vs-score-function
taxonomy); Williams, "Simple statistical gradient-following algorithms…" (1992,
REINFORCE — the contrast).

---

## ActorCritic — model-based neural actor–critic

**Learn `V`/`π` as networks, bootstrap the continuation.** The one neural solver that
does *not* differentiate through the model — so it is the only one that tolerates a
step it cannot differentiate (a discrete state / regime, a non-smooth transition). It
samples states, evaluates candidate actions against `E_w[r + β·V_{t+1}]` (the exact
quadrature, the *learned* critic for `V_{t+1}`), regresses the actor onto the
candidate `argmax`, and fits the critic to the on-policy Bellman target:

```
a_target(s) = argmax_a  Ê_w[ r(s, a, w) + β · V̂_{t+1}(f(s, a, w)) ]      (candidate search)
critic:  fit V̂_t(s) toward  Ê_w[ r(s, a_onpolicy, w) + β · V̂_{t+1}(·) ]   (on-distribution)
```

Bootstrapping a *learned* critic re-introduces the **deadly triad**: the `argmax`
systematically selects actions where `V̂` over-estimates, the actor trains toward it,
and the bias **compounds backward** over the horizon. More samples cannot fix it (the
Bellman expectation is already exact — it is approximation bias, not variance). The
controls that keep it honest, all composable:

- **Truncated critic ensemble** (`n_critics`, `drop_top_atoms`) — REDQ/TQC: pool the
  ensemble's value atoms and drop the most optimistic before averaging, removing
  exactly the optimism the `argmax` exploits. The heavy lifter (≈6%→1% reported-value
  bias at 80-D). `twin_critic` is the clipped-double-Q (TD3) special case.
- **Model-based value expansion** (`value_expansion`, `search_expansion`) — MVE: roll
  the *exact* model forward `k` steps before bootstrapping, shrinking the learned
  critic's (biased) share of the target.

- **Scope:** `ContinuousState` **and `DiscreteState`** (one-hot featurised; the regime
  may evolve), `ContinuousAction`, finite horizon. `DiscreteAction` and `MarkovChain`
  are planned. Its genuine niche is **high-D problems with discrete states or
  non-differentiable dynamics** — where the grid dies on dimension and the
  through-the-model solvers can't take the gradient.

*References:* Fujimoto et al.,
[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
(2018, TD3); Chen et al., "Randomized Ensembled Double Q-Learning: Learning Fast
Without a Model" (2021, REDQ); Kuznetsov et al., "Controlling Overestimation Bias with
Truncated Mixture of Continuous Distributional Quantile Critics" (2020, TQC); Feinberg
et al., "Model-Based Value Expansion…" (2018, MVE); Munos & Szepesvári, "Finite-time
bounds for fitted value iteration" (2008, the error-propagation theory).

---

## Cross-certification

Because all four share the `Problem` spec, **each certifies the others wherever they
overlap** — the library's core correctness story:

- **Against an exact oracle.** A linear-quadratic problem has a closed-form
  matrix-Riccati solution at *any* dimension. The
  [`11_liquidation`](../examples/11_liquidation/liquidation.ipynb) example certifies
  `iLQG` against it to **machine precision** and the neural solvers to ~1–2% at **80
  state dimensions** (~10¹⁷⁶ equivalent grid cells) — *matching the exact answer*, not
  just self-consistency.
- **Against the grid.** Where there is no closed form, the exact grid solver certifies
  the high-D solvers on the low-dimensional overlap
  ([`10_hydropower`](../examples/10_hydropower/hydropower.ipynb): exact at one
  reservoir), then `simulate()` checks forward-consistency at the scale where no grid
  can exist. The simulator shares the solver's `transition`/`reward`, so "the
  simulator was wrong" is not a possible bug.

Prove it where you can; trust it where you must.
