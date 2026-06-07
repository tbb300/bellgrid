"""``ActorCritic`` — model-based actor-critic solver (finite horizon).

A neural counterpart to ``BackwardInduction`` that breaks the grid's
curse-of-dimensionality by representing ``V`` and ``π`` as networks over sampled
states instead of tensors over a mesh. It is *model-based*: it uses the
``Problem``'s known, differentiable ``transition``/``reward`` and the exact shock
quadrature, so each Bellman target is the genuine expectation, not a bootstrap
off single transitions.

Structure = neural policy iteration run as a **backward sweep**. With ``V_{t+1}``
frozen (the terminal reward at the last step, the trained critic thereafter),
each period fits two networks against *stationary* targets — no moving bootstrap,
which is what makes the finite-horizon case the stable on-ramp. With ``V_{t+1}``
frozen, each step evaluates a **candidate set** of actions — the actor's own
proposal plus globally-sampled and locally-perturbed ones — against it, and
splits the result two ways:

  actor (improvement):  π_φ(s) ← argmax over candidates of E_w[r + β·V_{t+1}(s')]
  critic (evaluation):  V_θ(s) ← E_w[ r(s, π_φ(s), w, t) + β·V_{t+1}(s') ]

The actor improves toward the candidate-max action; the critic evaluates the
actor's *own* current action (**on-policy**). Two consequences matter:

  - The candidate max decouples improvement from the critic's gradient, so the
    actor can't drift into a region where the critic mis-extrapolates (the
    "consume everything in Merton" failure of a pathwise actor). It also seeds
    good candidates as the action dimension grows.
  - The on-policy critic target makes ``V_θ`` the value of the policy actually
    run — so it matches forward simulation *by construction*, rather than the
    optimistic candidate-max that an approximate actor cannot fully attain.
    (Fitting the max instead leaves ``V_solve`` biased above the achievable
    return — badly so on hard policies.)

The critic "compiles" period ``t``'s value into a reusable function so the
recursion stays O(T); it then becomes the frozen target for period ``t-1``.

Returns the same ``(policy, value)`` callables as the grid solver, so downstream
code and ``simulate`` are unchanged — and the grid solver can certify this one on
any problem small enough to solve both ways.

Correctness: the solution is approximate — the inner max is over a finite
candidate set and the nets approximate ``V``/``π`` over the sampled state region.
Sanity-check against the grid solver where both run, and against forward
simulation (which the on-policy critic is built to agree with) where it can't;
the per-period critic RMSE on ``value.residual_by_t`` is a cheap fit proxy.
"""

from dataclasses import dataclass

import torch

from ._setup import RLSetup, build_setup
from .nets import MLP, NormalizedCritic, squash_to_bounds


@dataclass(frozen=True)
class ActorCritic:
    """Model-based actor-critic solver for finite-horizon problems.

    Attributes
    ----------
    n_quad : int
        Quadrature nodes per shock dimension — the shock expectation is exact,
        as in the grid solver.
    hidden : tuple[int, ...]
        Hidden-layer widths for the actor MLP (and the critic too, unless
        ``critic_hidden`` is set).
    critic_hidden : tuple[int, ...] | None
        Hidden-layer widths for the critic MLP. Defaults to ``hidden``. The
        policy's accuracy is bounded by the critic's value fit (the actor
        regresses onto the argmax of the critic), so for hard high-dimensional
        problems it pays to make the critic much larger than the actor.
    twin_critic : bool
        Clipped double-critic (TD3-style; default ``False``). Trains two
        independent critics per period and uses their elementwise **minimum** as
        the continuation value (and the reported value). The actor regresses onto
        ``argmax`` over candidate actions of ``E[r + V_{t+1}]``; that argmax
        systematically selects actions where the critic *over-estimates*
        (the optimizer's curse), and because the actor is trained toward it the
        "on-policy" value inherits the bias and **compounds backward** through the
        bootstrap — the source of the seed-dependent value blow-ups on hard
        high-dimensional problems (the value can drift *above* the true optimum).
        More samples don't help: the Bellman expectation is already exact
        quadrature, so the error is approximation bias, not variance. Taking the
        min of two critics with independent initialisations cancels the
        per-critic over-estimation the argmax exploits, which is what makes the
        backward recursion stable across seeds. Cost: a second critic (cheap
        relative to the candidate search). See Fujimoto et al. 2018, "Addressing
        Function Approximation Error in Actor-Critic Methods". This is the
        ``n_critics=2, drop_top_atoms=1`` special case of the truncated ensemble
        below; setting ``twin_critic=True`` is a convenience for exactly that.
    n_critics : int
        Size of the critic **ensemble** (default 1). The continuation value is the
        *truncated mean* of the pooled atoms across all critics (see
        ``drop_top_atoms``), generalising the twin-critic min to an arbitrary
        ensemble (REDQ / TQC style). More critics give finer, lower-variance
        control of the over-estimation bias than the binary min of two.
    critic_quantiles : int
        Quantile heads per critic (default 1 ⇒ a plain scalar critic). With
        ``> 1`` each critic predicts a *distribution* over the return (the spread
        induced by the shock), fit by quantile/pinball regression — the
        distributional half of TQC (Kuznetsov et al. 2020). The pooled atoms used
        for truncation are then ``n_critics × critic_quantiles``.
    drop_top_atoms : int
        How many of the highest pooled atoms to drop before averaging to form the
        continuation value (default 0 ⇒ plain ensemble mean / single critic). This
        is the **truncation** knob: a continuous version of the twin-critic min
        (which is ``n_critics=2, drop_top_atoms=1``). Dropping the most optimistic
        atoms removes exactly the over-estimation the actor's argmax would exploit,
        with finer control than a hard min — raise it to be more conservative,
        lower it to report a tighter (less pessimistic) value.
    value_expansion : int
        Horizon ``k`` of the **model-based value expansion** target (default 1 ⇒
        the usual one-step bootstrap). With ``k > 1`` the continuation is computed
        by rolling the *frozen* policy forward ``k-1`` steps through the known
        model — accumulating exact per-step rewards and bootstrapping only at
        ``V_{t+k}`` — instead of bootstrapping immediately at ``V_{t+1}``. Because
        the model and the shock expectation are exact here (it's the ``Problem``'s
        own ``transition``/``reward`` under quadrature, not a learned model), the
        expansion is *unbiased* and strictly reduces the bootstrap's share of the
        target, which is what compounds backward over the horizon. This is where
        unlimited compute actually buys accuracy: each extra step costs a factor
        ``n_quad`` (nested quadrature), fully parallel. See Feinberg et al. 2018,
        "Model-Based Value Expansion". Keep ``k`` small (2–3); deep rollouts blow
        up memory as ``n_quad^k``.
    search_expansion : int | None
        Rollout horizon used for the **candidate search** (the actor's argmax
        target), decoupled from ``value_expansion`` which sets the horizon of the
        **critic target**. Defaults to ``None`` ⇒ same as ``value_expansion``. The
        expansion's cost is dominated by the search, which rolls every one of the
        ``M`` candidates forward (``n_quad^{k-1}`` nested quadrature *per candidate*),
        whereas the critic target rolls only the single on-policy action — so a
        ``k``-step target is cheap but a ``k``-step search is ``~M×`` dearer. Setting
        ``search_expansion=1`` keeps the low-bias ``k``-step *target* (which is what
        compounds backward through the bootstrap, and the dominant source of the
        value error) while running the search at the cheap one-step horizon: most of
        the accuracy at close to baseline cost. The search loses only the
        within-period sharpening of looking ``k`` exact steps ahead — and because the
        critic it searches against is now low-bias, that loss is small. Use it to
        make value expansion affordable at high dimension.
    activation : str
        ``"silu"`` (default), ``"tanh"``, ``"relu"``, or ``"gelu"``.
    state_samples : int
        States drawn per gradient step (resampled each step for coverage).
    steps : int
        Gradient steps per period (each updates both nets against the
        candidate-max target).
    lr : float
        Adam learning rate for both nets (the initial rate when ``anneal_lr``).
    anneal_lr : bool
        Cosine-anneal the learning rate to ``lr * 1e-2`` over each period's
        ``steps`` (default ``True``). A fixed LR leaves Adam jittering at a
        few-percent relative-error noise floor; annealing drives the per-period
        fit well below it. The benefit grows with ``steps``.
    value_target : str
        What the critic regresses onto. ``"on_policy"`` (default) fits the value
        at the actor's own action, so ``V_θ`` equals the value of the policy
        actually run (matches forward simulation by construction). ``"max"`` fits
        the Bellman max (value at the candidate-argmax action), estimating the
        *optimal* value directly — closer to an exact-DP oracle when the policy
        is near-optimal, at the cost of the on-policy consistency guarantee.
    inner_critic : int
        Extra critic gradient steps per main step (default ``0``). The main step
        runs the expensive candidate search (for the actor's argmax target) but
        the critic only needs the *on-policy* value (``reward + V_{t+1}`` at the
        actor's own action) — one cheap evaluation, no search. Each extra step
        draws a fresh state batch and updates only the critic against that cheap
        target, so the critic can reach the <1% fit a high-dimensional value
        needs without paying the search cost on every update. The total critic
        budget becomes ``steps × (1 + inner_critic)``.
    log_every : int
        If ``> 0``, print convergence telemetry every ``log_every`` gradient
        steps (per period, per sweep): the critic RMSE in value units, the actor
        regression loss, and the live (annealed) learning rate. ``0`` (default)
        trains silently.
    n_global : int
        Globally-sampled candidate actions per state (uniform in the bounds) —
        gives the inner max global reach so it can't be trapped in a local basin.
    n_local : int
        Locally-perturbed candidates around the actor's proposal (Gaussian, std
        ``local_frac`` of the bound width) — refines toward the true argmax.
    local_frac : float
        Std of the local candidate perturbation as a fraction of each action's
        bound width.
    warm_start : bool
        Initialise period ``t``'s nets from period ``t+1``'s converged weights.
    ergodic : bool
        Train on **on-distribution** states. A first pass samples states
        uniformly over the box; subsequent passes simulate the current policy
        forward and draw training states from where it actually drives the
        system (mixed with ``ergodic_mix`` uniform draws for coverage). This is
        the standard remedy for the distribution-shift bias of fitted value
        iteration: with uniform-only sampling the critic mis-fits the region the
        policy operates in (e.g. near a capacity/kink), and the 1-step bootstrap
        compounds that error over the backward sweep — a small per-period bias
        that saturates at a sizeable gap on long horizons. On-distribution
        sampling keeps the critic accurate where it is evaluated, so the value
        stays consistent with forward simulation at any horizon. Costs roughly
        ``1 + ergodic_passes`` full backward sweeps.
    ergodic_passes : int
        Number of refinement passes after the initial uniform pass (each
        re-simulates the latest policy and re-solves). 1 is usually enough.
    ergodic_mix : float
        Fraction of each training batch still drawn uniformly during ergodic
        passes, to retain global coverage (so off-trajectory states the policy
        rarely visits are still represented).
    ergodic_sim_paths : int
        Forward paths simulated (from uniform initial states) to build the
        visited-state buffer.
    seed : int | None
        Seed for sampling and net initialisation.
    init_state : dict | None
        Initial-state distribution for the *discrete-action* ergodic refinement
        (``{state_name: scalar | tensor}``). When set, the ergodic passes roll the
        policy forward from this start instead of from uniform initial states — the
        right training distribution when the dynamics concentrate the reachable set
        far from a uniform box (e.g. an asset price started at ``S0``, where the
        forward paths occupy a thin shell a uniform box would mostly miss). Ignored
        by the continuous-action path (which always uses uniform initial states).
    """

    n_quad: int = 7
    hidden: tuple = (64, 64)
    critic_hidden: tuple | None = None
    twin_critic: bool = False
    n_critics: int = 1
    critic_quantiles: int = 1
    drop_top_atoms: int = 0
    value_expansion: int = 1
    search_expansion: int | None = None
    activation: str = "silu"
    anneal_lr: bool = True
    log_every: int = 0
    value_target: str = "on_policy"
    inner_critic: int = 0
    state_samples: int = 2048
    steps: int = 300
    lr: float = 2e-3
    n_global: int = 8
    n_local: int = 8
    local_frac: float = 0.1
    warm_start: bool = True
    ergodic: bool = True
    ergodic_passes: int = 1
    ergodic_mix: float = 0.25
    ergodic_sim_paths: int = 4096
    seed: int | None = None
    init_state: dict | None = None


def _bellman_integrand(setup: RLSetup, state: dict, action: dict, t, value_next, discount):
    """Per-shock-node integrand ``r(s,a,w,t) + β·V_{t+1}(f(s,a,w,t))``, shape
    ``S + (nq,)``, together with the quadrature weights ``[nq]``. Summing the
    weighted integrand over the last axis gives the Bellman expectation; keeping it
    un-summed gives the return distribution a quantile critic fits."""
    shape = next(iter(state.values())).shape
    nq = setup.n_q
    exp = shape + (nq,)
    state_e = {k: v.unsqueeze(-1).expand(exp) for k, v in state.items()}
    action_e = {k: v.unsqueeze(-1).expand(exp) for k, v in action.items()}
    shock_e = {
        k: v.view((1,) * len(shape) + (nq,)).expand(exp)
        for k, v in setup.shock_nodes.items()
    }

    next_state = setup.problem.transition(state_e, action_e, shock_e, t)
    if setup.reward_takes_next_state:
        r = setup.problem.reward(state_e, action_e, shock_e, t, next_state)
    else:
        r = setup.problem.reward(state_e, action_e, shock_e, t)
    r = torch.as_tensor(r, dtype=setup.dtype, device=setup.device).broadcast_to(exp)

    v_next = value_next(next_state)  # broadcastable to exp

    if callable(discount):
        d = setup.problem.discount(state_e, t)
        d = torch.as_tensor(d, dtype=setup.dtype, device=setup.device).broadcast_to(exp)
    else:
        d = discount

    return r + d * v_next, setup.shock_weights


def _bellman_value(setup: RLSetup, state: dict, action: dict, t, value_next, discount):
    """``E_w[ r(s, a, w, t) + β·V_{t+1}(f(s, a, w, t)) ]`` over the quadrature.

    ``state``/``action`` are dicts of tensors sharing an arbitrary leading shape
    ``S`` (``[B]`` for a plain batch, ``[B, M]`` for ``M`` candidate actions per
    state); returns ``S``. The shock axis is appended and summed out with the
    quadrature weights.
    """
    integrand, w = _bellman_integrand(setup, state, action, t, value_next, discount)
    w = w.view((1,) * (integrand.ndim - 1) + (setup.n_q,))
    return (integrand * w).sum(dim=-1)


def _clamp_to_range(setup: RLSetup, next_state: dict) -> dict:
    """Clamp continuous next-state coords to each state's ``[low, high]`` for the
    continuation-value lookup (mirrors the grid solver's edge-clamp). Applied only
    to the value lookup, never to the reward — a 5-arg reward still sees the true
    next state.
    """
    clamped = dict(next_state)
    for s in setup.cont_states:
        low, high = setup._cont_meta[s.name][0:2]
        x = torch.as_tensor(next_state[s.name], dtype=setup.dtype, device=setup.device)
        clamped[s.name] = torch.clamp(x, min=low, max=high)
    return clamped


def _make_value_next(setup: RLSetup, critic_next):
    """Frozen continuation-value callable ``next_state_dict -> tensor``.

    ``critic_next is None`` at the last period → terminal reward (or zeros);
    otherwise the frozen next-period critic. Both evaluate on range-clamped
    next states.
    """
    if critic_next is None:
        tr = setup.problem.terminal_reward

        def value_next(next_state):
            sample = next(iter(next_state.values()))
            if tr is None:
                return torch.zeros(sample.shape, dtype=setup.dtype, device=setup.device)
            out = tr(_clamp_to_range(setup, next_state))
            return torch.as_tensor(
                out, dtype=setup.dtype, device=setup.device,
            ).broadcast_to(sample.shape)
        return value_next

    def value_next(next_state):
        with torch.no_grad():
            return critic_next(setup.featurize(_clamp_to_range(setup, next_state))).squeeze(-1)
    return value_next


class _TruncatedEnsemble:
    """Continuation value = *truncated mean* of the pooled critic atoms (REDQ/TQC).

    Pools every atom across the ``M`` critics (each contributing its
    ``critic_quantiles`` heads), drops the ``drop_top`` largest, and averages the
    rest. The actor's argmax selects the candidate maximising ``E[r + V_{t+1}]``,
    which systematically picks actions where the critics *over-estimate*; dropping
    the top atoms removes exactly that optimism, so the bias does not compound
    backward. ``M=2, drop_top=1`` (a single scalar head each) recovers the
    twin-critic min; ``M=1, drop_top=0`` is a plain single critic. Exposes the same
    ``cont(features) -> value`` call as a scalar critic.
    """

    def __init__(self, critics, drop_top: int = 0):
        self._critics = list(critics)
        self._drop = int(drop_top)

    def __call__(self, feat):
        # Each critic returns its atoms with a trailing quantile axis [..., Q]
        # (Q=1 for a scalar critic); concatenate into a pooled [..., M*Q].
        atoms = torch.cat([c.atoms(feat) for c in self._critics], dim=-1)
        keep = max(1, atoms.shape[-1] - self._drop)
        if self._drop > 0 and keep < atoms.shape[-1]:
            atoms = atoms.sort(dim=-1).values[..., :keep]   # drop the largest `drop`
        return atoms.mean(dim=-1)


def _rollout_value(setup: RLSetup, state: dict, tau: int, h: int,
                   actors_by_t: dict, conts_by_t: dict):
    """``k``-step model-based continuation: value of ``state`` at period ``tau``,
    rolling the frozen on-policy actor forward (with exact quadrature) until the
    bootstrap period ``h``, where it falls back to the truncated-ensemble critic
    (or the terminal reward if ``h`` is the horizon end).

    Each recursion level is one exact shock expectation, so the bootstrap's share
    of the target shrinks geometrically and the per-step reward content — which is
    *exact* here, not a learned model — replaces it. Depth is ``h - tau`` (≤ k-1).
    """
    horizon_len = len(list(setup.problem.horizon))
    if tau >= h:                                            # bootstrap here
        if h >= horizon_len:                                # ... at the terminal
            tr = setup.problem.terminal_reward
            sample = next(iter(state.values()))
            if tr is None:
                return torch.zeros(sample.shape, dtype=setup.dtype, device=setup.device)
            out = tr(_clamp_to_range(setup, state))
            return torch.as_tensor(out, dtype=setup.dtype, device=setup.device).broadcast_to(sample.shape)
        return conts_by_t[h](setup.featurize(_clamp_to_range(setup, state)))   # ... at V_h
    bounds = setup.resolve_bounds(state)
    action = _actor_proposal(setup, actors_by_t[tau], state, bounds)
    inner = lambda s2: _rollout_value(setup, s2, tau + 1, h, actors_by_t, conts_by_t)  # noqa: E731
    return _bellman_value(setup, state, action, tau, inner, setup.discount)


def _make_value_next(setup: RLSetup, t: int, k: int, actors_by_t: dict, conts_by_t: dict):
    """Continuation callable for period ``t``: value of the *next* state (at
    ``t+1``), via a ``k``-step rollout that bootstraps at ``min(t+k, T)``. For
    ``k=1`` this is the ordinary one-step lookup of ``V_{t+1}`` (the terminal
    reward at the last period)."""
    horizon_len = len(list(setup.problem.horizon))
    h = min(t + k, horizon_len)

    def value_next(next_state):
        with torch.no_grad():
            return _rollout_value(setup, next_state, t + 1, h, actors_by_t, conts_by_t)
    return value_next


def _actor_proposal(setup: RLSetup, actor, state: dict, bounds: dict) -> dict:
    """Actor net → action dict, squashed into each action's resolved bounds."""
    raw = actor(setup.featurize(state))  # [..., n_cont]
    action = {}
    for i, a in enumerate(setup.cont_actions):
        lo, hi = bounds[a.name]
        action[a.name] = squash_to_bounds(raw[..., i], lo, hi)
    return action


def _candidate_actions(setup, actor, state, bounds, solver, gen) -> dict:
    """Build ``{name: [B, M]}`` candidate actions: actor proposal + global
    uniform samples + local Gaussian perturbations of the proposal."""
    B = next(iter(state.values())).shape[0]
    Kg, Kl = solver.n_global, solver.n_local
    with torch.no_grad():
        prop = _actor_proposal(setup, actor, state, bounds)
    cand = {}
    for a in setup.cont_actions:
        lo, hi = bounds[a.name]                       # [B]
        lo_c, hi_c = lo.unsqueeze(-1), hi.unsqueeze(-1)
        width = hi_c - lo_c
        p = prop[a.name].unsqueeze(-1)                # [B, 1]
        glob = lo_c + width * torch.rand(
            B, Kg, generator=gen, dtype=setup.dtype, device=setup.device,
        )
        noise = solver.local_frac * width * torch.randn(
            B, Kl, generator=gen, dtype=setup.dtype, device=setup.device,
        )
        loc = torch.minimum(torch.maximum(p + noise, lo_c), hi_c)
        cand[a.name] = torch.cat([p, glob, loc], dim=-1)  # [B, 1+Kg+Kl]
    return cand


def _collect_visited(setup: RLSetup, actors_by_t: dict, n_paths: int, gen) -> dict:
    """Simulate the policy forward from uniform initial states; return the
    states visited at each period, ``{t: {name: [n_paths] tensor}}``.

    Starting from a *broad* (uniform) initial distribution and rolling forward
    captures the region the policy actually drives the system into at each
    period — including the transient and the operating region — which is what
    the next pass trains on.
    """
    horizon = list(setup.problem.horizon)
    state = setup.sample_states(n_paths, gen)
    visited: dict = {}
    for t in horizon:
        visited[t] = {k: v.clone() for k, v in state.items()}
        with torch.no_grad():
            action = _actor_proposal(
                setup, actors_by_t[t], state, setup.resolve_bounds(state)
            )
        shock = {}
        for s in setup.problem.shocks:
            sm = s.sample(n_paths, generator=gen, dtype=setup.dtype, device=setup.device)
            if isinstance(sm, dict):
                shock.update(sm)
            else:
                shock[s.name] = sm
        nxt = setup.problem.transition(state, action, shock, t)
        new_state = {}
        for s in setup.cont_states:                 # clamp into range → valid inputs
            low, high = setup._cont_meta[s.name][0:2]
            new_state[s.name] = torch.clamp(
                torch.as_tensor(nxt[s.name], dtype=setup.dtype, device=setup.device),
                min=low, max=high,
            )
        for s in setup.disc_states:
            new_state[s.name] = torch.as_tensor(
                nxt[s.name], dtype=torch.long, device=setup.device
            )
        state = new_state
    return visited


def _collect_visited_discrete(setup: RLSetup, policy, init_state: dict,
                              n_paths: int, gen) -> dict:
    """Roll the *discrete* argmax policy forward from ``init_state`` (broadcast to
    ``n_paths``); return the states visited at each period, ``{t: {name: [n_paths]}}``.

    Unlike the continuous ``_collect_visited`` (uniform initial states), this starts
    from a *point* (or given) initial distribution, because the reachable set of a
    concentrating process — an asset price from ``S0``, say — is a thin shell that a
    uniform box would not target. The visited continuous states are clamped into
    range so they are valid net inputs."""
    horizon = list(setup.problem.horizon)
    cont_names = {s.name for s in setup.cont_states}
    state = {}
    for nm in setup.state_names:
        v = init_state[nm]
        if isinstance(v, torch.Tensor) and v.numel() == n_paths:
            t = v.clone()
        else:
            dt = setup.dtype if nm in cont_names else torch.long
            fill = float(v) if nm in cont_names else int(v)
            t = torch.full((n_paths,), fill, dtype=dt, device=setup.device)
        state[nm] = t

    visited: dict = {}
    for t in horizon:
        visited[t] = {k: v.clone() for k, v in state.items()}
        action = policy(state, t)
        shock = {}
        for s in setup.problem.shocks:
            sm = s.sample(n_paths, generator=gen, dtype=setup.dtype, device=setup.device)
            if isinstance(sm, dict):
                shock.update(sm)
            else:
                shock[s.name] = sm
        nxt = setup.problem.transition(state, action, shock, t)
        new_state = {}
        for s in setup.cont_states:
            low, high = setup._cont_meta[s.name][0:2]
            new_state[s.name] = torch.clamp(
                torch.as_tensor(nxt[s.name], dtype=setup.dtype, device=setup.device),
                min=low, max=high,
            )
        for s in setup.disc_states:
            new_state[s.name] = torch.as_tensor(
                nxt[s.name], dtype=torch.long, device=setup.device
            )
        state = new_state
    return visited


def _ergodic_sampler(setup: RLSetup, visited: dict, solver: ActorCritic, gen):
    """A ``sample_fn(t)`` that draws training states for period ``t`` from the
    visited buffer (on-distribution), mixed with ``ergodic_mix`` uniform draws
    for coverage of off-trajectory states."""
    def sample_fn(t):
        n = solver.state_samples
        n_unif = int(round(solver.ergodic_mix * n))
        n_erg = max(0, n - n_unif)
        buf = visited[t]
        m = next(iter(buf.values())).shape[0]
        unif = setup.sample_states(n_unif, gen) if n_unif > 0 else None
        idx = (torch.randint(0, m, (n_erg,), generator=gen, device=setup.device)
               if n_erg > 0 else None)
        out = {}
        for k, v in buf.items():
            parts = []
            if idx is not None:
                parts.append(v[idx])
            if unif is not None:
                parts.append(unif[k])
            out[k] = torch.cat(parts) if len(parts) > 1 else parts[0]
        return out
    return sample_fn


def _quantile_loss(pred_norm, target_norm, weights, taus):
    """Quantile (pinball) regression of predicted atoms ``pred_norm`` ``[..., Q]``
    onto the weighted return samples ``target_norm`` ``[..., nq]`` (quadrature
    weights ``weights`` ``[nq]``, quantile fractions ``taus`` ``[Q]``). Both inputs
    are in the critic's normalised space. Returns a scalar."""
    u = target_norm.unsqueeze(-2) - pred_norm.unsqueeze(-1)          # [..., Q, nq]
    tau = taus.view((1,) * (u.ndim - 2) + (taus.shape[0], 1))        # [..., Q, 1]
    rho = u * (tau - (u < 0).to(u.dtype))                            # pinball
    w = weights.view((1,) * (u.ndim - 1) + (weights.shape[0],))      # [..., 1, nq]
    return (rho * w).sum(dim=-1).mean()


def _backward_sweep(setup: RLSetup, solver: ActorCritic, gen, sample_fn, phase: str = ""):
    """One backward sweep over the horizon, drawing per-period training states
    from ``sample_fn(t)``. Returns ``(actors_by_t, critics_by_t, residual_by_t)``."""
    device, dtype = setup.device, setup.dtype
    n_cont = len(setup.cont_actions)
    horizon = list(setup.problem.horizon)

    # Resolve the critic ensemble: M critics × Q quantile atoms, dropping the top
    # `drop` pooled atoms for the continuation. `twin_critic` is the M=2, drop=1,
    # Q=1 special case (its truncated mean of two atoms is the min). `k` is the
    # value-expansion horizon.
    M = 2 if solver.twin_critic else max(1, solver.n_critics)
    drop = 1 if solver.twin_critic else solver.drop_top_atoms
    Q = max(1, solver.critic_quantiles)
    k = max(1, solver.value_expansion)               # critic-target rollout horizon
    ks = k if solver.search_expansion is None else max(1, solver.search_expansion)
    taus = (torch.arange(Q, device=device, dtype=dtype) + 0.5) / Q

    actors_by_t: dict = {}
    conts_by_t: dict = {}        # period -> _TruncatedEnsemble (continuation value)
    critics_by_t: dict = {}
    residual_by_t: dict = {}

    prev_actor = None
    prev_critics = None
    # Backward sweep: the continuation is frozen at each step (terminal reward,
    # then the truncated-ensemble critics of the already-solved later periods).
    for t in reversed(horizon):
        # Two continuations: the low-bias `k`-step rollout the CRITIC regresses onto
        # (cheap — one on-policy action), and the `ks`-step rollout the candidate
        # SEARCH uses (dear — every candidate). They coincide unless `search_expansion`
        # decouples them (set it to 1 to keep the accurate target but a cheap search).
        value_next = _make_value_next(setup, t, k, actors_by_t, conts_by_t)
        value_next_search = (
            value_next if ks == k
            else _make_value_next(setup, t, ks, actors_by_t, conts_by_t)
        )

        actor = MLP(setup.n_feat, n_cont, solver.hidden, solver.activation).to(
            device=device, dtype=dtype
        )
        # M independently-initialised critics (each with Q quantile heads). They
        # share targets/batches but differ in init, so their approximation-error
        # surfaces differ; the truncated mean of the pooled atoms cancels the
        # over-estimation the actor's argmax exploits.
        critics = [
            NormalizedCritic(
                MLP(setup.n_feat, Q, solver.critic_hidden or solver.hidden, solver.activation)
            ).to(device=device, dtype=dtype)
            for _ in range(M)
        ]
        if solver.warm_start and prev_actor is not None:
            actor.load_state_dict(prev_actor.state_dict())
            for c, pc in zip(critics, prev_critics):
                c.load_state_dict(pc.state_dict())

        opt_a = torch.optim.Adam(actor.parameters(), lr=solver.lr)
        opt_cs = [torch.optim.Adam(c.parameters(), lr=solver.lr) for c in critics]
        # Cosine-anneal the LR to a small floor over the period's steps. With a
        # fixed LR, Adam jitters around the optimum at a few-percent relative
        # error (an SGD noise floor) regardless of net size — annealing drives
        # the per-period fit far below that, which matters because the action is
        # the critic's gradient and the per-period error compounds backward.
        if solver.anneal_lr:
            sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_a, T_max=solver.steps, eta_min=solver.lr * 1e-2)
            sch_cs = [torch.optim.lr_scheduler.CosineAnnealingLR(
                oc, T_max=solver.steps * (1 + solver.inner_critic),
                eta_min=solver.lr * 1e-2) for oc in opt_cs]
        scheds = sch_cs if solver.anneal_lr else [None] * M

        def critic_loss(c, state):
            """One critic's regression loss against the on-policy target at
            ``state``: scalar MSE for Q=1, quantile/pinball for Q>1 (which needs
            the per-shock return distribution, not just its mean)."""
            feat = setup.featurize(state)
            bnd = setup.resolve_bounds(state)
            with torch.no_grad():
                a_on = _actor_proposal(setup, actor, state, bnd)
                integ, qw = _bellman_integrand(setup, state, a_on, t, value_next, setup.discount)
            if Q == 1:
                tgt = (integ * qw.view((1,) * (integ.ndim - 1) + (-1,))).sum(-1)  # E_w
                return torch.mean((c.raw(feat).squeeze(-1) - c.normalize(tgt)) ** 2)
            return _quantile_loss(c.raw(feat), c.normalize(integ), qw, taus)

        # Per-period value normalisation from one on-policy sample (the value the
        # critic actually fits — NOT the candidate max, which would bias the
        # baseline above the target and compound over the sweep).
        with torch.no_grad():
            state = sample_fn(t)
            bounds = setup.resolve_bounds(state)
            cand = _candidate_actions(setup, actor, state, bounds, solver, gen)
            state_b = {k_: v.unsqueeze(-1).expand_as(next(iter(cand.values())))
                       for k_, v in state.items()}
            vals = _bellman_value(setup, state_b, cand, t, value_next, setup.discount)
            onp = vals[..., 0] if solver.value_target == "on_policy" else vals.max(dim=-1).values
            for c in critics:
                c.set_norm(onp.mean().item(), onp.std().item())

        last_resid = float("nan")
        for step in range(solver.steps):
            state = sample_fn(t)
            bounds = setup.resolve_bounds(state)
            with torch.no_grad():
                # The actor regresses toward the best candidate the search finds (the
                # critic fits the on-policy value separately, in critic_loss, so V_θ
                # matches forward simulation). One-shot random-shooting argmax over
                # proposal + global + local candidates, scored against the (possibly
                # cheaper) `ks`-step continuation; the critic still regresses onto the
                # low-bias `k`-step target.
                cand = _candidate_actions(setup, actor, state, bounds, solver, gen)
                state_b = {k_: v.unsqueeze(-1).expand_as(next(iter(cand.values())))
                           for k_, v in state.items()}
                vals = _bellman_value(setup, state_b, cand, t, value_next_search,
                                      setup.discount)
                best_idx = vals.max(dim=-1).indices               # [B]
                a_target = {
                    k_: torch.gather(v, -1, best_idx.unsqueeze(-1)).squeeze(-1)
                    for k_, v in cand.items()
                }

            feat = setup.featurize(state)

            # Critic(s): regress onto the on-policy target (scalar or distributional).
            for c, oc in zip(critics, opt_cs):
                loss_c = critic_loss(c, state)
                oc.zero_grad(set_to_none=True)
                loss_c.backward()
                oc.step()

            # Actor: regress (in normalised action space) onto the argmax action.
            raw = actor(feat)
            loss_a = 0.0
            for j, a in enumerate(setup.cont_actions):
                lo, hi = bounds[a.name]
                tgt_norm = (a_target[a.name] - lo) / (hi - lo)
                loss_a = loss_a + torch.mean((torch.sigmoid(raw[..., j]) - tgt_norm) ** 2)
            opt_a.zero_grad(set_to_none=True)
            loss_a.backward()
            opt_a.step()
            if solver.anneal_lr:
                sch_a.step()
                for sc in sch_cs:
                    sc.step()

            # Extra cheap critic updates on fresh batches: tighten the value fit
            # (the policy's accuracy is bounded by it) and decorrelate the
            # ensemble's errors so the truncation has real bite. No candidate
            # search — just the on-policy target.
            for _ in range(solver.inner_critic):
                for c, oc, sc in zip(critics, opt_cs, scheds):
                    lc = critic_loss(c, sample_fn(t))
                    oc.zero_grad(set_to_none=True)
                    lc.backward()
                    oc.step()
                    if sc is not None:
                        sc.step()

            last_resid = float(torch.sqrt(loss_c.detach()).item() * critics[0].std.item())
            if solver.log_every and (step + 1) % solver.log_every == 0:
                lr_now = opt_cs[0].param_groups[0]["lr"]
                print(
                    f"  [{phase} t={t}] step {step + 1}/{solver.steps}  "
                    f"critic_rmse≈{last_resid:.4f}  actor_loss={float(loss_a.detach()):.4f}  "
                    f"lr={lr_now:.2e}",
                    flush=True,
                )

        # Freeze, then bias-correct the *continuation* (the truncated-ensemble
        # value) so its mean residual against the on-policy target is zero on a
        # fresh batch. A finite fit leaves a small offset that COMPOUNDS over the
        # sweep (V_t inherits V_{t+1}'s offset plus its own, saturating at
        # offset/(1-β)); recentring the truncated mean keeps the reported value
        # accurate at any horizon while preserving the per-state de-biasing of the
        # truncation. Shifting every critic's mean by the same δ shifts the
        # truncated mean by δ. Cheap: one extra batch.
        for p in actor.parameters():
            p.requires_grad_(False)
        actor.eval()
        for c in critics:
            for p in c.parameters():
                p.requires_grad_(False)
            c.eval()
        cont = _TruncatedEnsemble(critics, drop)
        with torch.no_grad():
            state = sample_fn(t)
            bounds = setup.resolve_bounds(state)
            cand = _candidate_actions(setup, actor, state, bounds, solver, gen)
            state_b = {k_: v.unsqueeze(-1).expand_as(next(iter(cand.values())))
                       for k_, v in state.items()}
            bv = _bellman_value(setup, state_b, cand, t, value_next, setup.discount)
            onp = bv[..., 0] if solver.value_target == "on_policy" else bv.max(dim=-1).values
            diff = cont(setup.featurize(state)) - onp
            last_resid = float(diff.pow(2).mean().sqrt())
            delta = diff.mean()
            for c in critics:
                c.mean -= delta

        actors_by_t[t] = actor
        conts_by_t[t] = cont
        critics_by_t[t] = cont
        residual_by_t[t] = last_resid
        prev_actor = actor
        prev_critics = critics

    return actors_by_t, critics_by_t, residual_by_t


def _backward_sweep_discrete(setup: RLSetup, solver: ActorCritic, gen, sample_fn,
                             phase: str = ""):
    """Backward sweep for *discrete* actions — fitted value iteration with the known
    model, and no actor.

    At each period the critic ``V_t`` regresses onto ``max_a E_w[r + β·V_{t+1}(f)]``
    over the **enumerated** action grid (the ∏ nᵢ category combinations), and the
    policy is the ``argmax_a`` of that same one-step lookahead. The max over the
    grid is the discrete analogue of the continuous candidate-search argmax, and it
    over-estimates for the same reason — it selects whichever action the critics are
    most optimistic about — so the continuation is the truncated-ensemble value
    (REDQ/TQC), which strips that optimism before it compounds backward. Enumeration
    is ∏ nᵢ, so this is for *low-cardinality* discrete control over an arbitrary
    (possibly high-dimensional) state. Returns ``(conts_by_t, residual_by_t,
    action_grid)``; ``action_grid`` maps each action name to its ``[n_combos]``
    category index per enumerated combination."""
    device, dtype = setup.device, setup.dtype
    horizon = list(setup.problem.horizon)
    M = 2 if solver.twin_critic else max(1, solver.n_critics)
    drop = 1 if solver.twin_critic else solver.drop_top_atoms
    if max(1, solver.critic_quantiles) > 1:
        raise NotImplementedError(
            "discrete-action ActorCritic uses scalar critics (critic_quantiles=1); "
            "the quantile critic is a continuous-action path"
        )
    if max(1, solver.value_expansion) > 1 or solver.search_expansion is not None:
        raise NotImplementedError(
            "discrete-action ActorCritic uses one-step targets (value_expansion=1); "
            "multi-step model rollout under a discrete argmax policy is a follow-up"
        )

    # Enumerate the action grid: {name: [n_combos]} long tensors — the Cartesian
    # product of each discrete action's category indices.
    axes = [torch.arange(a.n, device=device) for a in setup.disc_actions]
    mesh = torch.cartesian_prod(*axes) if len(axes) > 1 else axes[0].unsqueeze(-1)
    if mesh.ndim == 1:
        mesh = mesh.unsqueeze(-1)
    n_combos = mesh.shape[0]
    action_grid = {a.name: mesh[:, i].contiguous() for i, a in enumerate(setup.disc_actions)}

    def q_over_grid(state, value_next, t):
        """``Q(s, a) = E_w[r + β·V_{t+1}(f)]`` for every enumerated action: ``[B, A]``."""
        lead = tuple(next(iter(state.values())).shape)
        state_b = {k: v.unsqueeze(-1).expand(*lead, n_combos) for k, v in state.items()}
        act_b = {nm: g.view(*([1] * len(lead)), n_combos).expand(*lead, n_combos)
                 for nm, g in action_grid.items()}
        return _bellman_value(setup, state_b, act_b, t, value_next, setup.discount)

    conts_by_t: dict = {}
    residual_by_t: dict = {}
    prev_critics = None
    for t in reversed(horizon):
        value_next = _make_value_next(setup, t, 1, {}, conts_by_t)   # one-step V_{t+1}
        critics = [
            NormalizedCritic(
                MLP(setup.n_feat, 1, solver.critic_hidden or solver.hidden, solver.activation)
            ).to(device=device, dtype=dtype)
            for _ in range(M)
        ]
        if solver.warm_start and prev_critics is not None:
            for c, pc in zip(critics, prev_critics):
                c.load_state_dict(pc.state_dict())
        opt_cs = [torch.optim.Adam(c.parameters(), lr=solver.lr) for c in critics]
        scheds = ([torch.optim.lr_scheduler.CosineAnnealingLR(
                       oc, T_max=solver.steps * (1 + solver.inner_critic),
                       eta_min=solver.lr * 1e-2) for oc in opt_cs]
                  if solver.anneal_lr else [None] * M)

        # Per-period value normalisation from one target sample (the max_a Q the
        # critic actually fits).
        with torch.no_grad():
            tgt0 = q_over_grid(sample_fn(t), value_next, t).max(dim=-1).values
            for c in critics:
                c.set_norm(tgt0.mean().item(), tgt0.std().item())

        last_resid = float("nan")
        for step in range(solver.steps):
            state = sample_fn(t)
            with torch.no_grad():
                target = q_over_grid(state, value_next, t).max(dim=-1).values   # [B]
            feat = setup.featurize(state)
            for c, oc, sc in zip(critics, opt_cs, scheds):
                loss_c = torch.mean((c.raw(feat).squeeze(-1) - c.normalize(target)) ** 2)
                oc.zero_grad(set_to_none=True)
                loss_c.backward()
                oc.step()
                if sc is not None:
                    sc.step()

            # Extra cheap critic updates on fresh batches — tighten the fit and
            # decorrelate the ensemble so the truncation has bite (mirrors the
            # continuous sweep's inner_critic).
            for _ in range(solver.inner_critic):
                s2 = sample_fn(t)
                with torch.no_grad():
                    tg2 = q_over_grid(s2, value_next, t).max(dim=-1).values
                f2 = setup.featurize(s2)
                for c, oc, sc in zip(critics, opt_cs, scheds):
                    lc = torch.mean((c.raw(f2).squeeze(-1) - c.normalize(tg2)) ** 2)
                    oc.zero_grad(set_to_none=True)
                    lc.backward()
                    oc.step()
                    if sc is not None:
                        sc.step()

            last_resid = float(torch.sqrt(loss_c.detach()).item() * critics[0].std.item())
            if solver.log_every and (step + 1) % solver.log_every == 0:
                print(f"  [{phase} t={t}] step {step + 1}/{solver.steps}  "
                      f"critic_rmse≈{last_resid:.4f}", flush=True)

        for c in critics:
            for p in c.parameters():
                p.requires_grad_(False)
            c.eval()
        cont = _TruncatedEnsemble(critics, drop)
        # Bias-correct the truncated mean to the max_a Q target (same compounding
        # argument as the continuous sweep: a per-period offset saturates at
        # offset/(1-β) over the horizon).
        with torch.no_grad():
            state = sample_fn(t)
            tgt = q_over_grid(state, value_next, t).max(dim=-1).values
            diff = cont(setup.featurize(state)) - tgt
            last_resid = float(diff.pow(2).mean().sqrt())
            for c in critics:
                c.mean -= diff.mean()
        conts_by_t[t] = cont
        residual_by_t[t] = last_resid
        prev_critics = critics

    return conts_by_t, residual_by_t, action_grid


def _actor_critic(problem, solver: ActorCritic, *, device, dtype: torch.dtype):
    setup = build_setup(problem, solver.n_quad, device=device, dtype=dtype)

    gen = torch.Generator(device="cpu" if str(device) == "cpu" else device)
    if solver.seed is not None:
        gen.manual_seed(int(solver.seed))
        torch.manual_seed(int(solver.seed))

    if setup.disc_actions:
        # Discrete actions: fitted value iteration (no actor; policy = argmax_a Q over
        # the enumerated action grid).
        if solver.ergodic and solver.init_state is not None:
            missing = [n for n in setup.state_names if n not in solver.init_state]
            if missing:
                raise ValueError(
                    f"ActorCritic.init_state is missing a start value for state(s) "
                    f"{missing}; it must name every state ({setup.state_names})"
                )
        uniform_fn = lambda t: setup.sample_states(solver.state_samples, gen)  # noqa: E731
        if solver.log_every:
            print("[sweep: uniform (discrete actions)]", flush=True)
        conts, residual, action_grid = _backward_sweep_discrete(
            setup, solver, gen, uniform_fn, phase="uniform")
        policy = _DiscretePolicy(setup, conts, action_grid)
        # Ergodic refinement from a given start: re-train on the realised forward
        # distribution rather than a uniform box (essential when the reachable set is
        # a thin shell — see ActorCritic.init_state). Needs init_state; without it the
        # uniform sweep stands (uniform-initial rollout would not concentrate).
        if solver.ergodic and solver.init_state is not None:
            for p in range(solver.ergodic_passes):
                if solver.log_every:
                    print(f"[sweep: ergodic {p + 1}/{solver.ergodic_passes} "
                          f"(discrete actions)]", flush=True)
                visited = _collect_visited_discrete(
                    setup, policy, solver.init_state, solver.ergodic_sim_paths, gen)
                sampler = _ergodic_sampler(setup, visited, solver, gen)
                conts, residual, action_grid = _backward_sweep_discrete(
                    setup, solver, gen, sampler, phase=f"ergodic{p + 1}")
                policy = _DiscretePolicy(setup, conts, action_grid)
        value = _NeuralValue(setup, conts)
        value.residual_by_t = residual
        return policy, value

    # Pass 1: uniform state sampling over the box.
    if solver.log_every:
        print("[sweep: uniform]", flush=True)
    uniform_fn = lambda t: setup.sample_states(solver.state_samples, gen)  # noqa: E731
    actors, critics, residual = _backward_sweep(setup, solver, gen, uniform_fn, phase="uniform")

    # Ergodic refinement: re-solve on states the policy actually visits, so the
    # critic is accurate where it is evaluated and the bootstrap stops compounding
    # off-distribution error (the source of horizon-growing consistency gaps).
    if solver.ergodic:
        for p in range(solver.ergodic_passes):
            if solver.log_every:
                print(f"[sweep: ergodic {p + 1}/{solver.ergodic_passes}]", flush=True)
            visited = _collect_visited(setup, actors, solver.ergodic_sim_paths, gen)
            actors, critics, residual = _backward_sweep(
                setup, solver, gen, _ergodic_sampler(setup, visited, solver, gen),
                phase=f"ergodic{p + 1}",
            )

    policy = _NeuralPolicy(setup, actors)
    value = _NeuralValue(setup, critics)
    value.residual_by_t = residual
    return policy, value


class _NeuralPolicy:
    """``policy(state, t) -> {action: tensor}`` — same interface as the grid
    solver's ``_Policy``. Evaluates the period-``t`` actor at the query state and
    returns tensors on the query's device."""

    def __init__(self, setup: RLSetup, actors_by_t: dict):
        self._setup = setup
        self._actors_by_t = actors_by_t

    def __call__(self, state: dict, t) -> dict:
        target_device = _query_device(state, self._setup.state_names)
        s = _to_setup_device(self._setup, state)
        with torch.no_grad():
            action = _actor_proposal(
                self._setup, self._actors_by_t[t], s, self._setup.resolve_bounds(s)
            )
        return {k: v.to(target_device) for k, v in action.items()}


class _NeuralValue:
    """``value(state, t) -> tensor`` — same interface as the grid solver's
    ``_Value``. ``residual_by_t`` exposes the per-period critic RMSE in value
    units, a cheap fit-quality proxy."""

    def __init__(self, setup: RLSetup, critics_by_t: dict):
        self._setup = setup
        self._critics_by_t = critics_by_t
        self.residual_by_t: dict = {}

    def __call__(self, state: dict, t) -> torch.Tensor:
        target_device = _query_device(state, self._setup.state_names)
        s = _to_setup_device(self._setup, state)
        with torch.no_grad():
            out = self._critics_by_t[t](self._setup.featurize(s)).squeeze(-1)
        return out.to(target_device)


class _DiscretePolicy:
    """``policy(state, t) -> {action: long tensor}`` for discrete actions.

    Returns the ``argmax_a Q(s, a)`` of a one-step Bellman lookahead against the
    trained continuation critic, evaluated over the enumerated action grid — the
    same call interface as the grid solver's ``_Policy`` (each action's chosen
    category index per state). Model-based at query time: it re-evaluates the known
    transition/reward rather than storing a separate policy net."""

    def __init__(self, setup: RLSetup, conts_by_t: dict, action_grid: dict):
        self._setup = setup
        self._conts_by_t = conts_by_t
        self._action_grid = action_grid

    def __call__(self, state: dict, t) -> dict:
        setup = self._setup
        target_device = _query_device(state, setup.state_names)
        s = _to_setup_device(setup, state)
        value_next = _make_value_next(setup, t, 1, {}, self._conts_by_t)
        lead = tuple(next(iter(s.values())).shape)
        n_combos = next(iter(self._action_grid.values())).shape[0]
        state_b = {k: v.unsqueeze(-1).expand(*lead, n_combos) for k, v in s.items()}
        act_b = {nm: g.view(*([1] * len(lead)), n_combos).expand(*lead, n_combos)
                 for nm, g in self._action_grid.items()}
        with torch.no_grad():
            q = _bellman_value(setup, state_b, act_b, t, value_next, setup.discount)
        best = q.argmax(dim=-1)                                  # [*lead]
        return {nm: g[best].to(target_device) for nm, g in self._action_grid.items()}


def _query_device(state: dict, state_names: list) -> torch.device:
    v = state[state_names[0]]
    return v.device if isinstance(v, torch.Tensor) else torch.device("cpu")


def _to_setup_device(setup: RLSetup, state: dict) -> dict:
    """Coerce query tensors onto the solver's device with sensible dtypes."""
    out = {}
    cont = {s.name for s in setup.cont_states}
    for name, v in state.items():
        dt = setup.dtype if name in cont else torch.long
        out[name] = torch.as_tensor(v, dtype=dt, device=setup.device)
    return out
