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
        Hidden-layer widths for both the actor and critic MLPs.
    activation : str
        ``"silu"`` (default), ``"tanh"``, ``"relu"``, or ``"gelu"``.
    state_samples : int
        States drawn per gradient step (resampled each step for coverage).
    steps : int
        Gradient steps per period (each updates both nets against the
        candidate-max target).
    lr : float
        Adam learning rate for both nets.
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
    """

    n_quad: int = 7
    hidden: tuple = (64, 64)
    activation: str = "silu"
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


def _bellman_value(setup: RLSetup, state: dict, action: dict, t, value_next, discount):
    """``E_w[ r(s, a, w, t) + β·V_{t+1}(f(s, a, w, t)) ]`` over the quadrature.

    ``state``/``action`` are dicts of tensors sharing an arbitrary leading shape
    ``S`` (``[B]`` for a plain batch, ``[B, M]`` for ``M`` candidate actions per
    state); returns ``S``. The shock axis is appended and summed out with the
    quadrature weights.
    """
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

    integrand = r + d * v_next
    w = setup.shock_weights.view((1,) * len(shape) + (nq,))
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


def _backward_sweep(setup: RLSetup, solver: ActorCritic, gen, sample_fn):
    """One backward sweep over the horizon, drawing per-period training states
    from ``sample_fn(t)``. Returns ``(actors_by_t, critics_by_t, residual_by_t)``."""
    problem = setup.problem
    device, dtype = setup.device, setup.dtype
    n_cont = len(setup.cont_actions)
    horizon = list(problem.horizon)

    actors_by_t: dict = {}
    critics_by_t: dict = {}
    residual_by_t: dict = {}

    prev_actor = prev_critic = None
    # Backward sweep: V_{t+1} frozen at each step (terminal reward, then critics).
    for i, t in enumerate(reversed(horizon)):
        value_next = _make_value_next(setup, None if i == 0 else prev_critic)

        actor = MLP(setup.n_feat, n_cont, solver.hidden, solver.activation).to(
            device=device, dtype=dtype
        )
        critic = NormalizedCritic(
            MLP(setup.n_feat, 1, solver.hidden, solver.activation)
        ).to(device=device, dtype=dtype)
        if solver.warm_start and prev_actor is not None:
            actor.load_state_dict(prev_actor.state_dict())
            critic.load_state_dict(prev_critic.state_dict())

        opt_a = torch.optim.Adam(actor.parameters(), lr=solver.lr)
        opt_c = torch.optim.Adam(critic.parameters(), lr=solver.lr)

        # Set the per-period value normalisation from one target sample. The
        # critic fits the value at the actor's *own* action (candidate 0), so
        # normalise from that, not from the candidate max.
        with torch.no_grad():
            state = sample_fn(t)
            bounds = setup.resolve_bounds(state)
            cand = _candidate_actions(setup, actor, state, bounds, solver, gen)
            state_b = {k: v.unsqueeze(-1).expand_as(next(iter(cand.values())))
                       for k, v in state.items()}
            vals = _bellman_value(setup, state_b, cand, t, value_next, setup.discount)
            # Normalise from the *on-policy* value the critic actually fits
            # (vals[..,0], the value at the actor's own action) — NOT the
            # candidate max. Setting the baseline mean above the fit target makes
            # the net under-correct the offset and systematically overestimate;
            # that per-period bias is tiny but COMPOUNDS over the backward sweep
            # (≈5% at T=10 → ≈15% at T=60). Aligning the normalisation with the
            # target keeps the per-period error zero-mean, so it no longer grows
            # with the horizon.
            onp = vals[..., 0]
            critic.set_norm(onp.mean().item(), onp.std().item())

        last_resid = float("nan")
        for _ in range(solver.steps):
            state = sample_fn(t)
            bounds = setup.resolve_bounds(state)
            with torch.no_grad():
                # Candidate 0 is the actor's own proposal; the rest are global
                # + local perturbations. We evaluate all against the frozen
                # continuation, then split the targets two ways:
                #   - actor improves toward the candidate-max action (argmax),
                #   - critic evaluates the actor's *current* action (vals[..,0]).
                # This on-policy critic target is the key to consistency: V_θ
                # then equals the value of the policy actually run, so it matches
                # forward simulation by construction (rather than the optimistic
                # candidate-max, which the approximate actor can't fully attain).
                cand = _candidate_actions(setup, actor, state, bounds, solver, gen)
                state_b = {k: v.unsqueeze(-1).expand_as(next(iter(cand.values())))
                           for k, v in state.items()}
                vals = _bellman_value(setup, state_b, cand, t, value_next, setup.discount)
                actor_val = vals[..., 0]                          # value at π_φ(s)
                best_idx = vals.max(dim=-1).indices               # [B]
                a_target = {
                    k: torch.gather(v, -1, best_idx.unsqueeze(-1)).squeeze(-1)
                    for k, v in cand.items()
                }

            feat = setup.featurize(state)

            # Critic: regress (normalised) value onto the actor's-action value.
            pred_raw = critic.raw(feat)
            loss_c = torch.mean((pred_raw - critic.normalize(actor_val)) ** 2)
            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_c.step()

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

            last_resid = float(torch.sqrt(loss_c.detach()).item() * critic.std.item())

        # Bias-correct the critic: recenter so its *mean* residual against the
        # on-policy target is zero on a fresh batch. A finite MSE fit can leave a
        # small systematic offset, and while that offset is negligible per period
        # it COMPOUNDS over the backward sweep — V_{t} inherits V_{t+1}'s offset
        # plus its own — saturating at offset/(1-β) (≈11% at our scale). Zeroing
        # the per-period mean residual keeps the error zero-mean, so the reported
        # value stays accurate regardless of horizon. Cheap: one extra batch.
        with torch.no_grad():
            state = sample_fn(t)
            bounds = setup.resolve_bounds(state)
            cand = _candidate_actions(setup, actor, state, bounds, solver, gen)
            state_b = {k: v.unsqueeze(-1).expand_as(next(iter(cand.values())))
                       for k, v in state.items()}
            onp = _bellman_value(setup, state_b, cand, t, value_next, setup.discount)[..., 0]
            offset = (critic(setup.featurize(state)) - onp).mean()
            critic.mean -= offset

        for p in actor.parameters():
            p.requires_grad_(False)
        for p in critic.parameters():
            p.requires_grad_(False)
        actor.eval()
        critic.eval()

        actors_by_t[t] = actor
        critics_by_t[t] = critic
        residual_by_t[t] = last_resid
        prev_actor, prev_critic = actor, critic

    return actors_by_t, critics_by_t, residual_by_t


def _actor_critic(problem, solver: ActorCritic, *, device, dtype: torch.dtype):
    setup = build_setup(problem, solver.n_quad, device=device, dtype=dtype)

    gen = torch.Generator(device="cpu" if str(device) == "cpu" else device)
    if solver.seed is not None:
        gen.manual_seed(int(solver.seed))
        torch.manual_seed(int(solver.seed))

    # Pass 1: uniform state sampling over the box.
    uniform_fn = lambda t: setup.sample_states(solver.state_samples, gen)  # noqa: E731
    actors, critics, residual = _backward_sweep(setup, solver, gen, uniform_fn)

    # Ergodic refinement: re-solve on states the policy actually visits, so the
    # critic is accurate where it is evaluated and the bootstrap stops compounding
    # off-distribution error (the source of horizon-growing consistency gaps).
    if solver.ergodic:
        for _ in range(solver.ergodic_passes):
            visited = _collect_visited(setup, actors, solver.ergodic_sim_paths, gen)
            actors, critics, residual = _backward_sweep(
                setup, solver, gen, _ergodic_sampler(setup, visited, solver, gen)
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
