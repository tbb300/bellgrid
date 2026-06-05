"""Pathwise (analytic) policy-gradient solver — `PolicyGradient`.

Trains a policy by **backpropagating the expected return straight through the
differentiable model**: roll the policy forward under reparameterized (sampled)
shocks using the `Problem`'s own `transition`/`reward`, accumulate the discounted
return, and take its gradient w.r.t. the policy parameters. This is the pathwise /
"stochastic value gradient" estimator (Heess et al. 2015) — *not* score-function /
REINFORCE — so it is low-variance, and crucially it has **no learned critic and no
bootstrap**, which is exactly where the `ActorCritic`'s overestimation (and its
whole drawer of ensemble/value-expansion knobs) came from. With a known
differentiable model and a short horizon, that machinery is simply unnecessary:
the model *is* the value gradient.

It shares the actor net + `Problem`→setup machinery with `ActorCritic` (so it sits
behind the same spec and returns the same `(policy, value)` callables) but drops
the critic entirely. The action squash handles state-dependent bounds, so it solves
the *constrained* problem; `value(state, t)` is an honest Monte-Carlo rollout of the
trained policy (value = what the policy actually earns — no bootstrap to over-state).

Scope (v1): `ContinuousState` + `ContinuousAction` only (the model must be
differentiable — discrete-state/Markov dynamics break the pathwise gradient),
finite horizon, scalar discount. Full-horizon backprop suits short horizons; very
long or chaotic rollouts may need gradient truncation + a learned terminal value
(SHAC), a planned follow-up. Discrete/Markov states, discrete actions, callable
discount, and the infinite-horizon case raise `NotImplementedError` pointing at the
grid / `ActorCritic` solvers.
"""

from dataclasses import dataclass

import torch

from ..problem import ContinuousAction, ContinuousState
from ._setup import build_setup
from .nets import MLP
from .solver import _NeuralPolicy, _actor_proposal


@dataclass(frozen=True)
class PolicyGradient:
    """Pathwise policy-gradient (differentiable-model) solver.

    Attributes
    ----------
    hidden : tuple
        Per-period policy MLP hidden sizes (default ``(64, 64)``).
    activation : str
        ``"silu"`` (default), ``"tanh"``, ``"relu"``, or ``"gelu"``.
    steps : int
        Gradient steps (default 400). Each step samples a fresh batch of initial
        states, rolls the policy forward, and backprops the return.
    lr : float
        Adam learning rate (default 3e-3).
    anneal_lr : bool
        Cosine-anneal the LR to a small floor over ``steps`` (default True) — the
        per-period policy fit benefits from it just as the actor-critic did.
    batch : int
        Rollout paths per step (default 2048). The pathwise gradient averages over
        these, so larger batches lower its (already low) variance.
    value_samples : int
        Monte-Carlo paths used by ``value(state, t)`` (default 4096).
    warm_start : bool
        Initialise each period's net from the next period's trained weights
        before the joint optimisation (default True) — a mild curriculum.
    seed : int | None
        Seed for sampling and net initialisation.
    log_every : int
        Print the mean return every ``log_every`` steps (0 = silent).
    """

    hidden: tuple = (64, 64)
    activation: str = "silu"
    steps: int = 400
    lr: float = 3e-3
    anneal_lr: bool = True
    batch: int = 2048
    value_samples: int = 4096
    warm_start: bool = True
    seed: int | None = None
    log_every: int = 0


def _sample_shocks(setup, n: int, gen) -> dict:
    """Reparameterized shock draws ``{name: [n] tensor}`` (no grad — fixed noise
    per path; the pathwise gradient flows through the action, not the noise)."""
    shock: dict = {}
    for s in setup.problem.shocks:
        sm = s.sample(n, generator=gen, dtype=setup.dtype, device=setup.device)
        if isinstance(sm, dict):
            shock.update(sm)
        else:
            shock[s.name] = sm
    return shock


def _rollout_return(setup, actors_by_t, state: dict, gen, discount) -> torch.Tensor:
    """Discounted return ``Σ_t β^t r_t + β^T·terminal`` of the policy from ``state``,
    one reparameterized shock path per batch element. Differentiable w.r.t. the
    policy parameters (the gradient threads the whole rollout)."""
    horizon = list(setup.problem.horizon)
    B = next(iter(state.values())).shape[0]
    total = torch.zeros(B, dtype=setup.dtype, device=setup.device)
    disc = 1.0
    for t in horizon:
        bounds = setup.resolve_bounds(state)
        action = _actor_proposal(setup, actors_by_t[t], state, bounds)
        shock = _sample_shocks(setup, B, gen)
        nxt = setup.problem.transition(state, action, shock, t)
        if setup.reward_takes_next_state:
            r = setup.problem.reward(state, action, shock, t, nxt)
        else:
            r = setup.problem.reward(state, action, shock, t)
        total = total + disc * torch.as_tensor(
            r, dtype=setup.dtype, device=setup.device
        ).broadcast_to((B,))
        disc = disc * discount
        state = {k: torch.as_tensor(v, dtype=setup.dtype, device=setup.device)
                 for k, v in nxt.items()}
    if setup.problem.terminal_reward is not None:
        g = setup.problem.terminal_reward(state)
        total = total + disc * torch.as_tensor(
            g, dtype=setup.dtype, device=setup.device
        ).broadcast_to((B,))
    return total


def _policy_gradient(problem, solver: PolicyGradient, *, device, dtype: torch.dtype):
    if any(not isinstance(s, ContinuousState) for s in problem.states):
        raise NotImplementedError(
            "PolicyGradient backpropagates through the model, so it needs all "
            "ContinuousState states (discrete/Markov dynamics break the pathwise "
            "gradient); use the grid solver, or ActorCritic for a DiscreteState"
        )
    if any(not isinstance(a, ContinuousAction) for a in problem.actions):
        raise NotImplementedError(
            "PolicyGradient supports only ContinuousAction; use the grid solver"
        )
    if callable(problem.discount):
        raise NotImplementedError(
            "PolicyGradient v1 supports a scalar discount only; use the grid solver"
        )

    setup = build_setup(problem, 1, device=device, dtype=dtype)  # n_quad unused (sampled rollout)
    horizon = list(problem.horizon)
    n_cont = len(setup.cont_actions)
    discount = float(problem.discount)

    gen = torch.Generator(device="cpu" if str(device) == "cpu" else device)
    if solver.seed is not None:
        gen.manual_seed(int(solver.seed))
        torch.manual_seed(int(solver.seed))

    # Per-period policy nets, trained jointly by full-horizon backprop. Warm-start
    # period t from t+1's weights (built back-to-front) as a mild curriculum.
    actors_by_t: dict = {}
    prev = None
    for t in reversed(horizon):
        net = MLP(setup.n_feat, n_cont, solver.hidden, solver.activation).to(
            device=device, dtype=dtype)
        if solver.warm_start and prev is not None:
            net.load_state_dict(prev.state_dict())
        actors_by_t[t] = net
        prev = net

    params = [p for t in horizon for p in actors_by_t[t].parameters()]
    opt = torch.optim.Adam(params, lr=solver.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=solver.steps, eta_min=solver.lr * 1e-2) if solver.anneal_lr else None)

    for step in range(solver.steps):
        state = setup.sample_states(solver.batch, gen)
        ret = _rollout_return(setup, actors_by_t, state, gen, discount)
        loss = -ret.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if solver.log_every and (step + 1) % solver.log_every == 0:
            print(f"  [pgrad] step {step + 1}/{solver.steps}  "
                  f"mean return {float(ret.mean().detach()):.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    for t in horizon:
        actors_by_t[t].eval()
        for p in actors_by_t[t].parameters():
            p.requires_grad_(False)

    return (
        _NeuralPolicy(setup, actors_by_t),
        _PGValue(setup, actors_by_t, discount, solver.value_samples, gen),
    )


class _PGValue:
    """``value(state, t) -> tensor``: an honest Monte-Carlo rollout of the trained
    policy from the query state — the value *is* what the policy earns, with no
    bootstrap or learned critic to over-state it."""

    def __init__(self, setup, actors_by_t, discount, n_samples, gen):
        self._setup = setup
        self._actors_by_t = actors_by_t
        self._discount = discount
        self._n = int(n_samples)
        self._gen = gen

    def __call__(self, state: dict, t) -> torch.Tensor:
        setup = self._setup
        names = setup.state_names
        q = state[names[0]]
        out_dev = q.device if isinstance(q, torch.Tensor) else torch.device("cpu")
        # query states -> [B]; roll n_samples paths each, average the return
        base = {n: torch.as_tensor(state[n], dtype=setup.dtype, device=setup.device)
                .reshape(-1) for n in names}
        B = base[names[0]].shape[0]
        rep = {n: v.repeat_interleave(self._n) for n, v in base.items()}  # [B*n]
        horizon = [h for h in setup.problem.horizon if h >= t]
        with torch.no_grad():
            total = torch.zeros(B * self._n, dtype=setup.dtype, device=setup.device)
            st = rep
            disc = 1.0
            for tt in horizon:
                bounds = setup.resolve_bounds(st)
                action = _actor_proposal(setup, self._actors_by_t[tt], st, bounds)
                shock = _sample_shocks(setup, B * self._n, self._gen)
                nxt = setup.problem.transition(st, action, shock, tt)
                if setup.reward_takes_next_state:
                    r = setup.problem.reward(st, action, shock, tt, nxt)
                else:
                    r = setup.problem.reward(st, action, shock, tt)
                total = total + disc * torch.as_tensor(
                    r, dtype=setup.dtype, device=setup.device).broadcast_to((B * self._n,))
                disc = disc * self._discount
                st = {k: torch.as_tensor(v, dtype=setup.dtype, device=setup.device)
                      for k, v in nxt.items()}
            if setup.problem.terminal_reward is not None:
                total = total + disc * torch.as_tensor(
                    setup.problem.terminal_reward(st),
                    dtype=setup.dtype, device=setup.device).broadcast_to((B * self._n,))
        return total.reshape(B, self._n).mean(dim=1).to(out_dev)
