"""iLQG / DDP solver — finite-horizon trajectory optimization for a known,
differentiable model.

For a ``Problem`` whose states and actions are all continuous and whose horizon
is finite, this runs the iterative Linear-Quadratic-Gaussian / Differential
Dynamic Programming sweep: roll the nominal trajectory forward, build the *local
quadratic model* of the cost-to-go from autograd Jacobians/Hessians of the
``Problem``'s own ``transition``/``reward``, solve the resulting LQ subproblem for
an affine feedback law, and line-search forward. We optimise in cost units
(``cost = -reward``, since ``bellgrid`` maximises reward) and report
``value = -cost``.

Because the per-step model is *exact* on a linear-quadratic problem, on an LQ
``Problem`` this converges in a single Newton step to the matrix-Riccati solution
— gains, trajectory, and value to machine precision. That is how it certifies
itself (see ``examples``), a far tighter bar than the simulation-consistency the
neural solver reports.

Scope (v1): ``ContinuousState`` + ``ContinuousAction`` only, finite horizon,
scalar discount, additive (certainty-equivalent) shocks; action bounds are *not*
enforced (the returned law is the unconstrained optimum — control-limited DDP is
a planned follow-up). Discrete/Markov states, discrete actions, callable discount,
and the infinite-horizon case raise ``NotImplementedError`` pointing at the grid
solver. Unlike the grid/neural solvers — which return a *global* policy over the
whole state box — iLQG returns a time-varying affine feedback that is globally
optimal for LQ problems and a *local* law around the optimised trajectory
otherwise.
"""

import inspect
from dataclasses import dataclass

import torch

from ..problem import ContinuousAction, ContinuousState, Problem


@dataclass(frozen=True)
class iLQG:
    """iterative Linear-Quadratic-Gaussian (DDP-family) trajectory optimiser.

    Attributes
    ----------
    x0 : dict | None
        Operating point ``{state_name: value}`` to optimise from. For an LQ
        problem the affine feedback is globally optimal regardless of ``x0``;
        otherwise the returned law is the local feedback around the optimised
        trajectory from this point. ``None`` (default) ⇒ the centre of each
        ``ContinuousState``'s ``range``.
    max_iter : int
        Maximum DDP iterations (default 100). LQ problems converge in 1.
    tol : float
        Convergence threshold on the per-iteration cost reduction (default 1e-10).
    reg_init, reg_min, reg_max, reg_factor : float
        Levenberg–Marquardt regularisation of ``Q_uu`` (added to its diagonal):
        increased when the backward pass is non-convex or the forward pass fails,
        decreased on success. Defaults give exact behaviour on LQ (reg → reg_min).
    n_quad : int
        Gauss–Hermite nodes used to compute each shock's mean and variance
        (default 7). Only the mean (nominal dynamics) and variance (the value's
        noise correction) are used — iLQG is certainty-equivalent in v1.
    verbose : bool
        Print per-iteration cost / regularisation.
    """

    x0: dict | None = None
    max_iter: int = 100
    tol: float = 1e-10
    reg_init: float = 1e-6
    reg_min: float = 1e-9
    reg_max: float = 1e10
    reg_factor: float = 10.0
    n_quad: int = 7
    verbose: bool = False


# Backtracking line-search step sizes (1.1^{-k^2}, the Tassa et al. schedule).
_ALPHAS = tuple(float(1.1 ** (-(k ** 2))) for k in range(11))


def _ilqg(problem: Problem, solver: iLQG, *, device, dtype: torch.dtype):
    # ---- scope validation (mirror the grid/neural NotImplementedError style) ----
    if problem.horizon is None:
        raise NotImplementedError(
            "iLQG is finite-horizon only; use PolicyIteration (grid) for the "
            "infinite-horizon stationary case"
        )
    if any(not isinstance(s, ContinuousState) for s in problem.states):
        raise NotImplementedError(
            "iLQG supports only ContinuousState states (it linearises a "
            "continuous state vector); model a discrete regime with the grid "
            "solver, or the neural solver for a DiscreteState"
        )
    if any(not isinstance(a, ContinuousAction) for a in problem.actions):
        raise NotImplementedError(
            "iLQG supports only ContinuousAction actions; use the grid solver "
            "for DiscreteAction"
        )
    if not problem.actions:
        raise ValueError("Problem has no actions")
    if callable(problem.discount):
        raise NotImplementedError(
            "iLQG v1 supports a scalar discount only; use the grid solver for a "
            "state-dependent (callable) discount"
        )

    dev = torch.device(device)
    sname = [s.name for s in problem.states]
    aname = [a.name for a in problem.actions]
    nx, nu = len(sname), len(aname)
    horizon = list(problem.horizon)
    T = len(horizon)
    beta = float(problem.discount)

    # reward arity (4-arg or 5-arg with next_state), same rule as the RL setup
    n_pos = len([
        p for p in inspect.signature(problem.reward).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ])
    if n_pos not in (4, 5):
        raise ValueError(
            f"reward must take 4 or 5 positional args; got {n_pos}"
        )
    reward_takes_next = (n_pos == 5)

    # ---- shock mean + (diagonal) variance from quadrature ----
    shock_names: list = []
    shock_mean: dict = {}
    shock_var: dict = {}
    for sh in problem.shocks:
        raw, w = sh.nodes_and_weights(solver.n_quad, dtype=dtype, device=dev)
        nodes = raw if isinstance(raw, dict) else {sh.name: raw}
        for nm, nd in nodes.items():
            m = (w * nd).sum()
            shock_names.append(nm)
            shock_mean[nm] = m
            shock_var[nm] = (w * nd ** 2).sum() - m ** 2
    nw = len(shock_names)
    sigma_diag = torch.tensor([float(shock_var[nm]) for nm in shock_names],
                              dtype=dtype, device=dev)  # [nw]

    def mean_shock_dict():
        return {nm: shock_mean[nm] for nm in shock_names}

    # ---- dict <-> vector ----
    def to_state(x):
        return {sname[i]: x[i] for i in range(nx)}

    def to_action(u):
        return {aname[i]: u[i] for i in range(nu)}

    def vec_state(d):
        return torch.stack([
            torch.as_tensor(d[n], dtype=dtype, device=dev).reshape(()) for n in sname
        ])

    # ---- model functions of the stacked z = [x; u] at the mean shock ----
    def dynamics(z, t):
        x, u = z[:nx], z[nx:]
        nxt = problem.transition(to_state(x), to_action(u), mean_shock_dict(), t)
        return vec_state(nxt)

    def stage_cost(z, t):  # cost = -reward
        x, u = z[:nx], z[nx:]
        st, ac, sh = to_state(x), to_action(u), mean_shock_dict()
        if reward_takes_next:
            nxt = problem.transition(st, ac, sh, t)
            r = problem.reward(st, ac, sh, t, nxt)
        else:
            r = problem.reward(st, ac, sh, t)
        return -torch.as_tensor(r, dtype=dtype, device=dev).reshape(())

    def terminal_cost(x):  # cost = -terminal_reward
        if problem.terminal_reward is None:
            return torch.zeros((), dtype=dtype, device=dev)
        r = problem.terminal_reward(to_state(x))
        return -torch.as_tensor(r, dtype=dtype, device=dev).reshape(())

    def noise_input(x, u, t):  # C = d f / d w at the mean shock, [nx, nw]
        if nw == 0:
            return torch.zeros((nx, 0), dtype=dtype, device=dev)
        w0 = torch.stack([shock_mean[nm].reshape(()) for nm in shock_names])

        def f_of_w(w):
            sh = {nm: w[i] for i, nm in enumerate(shock_names)}
            return vec_state(problem.transition(to_state(x), to_action(u), sh, t))

        return torch.autograd.functional.jacobian(f_of_w, w0)

    # ---- nominal initialisation ----
    if solver.x0 is not None:
        x0 = vec_state({k: torch.as_tensor(v, dtype=dtype, device=dev)
                        for k, v in solver.x0.items()})
    else:
        x0 = torch.tensor([0.5 * (float(s.range[0]) + float(s.range[1]))
                           for s in problem.states], dtype=dtype, device=dev)
    U = torch.zeros(T, nu, dtype=dtype, device=dev)

    def forward(x0, U_nom, k_ff, K_fb, X_nom, alpha):
        """Roll the closed-loop law forward; return (X, U, total cost)."""
        X = [x0]
        U_new = []
        cost = torch.zeros((), dtype=dtype, device=dev)
        x = x0
        for i, t in enumerate(horizon):
            if k_ff is None:
                u = U_nom[i]
            else:
                u = U_nom[i] + alpha * k_ff[i] + K_fb[i] @ (x - X_nom[i])
            z = torch.cat([x, u])
            cost = cost + (beta ** i) * stage_cost(z, t)
            x = dynamics(z, t)
            X.append(x)
            U_new.append(u)
        cost = cost + (beta ** T) * terminal_cost(x)
        return torch.stack(X), torch.stack(U_new), cost

    # initial nominal rollout
    X, U, J = forward(x0, U, None, None, None, 1.0)

    reg = solver.reg_init
    eye_u = torch.eye(nu, dtype=dtype, device=dev)

    # per-period stores for the returned policy/value
    K_by_t = [None] * T
    Vx_by_t = [None] * T
    Vxx_by_t = [None] * T
    a_by_t = [None] * T          # cost-to-go constant at the nominal (incl. noise)

    for it in range(solver.max_iter):
        # ---- backward pass: linearise along (X, U), build the LQ subproblem ----
        Vx = torch.autograd.functional.jacobian(terminal_cost, X[T])          # [nx]
        Vxx = torch.autograd.functional.hessian(terminal_cost, X[T])          # [nx,nx]
        Vxx = 0.5 * (Vxx + Vxx.T)
        k_ff = [None] * T
        K_fb = [None] * T
        Vx_t = [None] * T
        Vxx_t = [None] * T
        dV1 = torch.zeros((), dtype=dtype, device=dev)
        dV2 = torch.zeros((), dtype=dtype, device=dev)
        ok = True
        for i in reversed(range(T)):
            t = horizon[i]
            z = torch.cat([X[i], U[i]])
            fz = torch.autograd.functional.jacobian(lambda zz: dynamics(zz, t), z)  # [nx, nx+nu]
            fx, fu = fz[:, :nx], fz[:, nx:]
            lz = torch.autograd.functional.jacobian(lambda zz: stage_cost(zz, t), z)  # [nx+nu]
            lzz = torch.autograd.functional.hessian(lambda zz: stage_cost(zz, t), z)  # [nx+nu, nx+nu]
            lzz = 0.5 * (lzz + lzz.T)
            lx, lu = lz[:nx], lz[nx:]
            lxx, luu, lux = lzz[:nx, :nx], lzz[nx:, nx:], lzz[nx:, :nx]

            # Bellman quadratic of the cost-to-go (discount β on the continuation;
            # Gauss–Newton / iLQR — drop the f_xx term, exact for LQ).
            Qx = lx + beta * fx.T @ Vx
            Qu = lu + beta * fu.T @ Vx
            Qxx = lxx + beta * fx.T @ Vxx @ fx
            Quu = luu + beta * fu.T @ Vxx @ fu
            Qux = lux + beta * fu.T @ Vxx @ fx

            Quu_reg = Quu + reg * eye_u
            # require positive-definite Q_uu (convex in u) for the min
            evals = torch.linalg.eigvalsh(Quu_reg)
            if bool((evals <= 0).any()):
                ok = False
                break
            Quu_inv = torch.linalg.inv(Quu_reg)
            k = -(Quu_inv @ Qu)
            K = -(Quu_inv @ Qux)

            dV1 = dV1 + (beta ** i) * (k @ Qu)
            dV2 = dV2 + (beta ** i) * 0.5 * (k @ Quu @ k)

            # value backup (standard DDP)
            Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            Vxx = 0.5 * (Vxx + Vxx.T)

            k_ff[i], K_fb[i] = k, K
            Vx_t[i], Vxx_t[i] = Vx, Vxx

        if not ok:
            reg = min(reg * solver.reg_factor, solver.reg_max)
            if reg >= solver.reg_max:
                break
            continue

        # ---- forward line search ----
        accepted = False
        for alpha in _ALPHAS:
            Xn, Un, Jn = forward(x0, U, k_ff, K_fb, X, alpha)
            expected = -(alpha * dV1 + alpha ** 2 * dV2)   # predicted cost decrease
            actual = J - Jn
            if expected > 0 and (actual / expected) > 1e-4:
                accepted = True
                break
        if not accepted:
            reg = min(reg * solver.reg_factor, solver.reg_max)
            if solver.verbose:
                print(f"[iLQG it {it}] no improvement; reg->{reg:.1e}")
            if reg >= solver.reg_max:
                break
            continue

        dJ = float(J - Jn)
        X, U, J = Xn, Un, Jn
        K_by_t, Vx_by_t, Vxx_by_t = K_fb, Vx_t, Vxx_t
        reg = max(reg / solver.reg_factor, solver.reg_min)
        if solver.verbose:
            print(f"[iLQG it {it}] cost={float(J):.6e}  dJ={dJ:.2e}  reg={reg:.1e}")
        if 0 <= dJ < solver.tol:
            break

    # ---- value constants with the additive-noise correction (backward) ----
    # a_t = l(x̄_t,ū_t) + β [ a_{t+1} + ½ tr(C_t' Vxx_{t+1} C_t Σ) ]
    a_next = terminal_cost(X[T])
    for i in reversed(range(T)):
        t = horizon[i]
        z = torch.cat([X[i], U[i]])
        Vxx_next = Vxx_by_t[i + 1] if i + 1 < T else \
            torch.autograd.functional.hessian(terminal_cost, X[T])
        Vxx_next = 0.5 * (Vxx_next + Vxx_next.T)
        C = noise_input(X[i], U[i], t)
        noise = 0.5 * torch.einsum("ij,jk,kl,l->", C.T, Vxx_next, C, sigma_diag) \
            if nw > 0 else torch.zeros((), dtype=dtype, device=dev)
        a_next = stage_cost(z, t) + beta * (a_next + noise)
        a_by_t[i] = a_next

    return (
        _ILQGPolicy(sname, aname, X.detach(), U.detach(),
                    [k.detach() for k in K_by_t], dev, dtype),
        _ILQGValue(sname, X.detach(),
                   [g.detach() for g in Vx_by_t], [H.detach() for H in Vxx_by_t],
                   [float(a) for a in a_by_t], dev, dtype),
    )


def _stack_state(state: dict, names, dtype, dev):
    """{name: tensor[B] or scalar} -> [..., nx] (a new trailing state axis)."""
    cols = [torch.as_tensor(state[n], dtype=dtype, device=dev) for n in names]
    return torch.stack(cols, dim=-1)


class _ILQGPolicy:
    """``policy(state, t) -> {action: tensor}``: the affine feedback law
    ``u = ū_t + K_t (x - x̄_t)`` at the converged nominal (α = 1)."""

    def __init__(self, sname, aname, X, U, K, dev, dtype):
        self._sname, self._aname = sname, aname
        self._X, self._U, self._K = X, U, K
        self._dev, self._dtype = dev, dtype

    def __call__(self, state: dict, t) -> dict:
        i = t  # horizon values are the period index 0..T-1
        x = _stack_state(state, self._sname, self._dtype, self._dev)  # [..., nx]
        out_dev = (state[self._sname[0]].device
                   if isinstance(state[self._sname[0]], torch.Tensor) else self._dev)
        dx = x - self._X[i]
        u = self._U[i] + dx @ self._K[i].T            # [..., nu]
        return {self._aname[j]: u[..., j].to(out_dev) for j in range(len(self._aname))}


class _ILQGValue:
    """``value(state, t) -> tensor``: the local quadratic of the cost-to-go,
    negated to a reward-value ``V_t(x) = -(a_t + g_t·δx + ½ δx' H_t δx)``."""

    def __init__(self, sname, X, Vx, Vxx, a, dev, dtype):
        self._sname, self._X = sname, X
        self._Vx, self._Vxx, self._a = Vx, Vxx, a
        self._dev, self._dtype = dev, dtype

    def __call__(self, state: dict, t) -> torch.Tensor:
        i = t
        x = _stack_state(state, self._sname, self._dtype, self._dev)
        out_dev = (state[self._sname[0]].device
                   if isinstance(state[self._sname[0]], torch.Tensor) else self._dev)
        dx = x - self._X[i]
        quad = 0.5 * torch.einsum("...i,ij,...j->...", dx, self._Vxx[i], dx)
        lin = dx @ self._Vx[i]
        cost = self._a[i] + lin + quad
        return (-cost).to(out_dev)
