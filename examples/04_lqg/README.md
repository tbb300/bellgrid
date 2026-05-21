# Example 04: 2-D linear-quadratic-Gaussian control

The canonical multi-dim DP benchmark. Discrete-time LQR with a 2-D
continuous state, a scalar action, and a single Normal noise channel.
Has a closed-form Riccati solution against which bellgrid can be
validated to arbitrary precision.

**Goal:** match the discrete-time Riccati `(P_t, K_t, c_t)` recursion
for the value function and optimal policy across the 2-D state grid,
within the multilinear-on-quadratic interpolation tolerance.

**What this example demonstrates in bellgrid:**

- Multiple `ContinuousState`s composed in the same `Problem`.
- Joint state grid (cartesian product, K-D multilinear interpolation).
- Quadratic reward composing cleanly with the solver.
- A canonical analytical benchmark where every result has a closed form.
