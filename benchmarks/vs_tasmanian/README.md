# bellgrid vs. Tasmanian

Tasmanian is a sparse-grid math library. bellgrid is an end-to-end DP solver on multilinear regular and warped grids. This benchmark compares bellgrid's full solver against a hand-rolled backward-induction loop on top of Tasmanian — a different layer of abstraction, but the closest competing option for someone solving DP on a grid in the 1–6 continuous-dim range.

The point is end-to-end performance and ergonomics on the problems bellgrid targets, not head-to-head on sparse-grid math (which bellgrid deliberately doesn't do).
