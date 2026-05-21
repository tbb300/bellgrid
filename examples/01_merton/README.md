# Example 01: Merton consumption-portfolio

The unit test of correctness. Merton (1969) has a closed-form solution for optimal consumption and portfolio share under power utility and geometric Brownian motion asset returns. bellgrid's first example reproduces it numerically and overlays the closed-form curve.

**Goal:** the consumption and portfolio policies from `solve()` should overlay the analytical Merton policies to within interpolation error on the figure.

The CRRA reward function and any economics-specific helpers live in this example's own code — they are not core bellgrid primitives. The library sees a `reward` callable, nothing more.

**What to seed from rl-inv:** the power-utility reward and the GBM-style return generator. Strip out the multi-account / mortality / regime machinery — Merton is single-asset, infinite-horizon, no labor income, no constraints. The whole notebook should run in under 30 seconds on a laptop CPU.
