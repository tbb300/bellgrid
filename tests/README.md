# bellgrid tests

Planned layout for v0.1:

- `test_merton.py` — Merton (1969) closed-form validation. The library's correctness contract: optimal consumption and portfolio share within interpolation tolerance of the analytical solution. Lands the day `solve()` works and gates everything else.
- `test_carroll_deaton.py` — reproduces the kinked consumption function near the borrowing constraint with the marginal propensity to consume matching literature values.
- `test_american_option.py` — Longstaff-Schwartz benchmark to basis-point tolerance; regime-switching variant validates `MarkovChain`.
- `test_shocks.py` — unit tests for shock quadrature: Gauss-Hermite nodes/weights for `Normal` and `Lognormal`, Cholesky factorization for `MultivariateNormal`, jump mixture for `Jump`.
- `test_grids.py` — interpolation accuracy on `RegularGrid` and `WarpedGrid`; bounds and ranges.
- `test_problem.py` — `Problem` construction validation: no name collisions across states/actions/shocks, bound expressions reference existing states, shock dimensions match `transition`'s expected keys.
- `test_simulate.py` — simulator/solver agreement: forward-simulated discounted-reward expectation matches the solver's value function at the initial state.
