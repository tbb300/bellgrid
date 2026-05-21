# bellgrid examples

Install with `pip install -e ".[examples]"` from the repo root to pull in `matplotlib` and `jupyter` alongside the core dependencies. Each example directory has its own README with the goal, setup, and what it validates.

- `01_merton/` — closed-form Merton (1969) validation. The library's correctness contract.
- `02_carroll_deaton/` — lifecycle consumption-savings with stochastic labor income.
- `03_american_option/` — American option pricing with Longstaff-Schwartz benchmark; regime-switching variant exercises `MarkovChain`.
