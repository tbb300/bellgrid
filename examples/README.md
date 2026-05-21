# bellgrid examples

Install with `uv sync --extra examples` (or `pip install -e ".[examples]"`)
from the repo root to pull in `matplotlib`, `jupyter`, and `jupytext`
alongside the core dependencies. Open the `.ipynb` files in JupyterLab,
or run the source `.py` files directly.

- `01_merton/merton.ipynb` — log-utility Merton, validated against the
  $V(w) = A + B\log w$ closed form. Convergence sweep included.
- `02_carroll_deaton/carroll_deaton.ipynb` — CRRA lifecycle savings with
  a borrowing constraint; shows the kinked consumption function and the
  buffer-stock target.
- `03_american_option/american_option.ipynb` — American put under
  risk-neutral GBM, validated against a high-resolution CRR binomial
  tree and the Black-Scholes European put. Shows the early-exercise
  premium, the optimal exercise boundary $S^*(t)$ across time, and a
  spot sweep table.
- `04_lqg/lqg.ipynb` — 2-D LQR + Gaussian noise, validated against the
  closed-form discrete-time Riccati recursion. Heatmaps and slices of
  $V_0(x)$ and $u^*_0(x)$ side-by-side.
- `05_two_asset_merton/two_asset_merton.ipynb` — log-utility
  consumption-portfolio choice between two risky assets with **correlated
  lognormal returns**. Exercises `MultivariateNormal` as a 2-D shock.
  Validated against the numerical FOC for the optimal portfolio share;
  sweeps correlation to show the diversification effect (π* runs from
  ~0.75 at ρ = -0.8 to a corner at 1.0 once ρ ≳ 0.4).
- `06_regime_switching_option/regime_switching_option.ipynb` —
  American put under **regime-switching volatility**: a 2-state
  `MarkovChain` flips between calm (σ=0.15) and turbulent (σ=0.40)
  regimes. The solver advances the regime via its transition matrix
  during backward induction. Value functions and exercise boundaries
  shown per regime, sandwiched between three constant-vol references
  (σ_low, σ_high, σ_stationary_avg); the turbulent exercise boundary
  sits well below the calm one (high vol → hold longer).
- `07_retirement_decision/retirement_decision.ipynb` — lifecycle
  consumption-savings problem where the agent also chooses **when to
  retire**. Uses `DiscreteState` for the irreversible
  working/retired phase (user-controlled dynamics: once retired, stay
  retired) and a leisure bonus in the utility function. Shows the
  retirement boundary $w^*(t)$ falling from ~31 at age 0 to ~6 near
  the end of the horizon, plus a 500-path forward simulation of the
  accumulation → retirement → decumulation pattern.
- `08_jump_diffusion_option/jump_diffusion_option.ipynb` — American
  put under **Merton (1976) jump-diffusion**: standard GBM diffusion
  plus rare downward jumps. First multi-shock example: pairs a
  `Normal` diffusion shock with a `Jump` (Bernoulli-approximated
  Poisson with Normal log-magnitudes). European value validated
  against the Merton 1976 series-expansion closed form (agreement
  within ~1e-3). For the American case, shows the jump premium and
  the lower exercise boundary that jumps induce (downward-biased
  jumps → more reason to hold).

The notebooks are auto-generated from the `.py` source files via
`jupytext --to ipynb <file>.py`. Edit the `.py` (easier to diff,
version-controllable as text), then regenerate the `.ipynb`.
