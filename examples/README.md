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
- `03_american_option/` — README only; the test in `tests/test_american_option.py`
  validates against a CRR binomial reference and prints the early-exercise
  premium across spot. A notebook variant is on the roadmap.
- `04_lqg/lqg.ipynb` — 2-D LQR + Gaussian noise, validated against the
  closed-form discrete-time Riccati recursion. Heatmaps and slices of
  $V_0(x)$ and $u^*_0(x)$ side-by-side.

The notebooks are auto-generated from the `.py` source files via
`jupytext --to ipynb <file>.py`. Edit the `.py` (easier to diff,
version-controllable as text), then regenerate the `.ipynb`.
