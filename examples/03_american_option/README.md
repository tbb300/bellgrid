# Example 03: American option exercise

Same backward-induction machinery, totally different audience. An American option holder chooses each period whether to exercise (terminal payoff) or continue (continuation value). The state is the underlying price; the action is binary (exercise / hold).

**Goal:** reproduce a published benchmark (e.g., Longstaff-Schwartz test cases) within a small basis-point tolerance, and demonstrate that the bellgrid policy outperforms Longstaff-Schwartz on path-dependent tail scenarios.

**Stretch variant — regime-switching volatility:** extend the example with a `MarkovChain` state for low/high vol regimes. This exercises the mixed discrete-continuous state machinery and is the canonical end-to-end validation of `MarkovChain` for v0.1.

**Why this example exists:** credibility with the quant audience. The same library that solves retirement decumulation also prices American options. The cross-domain flex is part of the pitch.
