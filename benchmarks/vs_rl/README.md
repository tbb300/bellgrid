# bellgrid vs. RL baselines

A comparison against modern model-free RL (PPO, SAC) on problems in bellgrid's sweet spot — 1–6 continuous state dimensions plus discrete states, where a transition model is available. The point is two-fold:

1. **Correctness.** Backward induction is exact (up to interpolation error) across the full state space. RL is approximate and on-distribution. We measure the gap in tail regions an agent rarely visits during training.
2. **Wall-clock to a usable policy.** For problems where bellgrid applies, the time-to-policy is often dramatically shorter than the corresponding RL training run.

Where RL wins (high-dim state, no model, learned representations), bellgrid is simply the wrong tool — see `docs/when_to_use.md`.
