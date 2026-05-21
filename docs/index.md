# bellgrid documentation

- [When to use bellgrid (vs. RL)](when_to_use.md) — start here; this is the positioning piece.
- [API draft](api.md) — public surface, subject to change.
- [Examples](../examples/) — Merton, Carroll/Deaton, American option (more to come; see roadmap).

## Roadmap

| Phase | Months | Deliverables |
|---|---|---|
| Core | 1–3 | API skeleton, regular + warped grids, GPU backward induction, Merton example with closed-form validation |
| Multi-dim | 4–6 | Multi-dim continuous states, shock framework, Carroll/Deaton + American option, **MVP on PyPI** |
| Mixed | 7–9 | Discrete-continuous mixed states, account/tax machinery, lifecycle decumulation + Roth ladder |
| Launch | 10–12 | Docs site, benchmark suite (vs. Tasmanian, vs. hand-rolled, vs. RL), JOSS submission, public launch |
