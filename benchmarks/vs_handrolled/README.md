# bellgrid vs. hand-rolled backward induction

A comparison against a backward-induction loop written directly in PyTorch (or NumPy) — no library, just user code on the same hardware. The point is to bound bellgrid's abstraction overhead: solving the same problem through the library should be within a small constant factor of the hand-rolled implementation.

If bellgrid is appreciably slower, the library is paying for ergonomics with performance and that's a bug worth flagging. If bellgrid is appreciably faster, the GPU vectorization is doing its job vs. a naive loop.
