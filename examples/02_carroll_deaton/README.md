# Example 02: Carroll/Deaton lifecycle consumption-savings

The canonical multi-period reference. A finite-lived household with power utility, stochastic labor income, and a single risk-free asset. Proves the multi-period machinery and the warped-wealth grid handling of the borrowing constraint.

**Goal:** reproduce the kinked consumption function near the borrowing constraint, with the marginal propensity to consume matching the literature.

The reward function (power utility) and the wage process live in this example's own code. So does the mortality treatment (a state/age-dependent `discount` callable) if and when we extend the example beyond a fixed horizon. None of these are bellgrid primitives — they demonstrate how the library composes with domain code.

**Seeds from rl-inv:** wage process, power-utility reward, asinh-wealth grid. Drop the account-type machinery — Carroll/Deaton is a single liquid wealth state.
