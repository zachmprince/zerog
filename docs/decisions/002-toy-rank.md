# 002: Toy rank study supports continuing

Date: 2026-11-01

## Context

[Study 00](../../studies/00-toy-rank/README.md) measured the space-energy rank of a pointwise
slowing-down flux on a 1D slab with synthetic resonances. It needed 17-30 modes for 0.1% worst
per-cell capture accuracy, against 36,000-92,000 energy points. This is an SVD lower bound.

## Decision

Continue with the separated representation. Leave band splitting open: more bands needed more
pooled modes in every case, but band-local modes span fewer energy points and total unknowns were
not measured.

## Consequences

- The result does not show PGD converges at this rank; a later phase has to measure that directly.
- Real data, fission, anisotropic and inelastic scattering and S(α,β) remain untested here.
- The two-pin case needed the most modes, but it changes several things at once, so later rank
  studies should vary them separately. The band comparison covers the scalar flux only.
