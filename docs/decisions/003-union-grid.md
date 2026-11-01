# 003: Union grid size and Phase 0 close

Date: 2026-11-01

## Context

[Study 01](../../studies/01-union-grid/README.md) measured the thinned union grid. A PWR pin needs
315,802 points at 1% and 566,878 at 0.1% for all reactions at one temperature per material, and
442,276 and 667,594 across a span of tabulated temperatures. Grids that test only the major curves
are smaller, but do not bound inelastic reactions. Composition-specific macroscopic grids are 58,589
and 165,509 points. The toy used 36,000-92,000.

## Decision

- Plan the energy solve for roughly 300,000-700,000 points on an all-reactions grid.
- Do not fix a tolerance or grid type yet. Phase 1 should choose them from reaction-rate error,
  and should test whether reduced grids are enough.
- Keep Phase 0 open until the scattering, thermal and probability-table readers are done, even though
  the plan's stated gate needs only cross sections, nu, chi and N_E. Phase 1 depends on them.

## Consequences

- Estimate energy-solve cost, and the effective cross-section cost in Phase 3's gate, with this N_E
  and not the toy's. These figures are lower estimates for a real core: S(α,β), burned fuel and a
  continuous temperature field are not included.
- OpenMC twins should run at tabulated temperatures until a slowing-down comparison measures the
  effect of its statistical interpolation. Its first-flight rate moves from the arithmetic towards
  the harmonic mean of the bounds as flights thicken, so linear curves are not its reaction rate.
  Phase 3 still needs its own temperature treatment, because the mix is not Doppler broadening.
- Density-weighted thinning and lumping trace isotopes are worth testing in Phase 1. Both would tie
  the grid to a composition range, so burnup needs care.
- Effective densities in PGD can be negative, which removes the relative bound on a sum. Phase 1
  should measure the error directly.
