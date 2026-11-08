# 010: The energy grid is the all-reactions union at 0.1%, subdivided twice

Date: 2026-11-08

## Context

Decision 003 sized the thinned union grid but left its tolerance and type to Phase 1, to be chosen
from reaction-rate error, with reduced grids, density-weighted grids and negative effective
densities to be tested. Study 04 ran study 03's two mixes against their twins on each.

## Decision

- **Type:** the union of each nuclide's grid thinned on every non-redundant reaction, as before.
  A grid thinned on each nuclide's major curves is under 0.3% smaller, so it saves nothing. A
  macroscopic grid is half the size, but it holds one composition, misses minor nuclides'
  resonances (Cr52 capture 0.22% off at 0.1%), and at equal size is about as accurate as the
  union thinned three times looser.
- **Tolerance:** 0.1%, with a 1e-8 b floor. Both mixes are then within 13 pcm and 0.015% of
  their twins, and 0.03% changes k by at most 1.4 pcm. 0.3% also meets the gates (+24 and
  +20 pcm, rates within 0.033%) with about 20% fewer nodes, and is the fallback if M9 needs it.
  1% fails the SFR's k by 22 pcm.
- **Subdivision:** each thinned panel split in two (`subdivide=2`), as studies 02 and 03 found:
  the thinned grid resolves cross sections, not F.
- **No special treatment for negative densities.** On per-nuclide grids they added no measurable
  error.

## Consequences

- The study 03 mixes take 837k (SFR) and 509k (PWR) nodes, above the plan's 500k speed point.
  M9's 10 s target is for these grids.
- nu-fission and inelastic emission, which no grid is thinned on, are within 2.2e-5 and 4.2e-5 of
  OpenMC's sampling on every grid tested, mostly from interpolation between the library's points.
- PGD may use any effective densities on this grid; a composition-specific grid would tie it to the
  physical mix.
- Not covered: temperatures other than one per mix, higher Legendre orders (M8), and probability
  table bands (M7). The PWR's +13 pcm floor is not from the grid.
