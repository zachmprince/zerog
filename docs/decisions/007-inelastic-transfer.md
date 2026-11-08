# 007: Inelastic and (n,xn) scattering by a coarse transfer matrix

Date: 2026-11-05

## Context

Above the first inelastic threshold, scattering is no longer a window under the incoming energy.
Levels, continua, Kalbach-Mann, correlated and N-body laws, and (n,xn), carry neutrons across
many lethargy units, with secondary distributions tabulated at a few dozen incoming energies. The
expected kernel has to be OpenMC's, including what its sampling adds to the data, since the twins
define the target (plan, "Match OpenMC's sampling").

## Decision

- **Expected kernels by stratified sampling.** Each OpenMC v0.16.0 sampler is transcribed with its
  random numbers replaced by quadrature: law choice by cumulative applicability, table i or i + 1
  with weight r, unit-base scaling, the ContinuousTabular clamp, the correlated law's nearer angle
  table, and centre-of-mass to lab with the target at rest. Outgoing energies are stratified in
  equal lethargy (at most 0.002) and cosines in 32 equal-probability strata, each with its exact
  weight. The docstring of `zerog.energy.inelastic` cites each sampler.
- **Strata are lab-energy segments, not points.** A cosine stratum maps to an interval in lab
  energy, since lab energy is linear in mu, and its weight is spread uniformly in lethargy over it.
  Point masses quantized the source once hats were narrower than the point spacing.
- **Restrict, transfer, prolong.** Flux is restricted to hats on a coarse lethargy grid of width
  `COARSE_DU = 0.00125`, with thresholds as required nodes, by exact sums of fine panel rates. A
  coarse matrix maps hats to hats, and its output is interpolated to fine nodes. Each row weights
  reactions by hat-integrated yield × σ, evaluated at the hat's centroid, so thresholds inside a
  hat cost nothing. A reaction with no emission at its centroid (below its threshold, in a hat
  that straddles it) is taken at the hat's top, so every row plus its loss sums to one.
- **Lagged sweeps.** Inelastic emission into a node can come from above or below it within a hat,
  so the march takes it from the previous sweep, repeated until F changes by less than 1e-12.
  Study 02's fast cases take 2 sweeps at the default spacing, and 6 at twice it.
- **Cached per grid.** Kernels depend only on the terms, grids and settings, so `build_inelastic`
  stores them as `.npz` under a key of those, this module's and its data readers' source, and the
  data file's stat. `build_operators` refuses kernels whose coarse grid this fine grid doesn't hold.
- **Stratum spacing follows the median hat**, not the narrowest. Close thresholds (U238 and O16, 6
  eV apart at 6.5 MeV) leave a 1e-6 sliver hat, and spacing by it made heavy-fast's kernels take
  5945 s instead of 102 s. Results moved by 5e-5 at the median node and not at all in any flux bin.

## Oracle

`benchmarks/03-secondary-laws` tallies OpenMC's first-collision outgoing spectrum by MT for
O16, Na23, Cr52, Fe56, U238 and Pu241 at 14 MeV, Fe56 at 2 MeV and U238 at 1 MeV. 14 MeV is a table
energy for U238 and Fe56, so the M3 review added cases between tables, where table mixing and
unit-base scaling act: U238 at 13.7 and 13.25 MeV, Fe56 at 13.7 MeV and O16 at 13.3 MeV. Every
reaction passes a χ² test with Poisson variance at p > 1e-4 (385 reactions, so near 4% false
alarms for the lot). Levels need 4096 cosine strata to resolve their edges at the tally's 50 bins
per lethargy. Elastic first collisions, outside this oracle, match direct integration of
openmc.data's angle tables to 6e-5 of a neutron per bin, at and between tables.

## Consequences

Study 02 part 2, truncated heavy-fast (1 to 14 MeV), against 1e8 pooled twin histories:

| COARSE_DU | Cosine strata | 7.68 MeV bin | rms z, 105 bins | Kernels | Solve |
| --- | --- | --- | --- | --- | --- |
| 0.005 | 32 | -2.27% (z -16) | 2.27 | 53 s | 215 s |
| 0.0025 | 32 | -0.91% (z -6.6) | 1.31 | 101 s | 164 s |
| 0.00125 | 32 | -0.25% | 1.07 | 304 s | 60 s |
| 0.000625 | 32 | -0.22% | 1.06 | 735 s | 61 s |
| 0.00125 | 128 | -0.15% | 1.03 | 1086 s | 59 s |

- The error is first order in the coarse spacing where forward elastic scattering smooths little,
  and 0.00125 is converged to well within twin noise. Finer hats also cut lagged sweeps.
- 128 strata move a few bins by up to 0.1%, inside the 0.14% twin noise, at 3.6 times the build.
  32 stays the default; study 04 should revisit it against the gate's spectrum test.
- Two bins stay off at every setting: 3.28 MeV (+0.34%, z 3.5) and 12.99 MeV (-0.30%, z -3.3).
  Ruled out for them: fine-grid and forward-peak resolution, the anisotropy cutoff, coarse spacing,
  cosine strata, and first-collision kernels at and between tables, elastic and inelastic. 71% of
  the 12.99 MeV flux is from second and later collisions. Both are below the reaction-rate gate,
  and stay open.

## Limits passed on

- Kernel setup dominates a fresh run (780 s for heavy-fast down to 20 eV) and is rebuilt whenever
  the fine grid changes, as it may during PGD. About two thirds is `law_points` and `deposit`, which
  re-copy whole law tables for every hat. M9 target.
- `term_transfer` holds a dense coarse × coarse matrix per term before compressing it: 0.9 GB for
  heavy-fast, about 4 GB for a PWR grid down to 1e-5 eV. The coarse grid also runs to the bottom of
  the fine grid, though every row below the lowest threshold is empty. Fix both before M5.
- `NBodyLaw` and `MaxwellSpectrum` raise `NotImplementedError`. H2's (n,2n) is N-body above 3.3
  MeV, so natural hydrogen fails until it is added.
- Inelastic transfer enters only the zeroth Legendre order. 0D needs nothing more, but the PGD
  spatial coupling of M8 will need its higher moments.
