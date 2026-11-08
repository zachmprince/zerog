# 009: Thermal scattering as coarse rows, with the upscatter block solved by GMRES

Date: 2026-11-08

## Context

Below a few hundred kT, neutrons gain energy as well as lose it, so the downward march of
decisions 005 to 007 is no longer a solve. OpenMC v0.16.0 treats this range by sampling:

- `sample_target_velocity` takes the target at rest at E >= 400 kT (`free_gas_threshold`)
  unless awr <= 1, and otherwise samples it with the constant cross section approximation; the
  resonance scattering methods are off by default.
- An S(alpha, beta) table replaces a bound nuclide's elastic cross section at E <= its
  `energy_max` (`Material::calculate_neutron_xs`, `Nuclide::calculate_sab_xs`), and
  `IncoherentInelasticAE::sample_params` samples the nearer incoming table, scaled or shifted to E.

The plan expected the block to hold U238's low resonances, so it can't be a small dense matrix,
and expected hydrogen to couple it throughout.

## Decision

- **Expected kernels.** Free gas is the analytic kernel of the constant cross section sampler with
  isotropic centre-of-mass scattering, normalized so the collision rate is the tabulated elastic
  cross section, evaluated through erfcx so it never overflows. S(alpha, beta) is the nearer
  table's lin-lin pdf, bin by bin, mapped as OpenMC maps it. `zerog.energy.thermal` cites each
  sampler, and tests compare both with OpenMC's sampling.
- **Hydrogen free gas to 4e4 kT.** OpenMC never takes a target of awr <= 1 at rest. Free gas
  smears the top of hydrogen's kernel by about sqrt(4 kT E), wider than U238's low resonances, and
  in study 03's PWR mix the top moved k by -36, -4.8 and -0.2 pcm from 400 to 4e3, 4e4 and 4e5 kT.
  So `LIGHT_FREE_GAS_THRESHOLD = 4e4`, and target at rest above it.
- **Rows on the inelastic coarse hats.** Each hat's row is its single-collision rows averaged by
  Gauss quadrature on each half-hat, weighted by hat and cross section, split at each top and each
  S(alpha, beta) midpoint. Where a hat also has inelastic rows (U235's MT 5 reaches 1e-5 eV), the
  two mix by hat-integrated emission. Rows plus loss sum to one.
- **Thermal hats.** Below the reach of thermal rows, hats are the smaller of a fifth of the
  heaviest free gas kernel's width, sqrt(4 kT / A E), and 0.05 / (1 + E / kT), between
  `COARSE_DU` and 0.01. Restriction and prolongation each smooth a kernel by a hat, which shifts
  a Maxwellian equilibrium by about du^2 (E / kT)^2 / 6 times the scattering to leakage ratio;
  the flux term bounds that. Thresholds are kept.
- **The upscatter block by GMRES.** The block runs from the first hat whose row reaches above its
  upper neighbour to the bottom. Each full sweep updates hats above it as before. Hats in it take
  the exact solution of their lagged problem given the rest: with the sweep's map r = G x + b,
  (I - G_bb) dx = r_b - x_b, by GMRES through `zerog.linalg`. The operator is a march from the
  block's reach with only the block's emission, which is exact because every running sum is
  linear in the F above it. So GMRES is preconditioned by the downward sweep. Sweeps end when one
  reproduces its input to 1e-12, as before; problems without upscatter take the old path.

## Consequences

- A free gas scatterer holds a Maxwellian to 1e-4 at its peak (to 6e-4 for A = 236), falling by
  4 when every hat is halved. The block solve matches 2000 plain sweeps to 1e-9.
- Study 03's PWR mix meets the k and rate gates at every subdivision: +22, +13 and +10 pcm
  against a 6.7 pcm twin, rates within 0.025% except O16 capture (0.07%, 1.2 sd). Halving the
  thermal hats changed nothing measurable.
- Each eigenvalue solve took 56 to 66 block marches. Hydrogen's free gas rows to 2 keV hold
  about 3.5e7 values (280 MB) and the block reaches 2.2 keV, which M9 should profile.
- Not covered: coherent and incoherent elastic thermal tables (graphite, ZrH, Be) are refused;
  thermal rows are order 0 only, so P1 for bound hydrogen waits for M8; bound oxygen and uranium
  tables are not used here, in either code.
