# Study 02: Fixed-source slowing down with real data

Status: done. Part 1 is elastic scattering only; part 2 starts the same media at 14 MeV, so that
inelastic scattering and (n,xn) join it (milestone M3).

## Question

Does the 0D energy solve reproduce OpenMC's spectrum and capture rates with real cross sections and
angular distributions?

## Method

A point source below every inelastic threshold slows down in an infinite medium at 294 K, so that
elastic scattering and capture are the only physics, with no edited data. Below 20 eV neutrons are cut
off in both codes, which stays above OpenMC's free gas threshold (400 kT, about 10 eV).

| Case | Nuclides, atoms/b-cm | Source |
| --- | --- | --- |
| light | Na23 0.02, Fe56 0.02, O16 0.01 | 400 keV |
| heavy | U238 0.02, O16 0.04 | 40 keV |

The twins are in [`benchmarks/02-elastic-fixed-source`](../../benchmarks/02-elastic-fixed-source),
with probability tables and fission neutrons off: 2e7 histories for light and 4e7 for heavy. MT 5 is
treated as removal, and reaches at most 1.1e-6 of the total here. U238 fission is removal in both
codes.

```bash
cd studies/02-fixed-source/scripts
```

```bash
pixi run python run_case.py light 1e-3 4
```

`subdivide` (the last argument) splits every panel of the 0.1% thinned grid, after forward peaks are
refined. `results/` holds the sweep's output, gitignored.

## Result

| Case | Subdivide | Nodes | Solve time | Residual / source | Capture vs twin | Flux rms z |
| --- | --- | --- | --- | --- | --- | --- |
| light | 1 | 17,957 | 25 s | -3.6e-3 | Fe56 +0.345%, Na23 +0.346% | 5.96 |
| light | 2 | 35,518 | 98 s | -9.6e-4 | Fe56 +0.084%, Na23 +0.095% | 1.85 |
| light | 4 | 70,640 | 368 s | -2.4e-4 | Fe56 +0.013%, Na23 +0.027% | 1.09 |
| light | 8 | 140,884 | 1435 s | -5.9e-5 | Fe56 -0.004%, Na23 +0.010% | 1.03 |
| heavy | 1 | 136,195 | 496 s | -2.1e-3 | U238 +0.211% | 3.23 |
| heavy | 2 | 272,086 | 1970 s | -4.9e-4 | U238 +0.056% | 1.29 |
| heavy | 4 | 543,868 | 7742 s | -1.2e-4 | U238 +0.019% | 1.03 |

Twin uncertainty is 0.020% on Fe56 capture, 0.008% on Na23 and 0.013% on U238. Flux bins have a median
uncertainty of 0.05% (light) and 0.12% (heavy). Times are from seven cases run at once on 32 cores,
so they are slower than a lone run: 16 s against 25 s for light at subdivide 1.

- **Accuracy:** both cases converge to OpenMC. On the finest grids every capture rate is within 1.5 sd
  of the twin, and flux bins have an rms z of 1.03.
- **Order:** residuals fall by 3.8-4.2 per subdivision, which is second order, and capture errors
  track them. The 0.2% rate target needs subdivide 2.
- **Residual gate:** decision 005's 1e-4 is met only at light subdivide 8. Heavy reaches -1.2e-4 at
  544k nodes.
- **Why the residual tracks rates:** on a panel where Σ and F slope in opposite directions, the lin-lin
  product bulges above the chord through the nodes. The running integral carries that excess
  downstream as a gain. The M2 review measured the flux error against the cumulative bulge above
  each energy: +3.71e-3 and +3.57e-3 at 25 eV in the isotropic light case. 200 of 18,000 panels hold
  60% of it, so inserting nodes by bulge should cost far less than uniform subdivision (study 04).
- **Speed:** the anisotropic part was a direct sum over each window, so time grew with the square of
  the node count: 7742 s for heavy at 544k nodes, about 800 times the 10 s target. Decision 006
  replaced it with running moments, which take 1.5 to 2.6 s there and give the same captures to
  7e-8. The times in the table are the direct sum's, which stays as the oracle.
- **Largest single bin:** the worst heavy flux bin differs by 3.4%, at z under 4.7, in a low-flux bin
  with a large tally error.

## Errors this study caught

Unit tests with synthetic data passed through all of these:

- Normalizing the anisotropic kernel with f varying in E' created neutrons: 17% on the light case. Its
  mean over incoming energy is truly nonzero, so it is now measured with f frozen at the node.
- That correction was also applied to windows crossing the anisotropy cutoff, where the frozen mean is
  cut short and truly nonzero. This left a floor near +0.03% (M2 review).
- Subdividing before refining forward peaks kept the capped panels wide, so rates converged at first
  order (M2 review).
- A cutoff at "the next table down" ran the direct sum across the whole range, since tables are sparse.
- Numba's cache kept a stale kernel across files; see decision 005.
- An earlier heavy result of +0.025% came from the normalization bug, and it was a coincidence.

## Part 2: from 14 MeV

### Method

The same two media, light-fast and heavy-fast, with a 14 MeV point source. Inelastic and (n,xn)
scattering go through the coarse transfer of decision 007 at its default spacing of 0.00125. Twins
run 8e7 histories each, with the part 1 settings.

```bash
pixi run python run_case.py heavy-fast 1e-3 2
```

### Result

| Case | Subdivide | Nodes | Kernels | Solve | Residual / source | Capture vs twin | Flux rms z |
| --- | --- | --- | --- | --- | --- | --- | --- |
| light-fast | 1 | 58,252 | 379 s | 40 s | -3.2e-3 | Fe56 +0.338%, Na23 +0.315% | 10.0 |
| light-fast | 2 | 105,199 | 378 s | 122 s | -1.0e-3 | Fe56 +0.104%, Na23 +0.096% | 3.19 |
| light-fast | 4 | 199,093 | 372 s | 415 s | -3.0e-4 | Fe56 +0.027%, Na23 +0.022% | 1.30 |
| heavy-fast | 1 | 195,979 | 784 s | 56 s | -1.2e-3 | U238 +0.148% | 2.87 |
| heavy-fast | 2 | 380,653 | 780 s | 180 s | -3.1e-4 | U238 +0.052% | 1.38 |
| heavy-fast | 4 | 750,001 | 775 s | 750 s | -7.7e-5 | U238 +0.028% | 1.16 |

Twin uncertainty is 0.011% on Fe56 capture, 0.009% on Na23 and 0.016% on U238. Flux bins have a
median uncertainty of 0.03% (light) and 0.08% (heavy). Four cases ran at once, and every solve
converged in 2 lagged sweeps.

- **Accuracy:** both cases converge to OpenMC at second order, as in part 1. The leading error is
  still elastic slowing down at low energy: apart from heavy-fast's 12.99 MeV bin, the worst bins at
  subdivide 1 and 2 lie below 20 keV. The 0.2% rate target needs subdivide 2.
- **Coarse spacing:** first order, and it was the largest error above 1 MeV at the old default of
  0.0025: -0.9% (z -6.6) in the 7.68 MeV bin of heavy-fast. Decision 007 has the convergence table.
- **Open bins:** heavy-fast's 3.28 MeV (+0.34%) and 12.99 MeV (-0.30%) bins stay at z near 3.4 for
  every setting tried. See decision 007 for what is ruled out. Against the original 2e7-history twin,
  12.99 MeV looked twice as far off (-0.68%, z -4.6). A second seed and a fourfold rerun showed that
  twin 0.37% high there, 2.5 of its sd, so the twins were rerun at 8e7.
- **Speed:** kernels cost 6 to 13 minutes and are cached by grid, and solves take 40 to 750 s. Both
  are far from the 10 s target. Decision 006 found that above 100 keV its quadrature fallback carries
  most elastic segments, which is the likely cost here; it has not been profiled for these cases.

### Errors this part caught

- Point masses for strata quantized the source once hats were narrower than their spacing. Strata
  are now lab-energy segments.
- Rows built only from reactions above threshold at the hat's node lost 7e-5 of the source in hats
  straddling a threshold. They now weight reactions by hat integrals at centroids.
- Spacing strata by the narrowest hat made kernels 58 times slower, from a 1e-6 sliver between two
  thresholds.

## Decision

Slowing down with real data, from 14 MeV through inelastic scattering and (n,xn), meets the gate's
accuracy at subdivide 2: rates within 0.11% for light-fast and 0.06% for heavy-fast, at about 100k
and 380k nodes. Cost is the main open risk. Kernel setup, the solve above 100 keV, and dense
coarse matrices all go to M9, and the residual gate needs flux-driven node insertion (study 04).
