# Study 02: Fixed-source slowing down with real data

Status: part 1 (elastic only) done. Part 2 (a high-energy source, with inelastic scattering) waits
for milestone M3.

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
- **Speed:** the anisotropic part is a direct sum over each window, so time grows with the square of
  the node count. Heavy takes 7742 s at 544k nodes with two terms, about 800 times the 10 s target.
  M9 has to replace it.
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

## Decision

Elastic slowing down with real data meets the gate's accuracy, at about 35k (light) and 270k (heavy)
nodes for 0.2% rates. Cost is now the main open risk, and the residual gate needs flux-driven node
insertion. Part 2 follows M3.
