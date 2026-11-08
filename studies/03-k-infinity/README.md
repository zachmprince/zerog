# Study 03: k-infinity of real mixes

Status: part 1 (fast mix, milestone M4) done; part 2 (PWR mix at 600 K) waits for the thermal block
of M5. Part 1's number is provisional, since it cuts neutrons off at 50 eV in both codes.

## Question

Does power iteration on the 0D energy solve, with fission spectra collapsed from the evaluated
data, reproduce OpenMC's k-infinity within 50 pcm and its nuclide reaction rates within 0.2%?

## Method

The mix is a homogenized sodium-cooled fast reactor cell at 900 K, defined in
[`scripts/cases.py`](scripts/cases.py): 35% oxide MOX (U0.8 Pu0.2)O2, 40% sodium and 25% steel,
12 nuclides from U235 to Ni58. Below 50 eV neutrons are killed in both codes. That is above
OpenMC's free gas threshold (400 kT, 31 eV at 900 K), so everything this study runs is target at
rest, and it is where ZeroG's grid ends.

The twin is [`benchmarks/05-fast-k-infinity`](../../benchmarks/05-fast-k-infinity): an infinite
medium in eigenvalue mode, probability tables off, 1e8 active histories (1e6 per batch, 20
inactive and 100 active batches). Its k is 1.3226540 ± 6.9 pcm, and rate uncertainties are 0.008%
to 0.042% (Cr52 capture), within a fifth of the 0.2% gate everywhere except Cr52. A second twin
without the cutoff (`sfr-nocut.json`, 4e7 histories) gives 1.3225841 ± 9.7 pcm, so the cutoff
moves k by -7 ± 12 pcm: it is not measurable at this precision, and part 2 replaces it anyway.

Fission spectra are the expected outgoing spectra of OpenMC's own sampling (tabular with unit-base
interpolation between incoming energies, Maxwell, correlated laws), per prompt and delayed
component. They were checked first against analog OpenMC emission at single incoming energies
([`benchmarks/04-fission-spectra`](../../benchmarks/04-fission-spectra), 11 cases from U235 to
Pu242, including U236 partial fission), all within χ² p > 1e-4 in `tests/energy/test_fission.py`.
OpenMC resamples χ until the outgoing energy is below the lowest top of the loaded grids (20 MeV
here, Na23's), so ZeroG truncates and renormalizes each spectrum there, and loses no tail.

```bash
cd studies/03-k-infinity/scripts
```

```bash
PYTHONPATH=. pixi run python run_k.py sfr 1e-3 2
```

The arguments are the case, the grid's thinning tolerance and `subdivide`, which splits every
panel as in study 02. The grid also holds every χ node and every twin tally edge. `results/` holds
the output and the cached inelastic kernels, gitignored. ZeroG's flux is normalized to one fission
neutron produced, and OpenMC tallies per source neutron, so ZeroG's rates are multiplied by k.

## Result

| Subdivide | Nodes | k - twin | Largest rate error | Flux rms z | Residual / source | Solve |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 423,777 | +46.8 pcm | Cr52 capture +0.104% (2.5 sd); +0.04 to +0.06% for most others at 4 to 4.6 sd | 1.47 | -4.9e-4 | 830 s |
| 2 | 837,234 | +11.1 pcm | Cr52 capture +0.060% (1.4 sd); all others within 0.018% | 0.97 | -1.2e-4 | 1831 s |
| 4 | 1,664,148 | +2.1 pcm | Cr52 capture +0.049% (1.2 sd), Fe56 -0.030% (1.1 sd); all others within 0.011% | 0.96 | -3.1e-5 | 4066 s |

The flux tally has 257 bins (20 per unit lethargy) with a median uncertainty of 0.047%. Every run
converged in 5 power iterations and 15 lagged inelastic sweeps. Kernel builds took 264 to 277 s
on 32 processes. Times are of runs one at a time on the 32-core machine.

- **Accuracy:** k converges to the twin, by about 4 per subdivision (47, 11, 2 pcm), the same
  second order as study 02. At subdivide 2 and above, k is within 2 sd and every rate is within
  1.4 sd of the twin.
- **Gates:** k (50 pcm) and rates (0.2%) are already met at subdivide 1, where k is +47 pcm and
  rates are biased by about +0.05%. Subdivide 2 meets both with a margin of 4 or more.
- **Residual:** the balance residual falls by 4 per subdivision too, and decision 005's 1e-4 is
  met at subdivide 4.
- **Iteration:** k changes by 40 pcm between the first and second iteration (the flat starting
  flux sets the first χ mix) and by under 0.2 pcm after that. The inelastic source of the last
  flux starts each solve's lagged sweeps, so the five solves take 15 sweeps in all.

- **Speed:** far off the 10 s target, and not because of fission. A profile at a 1% thinning
  tolerance (244,335 nodes) spent 98% of the eigenvalue solve in `solve_energy`, 137 s per call,
  against 1.5% for fission source and production. That is 17 times decision 008's 8 s for
  heavy-fast at a similar node count: this mix has 12 terms, each with an anisotropic elastic
  window, and the light ones (O16, Na23) span many nodes. Speed is M9's job, and this case is now its benchmark. (The 1% grid is also +282 pcm and
  up to 0.48% on rates, so it fails the gates. 0.1% stays the working tolerance.)

## Errors caught on the way

- A Maxwell spectrum whose energy shift u is positive put 1.1 million χ nodes on the grid and
  lost neutrons below the shift without counting them. Its nodes are now spaced geometrically in
  E - u where truncation acts, and an empty spectrum's emission is counted as lost.
- Near-duplicate χ nodes from different nuclides made sliver panels. Nodes within 1e-12 are
  merged and snapped to the grid.
- The twin first ran each benchmark 04 case in an infinite medium, where Pu239's secondaries
  multiplied without bound. They now run in a 4 cm vacuum box, normalized by a collision tally.

## Decision

- Power iteration with the warm-started lagged sweep stays. Acceleration is not needed at five
  iterations.
- The 0.1% thinned grid at subdivide 2 is the reference for k-infinity until study 04 (M6) chooses
  the grid. Subdivide 1 meets the gates, but with a bias of 4 to 5 sd on most rates.
- Study 03 finishes after M5 with the PWR mix at 600 K, where the 50 eV cutoff goes away.

Limits passed on:

- Correlated laws are not clamped at their last incoming energy, as tabular spectra are, if a
  grid's top lies above their tables. No nuclide here reaches that.
- Fission neutrons born below the cutoff are lost in both codes, about 0.1 pcm.
