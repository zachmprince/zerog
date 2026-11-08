# Study 03: k-infinity of real mixes

Status: done. Part 1 is a fast mix at 900 K, cut off at 50 eV in both codes (milestone M4). Part 2 is
a PWR mix at 600 K down to the library's bottom, through the thermal block of M5.

## Question

Does power iteration on the 0D energy solve, with fission spectra collapsed from the evaluated
data, reproduce OpenMC's k-infinity within 50 pcm and its nuclide reaction rates within 0.2%?

## Method, part 1: SFR mix at 900 K

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

## Result, part 1

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
  window, and the light ones (O16, Na23) span many nodes. Speed is M9's job, and this case is
  now its benchmark. (The 1% grid is also +282 pcm and up to 0.48% on rates, so it fails the
  gates. 0.1% stays the working tolerance.)

## Errors caught in part 1

- A Maxwell spectrum whose energy shift u is positive put 1.1 million χ nodes on the grid and
  lost neutrons below the shift without counting them. Its nodes are now spaced geometrically in
  E - u where truncation acts, and an empty spectrum's emission is counted as lost.
- Near-duplicate χ nodes from different nuclides made sliver panels. Nodes within 1e-12 are
  merged and snapped to the grid.
- The twin first ran each benchmark 04 case in an infinite medium, where Pu239's secondaries
  multiplied without bound. They now run in a 4 cm vacuum box, normalized by a collision tally.

## Method, part 2: PWR mix at 600 K

The mix is a homogenized PWR cell at 600 K, also in [`scripts/cases.py`](scripts/cases.py): 30%
UO2 at 4.0 wt% U235, 60% water at 0.70 g/cm3 and 10% zirconium (all Zr90), with H1 bound in
`c_H_in_H2O` and O16 free gas, as this library has no bound oxygen table. Neither code has a
cutoff: ZeroG's grid ends at the library's bottom, 1e-5 eV, and loses what scatters below it.

The twin is `pwr.json` in [`benchmarks/05-fast-k-infinity`](../../benchmarks/05-fast-k-infinity),
with 2e8 active histories (2e6 per batch, 20 inactive and 100 active batches): k is 1.3101318 ±
6.7 pcm, and rate uncertainties are 0.008% to 0.010%, except O16 capture (0.058%) and Zr90 capture
(0.032%), which are reported but not gated. The flux tally has 566 bins from 1e-5 eV to 20 MeV.

The thermal kernels follow OpenMC's sampling (decision 009 and `zerog.energy.thermal`):

- Free gas, the constant cross section kernel, for every nuclide below 400 kT (20.7 eV), with
  the target at rest above. Hydrogen is free gas at every energy in OpenMC, and up to 4e4 kT (2.07
  keV) here; see below.
- S(alpha, beta) for H1 at and below the table's `energy_max`, 10 eV, from the nearer incoming
  table, as OpenMC picks it, scaled or shifted to the incoming energy.
- Both are rows on coarse hats, like inelastic transfer. Their upward part is lagged in the march,
  and the block of hats below the first that scatters upward is solved by GMRES, preconditioned by
  the march.

```bash
PYTHONPATH=. pixi run python run_k.py pwr 1e-3 2
```

A fourth argument sets hydrogen's free gas top in kT.

## Result, part 2

| Subdivide | Nodes | k - twin | Largest rate error | Flux rms z | Residual / source | Solve |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 261,691 | +22.0 pcm | O16 capture +0.070% (1.2 sd); all others within 0.021% (2.2 sd) | 1.05 | -1.5e-4 | 597 s |
| 2 | 508,754 | +12.7 pcm | O16 capture +0.067% (1.2 sd); all others within 0.025% (1.5 sd) | 1.02 | -3.9e-5 | 1238 s |
| 4 | 1,002,880 | +10.3 pcm | O16 capture +0.066% (1.1 sd); all others within 0.024% (1.5 sd) | 1.02 | -9.9e-6 | 2577 s |

Every run took 4 power iterations, 10 full sweeps and 66 marches of the upscatter block inside
its GMRES solves; with the final block tolerance (see below), subdivide 1 takes 56 and gives the
same k to seven digits. Kernel builds took about 220 s on 32 processes. Solve times are with other jobs
running on the machine.

- **Gates:** k (50 pcm) and rates (0.2%) are met at every subdivision, by a margin of 2 or more
  at subdivide 1. The balance residual falls by 4 per subdivision and meets 1e-4 from
  subdivide 2.
- **Accuracy:** k falls from +22 to +13 to +10 pcm, so the grid's part converges at second order
  to about +9.5 pcm, 1.4 twin sd. The flux bins near the thermal peak sit +0.12% to +0.15%
  high (z about 3 to 4, at 0.027 and 0.15 eV) at every subdivision. Halving every thermal coarse
  hat at subdivide 2 changed k by under 0.1 pcm and these bins by under 0.01%, so neither is set by
  the thermal hats. It is within statistics on k, but a systematic of this size on the thermal
  spectrum is worth a look in study 04 (M6), which certifies the grid.
- **Hydrogen free gas:** the measurement that changed the kernels. With the target at rest above
  400 kT, as for every other nuclide, ZeroG was 62 pcm high at subdivide 1. Free gas smears the top
  of hydrogen's kernel by about sqrt(4 kT E), 2.7 eV at 36 eV, which is wider than U238's low
  resonances, so the cap moves resonance escape:

  | Hydrogen free gas top | k - twin, subdivide 1 |
  | --- | --- |
  | 400 kT (20.7 eV) | +62.3 pcm |
  | 4e3 kT (207 eV) | +26.8 pcm |
  | 4e4 kT (2.07 keV) | +22.0 pcm |
  | 4e5 kT (20.7 keV) | +21.8 pcm |

  4e4 kT is the default for targets of awr <= 1, 0.2 pcm from the converged value.
- **Speed:** as in part 1, far off the target. The block's GMRES takes 56 to 66 marches over the
  hats below 2 keV per eigenvalue solve, and hydrogen's rows hold about 3.5e7 values.

## Errors caught in part 2

- U235's MT 5 emits neutrons down to the library's bottom, so its hats hold both inelastic and
  thermal rows. They are mixed by each one's hat-integrated emission, as reactions are.
- The thermal coarse grid dropped reaction thresholds below its reach, U235's 77 eV level among
  them. They are put back.
- The physics review found that the block's marches started one coarse interval too low, which
  left its Krylov operator slightly inexact (the fixed point was unaffected), and that a
  hat whose inelastic row lands wholly off the grid would have scattered its neutrons rather
  than lost them. Both are fixed.
- A GMRES tolerance scaled by the square root of the block's size let single entries exceed the
  outer sweep test, which then never passed; and an absolute tolerance alone stalled on round-off
  in one test. The block solve now stops at a tenth of the outer tolerance or 1e-10 of its
  right-hand side, whichever is reached first, and the outer sweeps take the rest.

## Decision

- Power iteration with the warm-started lagged sweep stays. Acceleration is not needed at four
  or five iterations.
- The 0.1% thinned grid at subdivide 2 is the reference for k-infinity until study 04 (M6) chooses
  the grid. Subdivide 1 meets the gates for both mixes, but with biases of 4 to 5 sd on most SFR
  rates.
- Thermal scattering follows OpenMC's sampling, with hydrogen free gas to 4e4 kT. The upscatter
  block is solved by GMRES on the lagged hat emission, as decision 009 records.

Limits passed on:

- Correlated laws are not clamped at their last incoming energy, as tabular spectra are, if a
  grid's top lies above their tables. No nuclide here reaches that.
- Fission neutrons born below the SFR cutoff are lost in both codes, about 0.1 pcm.
- The PWR's +10 pcm floor and its +0.13% thermal-peak bias are unexplained.
- Thermal scattering is order 0 only, and coherent or incoherent elastic tables (graphite, ZrH,
  Be) are refused. No bound oxygen or uranium tables are used, in either code.
