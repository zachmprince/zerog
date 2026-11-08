# Study 04: Choice of energy grid

Status: done (milestone M6).

## Question

Decision 003 left the grid open. Which thinning tolerance and grid type give the gate's accuracy
(k within 50 pcm, gated rates within 0.2%) for both study 03 mixes? Are reduced or
density-weighted grids enough? How large is the error with negative effective densities? And what
do nu and secondary distributions, which no grid is thinned on, add?

## Method

Every run is a study 03 k-infinity solve on its twin (SFR at 900 K, cut off at 50 eV; PWR at 600 K,
with S(alpha, beta)), at subdivide 2, through `zerog.energy.mix.build_mix` and
`zerog.analysis.twin.compare`. The grid types are `zerog.energy.terms.thinned_grids`:

| Grid | Thinned on | Valid for |
| --- | --- | --- |
| all | every non-redundant reaction of each nuclide | any densities |
| major | each nuclide's total, elastic, capture and fission | those curves |
| macroscopic | the mix's total, elastic, capture and fission | that composition |

The tolerances are 1%, 0.3%, 0.1% and 0.03%, with a floor of 1e-8 (barns, or 1/cm for the
macroscopic grid). Twin uncertainty is 6.9 pcm (SFR) and 6.7 pcm (PWR) on k, and 0.008% to 0.017%
on gated rates; capture in O16, Na23, Fe56, Cr52, Ni58 and Zr90 is reported, not gated.

```bash
cd studies/04-grid-choice/scripts
```

```bash
pixi run python run_grid.py sfr all 1e-3 2
```

```bash
pixi run python certify.py sfr all 1e-3 2
```

```bash
pixi run python negative.py all 1e-3
```

`run_grid.py` writes a JSON summary and the flux to `results/`, gitignored. `certify.py` compares
nu_t sigma_f and inelastic emission (yield times sigma) as the solve sees them, products at the
library's points interpolated onto the grid, with the truth OpenMC samples, the yield evaluated at
E, weighted by that run's flux. `negative.py` solves the SFR mix as a fixed-source problem with the
physical densities and again with O16, Fe56 and Ni58 negative and Cr52 halved (with a streaming
term that keeps the diagonal positive), against the same problems on an all-reactions grid at
1e-5; `negative.py summary` tabulates them.

## Result

k against the twin, and the largest gated rate error (with its twin sd):

| Mix | Grid | Tolerance | Nodes | k - twin | Largest gated rate error | Flux rms z | Solve |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SFR | all | 1% | 478,350 | +72.4 pcm | Pu239 capture +0.116% (7.6) | 2.22 | 3137 s |
| SFR | all | 0.3% | 652,536 | +23.6 pcm | Pu240 capture +0.033% (2.2) | 1.09 | 3560 s |
| SFR | all | 0.1% | 837,234 | +11.1 pcm | Pu240 capture +0.015% (1.0) | 0.97 | 3866 s |
| SFR | all | 0.03% | 878,674 | +9.7 pcm | Pu240 capture +0.013% (0.9) | 0.96 | 3764 s |
| SFR | major | 0.1% | 835,852 | +11.1 pcm | Pu240 capture +0.015% (1.0) | 0.97 | 3198 s |
| SFR | macroscopic | 1% | 227,948 | +227.0 pcm | Pu239 capture +0.368% (24.1) | 6.15 | 1872 s |
| SFR | macroscopic | 0.3% | 299,010 | +84.7 pcm | Ni58 capture +0.140% (8.6) | 2.30 | 2074 s |
| SFR | macroscopic | 0.1% | 398,404 | +32.5 pcm | Ni58 capture +0.050% (3.1) | 1.19 | 2250 s |
| SFR | macroscopic | 0.03% | 507,060 | +16.4 pcm | Pu240 capture +0.022% (1.5) | 1.00 | 2419 s |
| PWR | all | 1% | 332,356 | +42.8 pcm | U238 capture +0.050% (5.3) | 1.14 | 999 s |
| PWR | all | 0.3% | 412,448 | +19.8 pcm | H1 capture +0.019% (2.0) | 1.04 | 1105 s |
| PWR | all | 0.1% | 508,754 | +12.7 pcm | U238 nu-fission -0.015% (1.5) | 1.02 | 1248 s |
| PWR | all | 0.03% | 536,974 | +12.4 pcm | U238 nu-fission -0.015% (1.5) | 1.02 | 1344 s |
| PWR | major | 0.1% | 508,444 | +12.7 pcm | U238 nu-fission -0.015% (1.5) | 1.02 | 1456 s |
| PWR | macroscopic | 1% | 276,604 | +72.5 pcm | U238 capture +0.090% (9.5) | 1.35 | 1169 s |
| PWR | macroscopic | 0.3% | 348,854 | +31.2 pcm | U238 capture +0.033% (3.5) | 1.08 | 1295 s |
| PWR | macroscopic | 0.1% | 428,616 | +16.2 pcm | H1 capture +0.016% (1.7) | 1.03 | 1407 s |
| PWR | macroscopic | 0.03% | 488,612 | +13.1 pcm | U238 nu-fission -0.015% (1.5) | 1.02 | 1480 s |

Four runs at a time shared the 32-core machine, so solve times are only comparable with each other.
Ungated capture was within 0.08% on every all-reactions grid from 0.3% (Cr52 in the SFR mix, O16
in the PWR), and up to 0.49% on macroscopic ones (Cr52, at 1%).

Negative effective densities, largest capture error against the 1e-5 grid (SFR fixed source):

| Grid | Tolerance | Nodes | Physical densities | With negative densities |
| --- | --- | --- | --- | --- |
| all | 1% | 244,110 | Ni58 -0.034% | Ni58 -0.035% |
| all | 0.3% | 331,200 | O16 +0.034% | O16 +0.034% |
| all | 0.1% | 423,549 | Cr52 -0.0003% | Cr52 -0.0003% |
| all | 0.03% | 444,269 | below 0.0001% | below 0.0001% |
| macroscopic | 1% | 118,948 | Cr52 +0.84% | Cr52 +0.82% |
| macroscopic | 0.3% | 154,469 | Cr52 +0.43% | Cr52 +0.42% |
| macroscopic | 0.1% | 204,157 | Cr52 +0.22% | Cr52 +0.21% |
| macroscopic | 0.03% | 258,473 | Cr52 +0.065% | Cr52 +0.065% |

The flux integral moved by the same amount, to 1e-6, with either set of densities on every grid.

nu and secondary distributions (`results/certify_*.txt`), as rate errors weighted by the run's
flux: nu-fission within 2.2e-5 for every nuclide on every grid checked (all at 1% and 0.1%,
macroscopic at 0.1%), and inelastic emission within 4.2e-5 (Fe56, macroscopic at 0.1%; 1.5e-7 on
the all-reactions grid at 0.1%). The largest nu-fission error, U238's 2.2e-5, is the same on the
library's own grid: it is the product interpolated between the library's points, not the
thinning.

- **Tolerance:** on the all-reactions grid both mixes meet the gates from 0.3%, where k is +24 and
  +20 pcm and rates are off by at most 2.2 sd. 1% fails the SFR's k (+72 pcm) and biases rates by
  5 to 8 sd. From 0.1% the grid's error is below twin noise: 0.03% changes k by 1.4 and 0.3 pcm.
- **Major curves:** within 0.3% of the all-reactions grid's size, with the same answers to 0.1
  pcm. Study 01 found rare reactions off by up to 99.5% on such grids, and nothing here repays
  that risk.
- **Macroscopic:** a composition-specific grid half the size, at the same tolerance, is about as
  accurate as an all-reactions grid three times looser: SFR 0.1% (398k nodes, +33 pcm) against all
  at 0.3% (653k, +24 pcm). A minor nuclide's resonances barely move the mix's curves, so they
  aren't resolved: Cr52's capture is off by 0.22% at 0.1%, beyond the gate. And PGD's effective
  densities differ from the physical mix in every mode.
- **Negative densities:** no measurable effect on any grid. A per-nuclide grid holds each cross
  section within its own tolerance whatever the densities, and the rate errors above are those of
  each nuclide's own curve, which the signs don't change. Decision 003's worry, that a sum of
  curves loses its relative bound, would show where terms nearly cancel; this mode has no such
  energy, and the margin to the diagonal kept by the streaming term may hide it in others.
- **nu and secondary laws:** four orders below the gate.
- **The PWR floor:** k stays +12.4 to +13.1 pcm from 0.1% on every grid type, and study 03 found
  the thermal hats don't set it either. So it is neither thinning nor the coarse hats, and stays
  open (1.9 twin sd).

## Decision

- The grid is the all-reactions union, thinned to 0.1%, at subdivide 2: SFR 837k and PWR 509k
  nodes, within 13 pcm and 0.015% of the twins. Decision 010 records it.
- 0.3% is the fallback if speed demands it (22% to 19% fewer nodes, about +10 pcm and twice the
  rate error, still within the gates).
- Macroscopic and major grids are not used.

Limits passed on:

- Only order-0 solves were tested: anisotropic moments (M8) and unresolved-range bands (M7) may
  want their own nodes.
- Each mix was tested at one temperature, and the grid is valid only at the temperatures it was
  thinned at.
