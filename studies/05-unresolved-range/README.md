# Study 05: The unresolved range as probability-table bands

Status: done (milestone M7).

## Question

With OpenMC's probability tables on, does a band treatment in the energy subproblem reproduce
k-infinity and reaction rates within the gates (50 pcm, 0.2%) for the SFR mix, where infinite
dilution does not? How many band combinations does it need, given that OpenMC samples every
nuclide's band independently?

## Method

The mix and settings are study 03's SFR case at 900 K, cut off at 50 eV, on study 04's grid (all
reactions at 0.1%, subdivide 2). Seven nuclides have tables, spanning 150 eV (Am241) to 149 keV
(U238); none of O16, Na23, Fe56, Cr52 or Ni58 does.

The twin is `sfr-ptables.json` in [`benchmarks/05-fast-k-infinity`](../../benchmarks/05-fast-k-infinity):
the same model as `sfr.json` with `ptables` on, and 1e8 active histories. Its k is 1.3236007 ±
6.6 pcm, 94.7 ± 9.6 pcm above the tables-off twin, and tables move U238 capture by -0.20% and
U235, Am241, Na23 and Fe56 capture by +0.20% to +0.26%.

The bands follow `Nuclide::calculate_urr_xs` (`zerog.energy.urr`). Each flight draws a random
number r per nuclide, the same r picks a band at both bracketing tables, and the stream advances
after each collision, so the next flight draws again. So the flux splits into one F_c per point c
of the unit cube in r, each with its own removal and all fed by one scattering source, which
the march solves in closed form at each node. The points are a fixed scrambled Sobol set with equal
weights. Each band reaction is scaled at each node so the points' mean is the exact mean over r,
leaving the points to carry only the distribution's shape: without it, 16 points over 20 bands
missed Pu241's mean capture by 1 to 3%, and Pu241 capture came out 2.7% high.

```bash
cd studies/05-unresolved-range/scripts
```

```bash
pixi run python run_bands.py 64 1e-3 2
```

```bash
pixi run python product.py
```

`product.py` is the check on independent sampling: U238 and Pu239 in sodium as a fixed-source
problem on a 1% grid, with Sobol sets against every pair of 64 equal strata per nuclide (4096
combinations).

## Result

Against the tables-on twin, at 837,326 nodes (the infinite dilution row is study 03's run on the
same grid):

| Combinations | k - twin | Largest gated rate error | Largest rate error | Flux rms z | Solve |
| --- | --- | --- | --- | --- | --- |
| none | -83.6 pcm | U238 capture +0.204% | Na23 capture -0.255% | - | 1831 s |
| 16 | +22.8 pcm | Am241 capture +0.097% (8.0 sd) | the same | 1.00 | 3326 s |
| 64 | +16.5 pcm | Pu240 capture +0.073% (5.0 sd) | Cr52 capture +0.091% (2.0 sd) | 1.19 | 3854 s |
| 256 | +18.7 pcm | Pu240 capture +0.043% (3.0 sd) | Cr52 capture +0.089% (2.0 sd) | 1.15 | 5819 s |

The banded runs shared the machine three at a time; the infinite dilution solve ran alone. Every
run took 5 power iterations and 15 sweeps.

Two nuclides, against the 64^2 product (rate errors):

| Points | U238 capture | Pu239 capture | Pu239 fission | Flux 2.5 to 149 keV |
| --- | --- | --- | --- | --- |
| none | +0.365% | -0.731% | -0.337% | -0.816% |
| Sobol 16 | -0.007% | +0.028% | +0.007% | +0.029% |
| Sobol 64 | -0.027% | +0.053% | +0.026% | +0.049% |
| Sobol 256 | -0.006% | +0.011% | +0.005% | +0.009% |

- **Gates:** infinite dilution fails both with tables on, by 34 pcm and on U238 capture. Bands
  meet both with any number of combinations tried: k within 23 pcm, every gated rate within 0.1%.
- **Combinations:** k moves by under 7 pcm between 16, 64 and 256, within the twin's 6.6 pcm.
  Rates settle by 64. The product check puts the Sobol sets within 0.05% of 4096 combinations, a
  fifteenth of what the bands change, so a fixed low-discrepancy set does handle independently
  sampled nuclides.
- **Against the tables-off result:** study 03's run on this grid was +11.1 pcm against the
  tables-off twin; with bands it is +16.5 pcm against the tables-on one. The bands raise ZeroG's k
  by 100.1 pcm, against OpenMC's 94.7 ± 9.6.
- **Cost:** 16% more solve time at 64 combinations than at 16, and another 51% at 256, all in the
  band loop at each node in a table's range.

## Decision

- Probability tables are bands with a shared scattering source, as above, with 64 Sobol
  combinations and each reaction scaled to the exact mean at each node. Decision 011 records it.
- Sampled resonance ladders are not needed: bands meet the gate with tables on.

Limits passed on:

- Inside a table's range OpenMC's total holds only the flagged inelastic reaction, so it drops
  MT 5 and the (n,alpha) reactions there, and ZeroG drops them too. Rows of hats there still mix in
  MT 5's tiny share.
- The march takes self-shielded cross sections at window edges and in anisotropic moment trees.
  On a synthetic heavy banded term the flux then misses a dense oracle of every band by 2.6e-3,
  6.3e-5 and 1.6e-5 at panels of 0.012, 0.006 and 0.003 in lethargy; the panel above each node
  and the running sums are band-exact.
- The anisotropic correction's own-panel part takes the smooth elastic at the node and is common
  to every band. No test covers an anisotropic banded term.
- Only the SFR mix was run. The PWR mix's tables (U235 and U238 from 2.25 keV, Zr90 from 200 keV)
  are not yet tested against a tables-on twin. Burned fuel adds tables down to 0.3 eV (Dy159,
  Eu156, Pm148, Sm153, Xe135), inside the thermal block, where a range's elastic goes through
  the target-at-rest window as OpenMC takes it at rest there, and hat rows keep free gas shapes.
- OpenMC keeps a flight's r across surfaces and materials. In a heterogeneous PGD problem
  in-leakage would feed every band afresh, which drops that correlation (Phase 2).
- The physics review found that a march starting inside a range lost the band state of the node
  above, which the upscatter block's marches would hit with burned fuel; it is fixed and tested,
  and no run here started a march anywhere but the top.
