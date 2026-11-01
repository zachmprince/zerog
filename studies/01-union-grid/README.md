# Study 01: Size of the union energy grid

## Question

How many energy points does a pointwise grid need for a PWR pin, and for an SFR mix, at a given
interpolation tolerance?

## Method

A greedy kernel (`zerog.data.grid.thin_indices`) drops points while lin-lin interpolation stays
within `rtol * |xs|` plus a floor, from 20 MeV to the bottom of ENDF/B-VIII.0. The floor is 1e-8 b
for microscopic curves and 1e-8 /cm for macroscopic ones. Each grid is valid only for the curves it
tested, at the temperatures it used:

| Grid | Curves tested | Valid for |
| --- | --- | --- |
| All reactions | Every non-redundant reaction, per nuclide, then merged | Any non-negative densities |
| Total + major | Total, elastic, capture, fission, per nuclide, then merged | Those curves only |
| Joint | The same curves, tested together per material, then merged | Those curves only |
| Macroscopic | Each material's macroscopic total, elastic, capture, fission | That composition only |

Rare reactions break the reduced grids. The first review measured errors of 99.5% on U238 MT 56 with
total + major, and 26.7% on Fe56 MT 51 with joint, both at 0.1% and for cross sections of at least
1 mb. Only all-reactions grids should be used where inelastic kernels are needed.

Materials: UO2 at 900 K (3.1% enriched, with U234 and U236); Zircaloy-4 (Zr, Sn, Fe, Cr) and borated
water (H, O, B) at 600 K. SFR mix: MOX at 1200 K, steel (Fe, Cr, Mo, Mn, Ni) and sodium at 600 K.
These are natural isotopes in round-number proportions, not the VERA specification. The nuclide list
changes every column, while the densities change only the macroscopic one.

```bash
cd studies/01-union-grid/scripts
pixi run python run_grid.py
```

There is no OpenMC twin, which is a deliberate exception to the study convention: this answers a
data-size question, with no transport result to compare. The script rechecks every grid against its
tested curves at every point, independently of the kernel.

## Result

Energy points, from `results/grid_sizes.csv`, gitignored:

| Case | rtol | All reactions | Total + major | Joint | Macroscopic |
| --- | --- | --- | --- | --- | --- |
| PWR pin (629,163 unthinned) | 1% | 315,802 | 311,688 | 183,486 | 58,589 |
| | 0.1% | 566,878 | 561,339 | 399,830 | 165,509 |
| | 0.01% | 601,016 | 595,468 | 546,876 | 252,295 |
| SFR mix (660,265 unthinned) | 1% | 336,329 | 333,019 | 184,307 | 69,513 |
| | 0.1% | 603,055 | 599,864 | 408,349 | 199,822 |
| | 0.01% | 633,802 | 631,734 | 577,675 | 344,990 |

Per material, PWR pin, at 0.1%:

| Material | All reactions | Joint | Macroscopic |
| --- | --- | --- | --- |
| UO2 | 172,358 | 125,422 | 103,454 |
| Zircaloy-4 | 395,229 | 274,777 | 61,940 |
| Borated water | 4,670 | 2,640 | 1,472 |

- **Grid valid for all reactions:** about 316,000 points at 1% and 567,000 at 0.1% for one temperature
  per material. Spanning 600, 900 and 1200 K in the fuel and 294 and 600 K elsewhere raises this to
  442,276 and 667,594. A grid thinned at one temperature breaks tolerance at another, so a real
  temperature field needs the larger figure or more.
- **Between tabulated temperatures:** with UO2 at 750 K and the rest at 565 K, a grid for exactly
  those temperatures needs 317,138 points at 1% and 585,647 at 0.1%, close to the one-temperature
  counts. Cross sections there are the per-lookup expected values of OpenMC's `interpolation` method,
  linear in T between bounds, from `zerog.data.temperature`. This mix is not physical Doppler
  broadening. For the mixed curves, each bound evaluated through its own thinned grid stays within
  tolerance; exact mixed values at the merged grid's nodes can exceed it slightly (err/tol 1.18 on
  U238 capture at 650 K). The review also measured OpenMC's first-flight rate moving towards the
  harmonic mean in thick media, so twins should stay at tabulated temperatures for now.
- **Comparison with the toy:** the toy needed 36,000-92,000 points, with one resonant nuclide and an
  upper limit of 5 keV. Only the 1% macroscopic grids fall in that range.
- **Cladding:** for this fresh-fuel list, Zircaloy-4 needs more microscopic points than UO2, and 6.4
  times its own macroscopic count. Trace isotopes count fully in a microscopic test, which suggests
  density weighting or lumping is worth trying. Burned fuel could change this.
- **Rare reactions:** they add about 1% to the point count, but the reduced grids do not bound them.
- **PWR against SFR:** all-reactions counts differ by 5-6%, and macroscopic ones by 19-37%.
- **Macroscopic floor:** the count depends strongly on it. At 0.1% the PWR pin gives 144,015 with
  1e-6 /cm and 68,064 with 1e-4 /cm.

Not covered:

- An interpolation tolerance is not a reaction-rate error, so the tolerance needed is a Phase 1
  question, as is the effect of lethargy-based quadrature.
- ν is not a tested curve, and neither are secondary distributions. The review measured ν·σf error of
  1.06e-3 for Pu239 on a 0.1% grid.
- PGD effective densities can be negative. The relative bound then fails through cancellation, and
  only an absolute one holds, so Phase 1 should measure error on actual effective densities.
- S(α,β) grids, burned fuel and a continuous temperature field are not included, so these figures
  are lower estimates for a real core. The thinning is greedy, so each is an upper bound on
  the smallest grid for its own curves, nuclides and temperatures only.
- Above 20 MeV is cut. χ tails will need handling before this feeds a solver.

## Decision

**Gate item met:** N_E for a PWR pin is known at the tabulated temperatures. Plan for roughly 300,000
to 700,000 points on a grid that covers all reactions, unless Phase 1 shows reduced grids are
enough. See [decision 003](../../docs/decisions/003-union-grid.md).
