# Study 00: Space-energy rank of the pointwise resonance-range flux (toy)

## Question

How many separated modes would a space-energy expansion need if energy were resolved pointwise
instead of in groups?

## Method

A pointwise slowing-down solve produces a reference flux, and a weighted SVD of that flux gives the
best possible rank-M result in its norm. These numbers are therefore a **lower bound** on what a
Galerkin PGD would need. No PGD is run here.

- **Solver:** 1D slab lattice, S_N step characteristics (16 angles), reflective boundaries, elastic
  target-at-rest kernel marched in lethargy with trapezoid quadrature.
- **Data:** synthetic. Doppler-broadened single-level Breit-Wigner resonances with a "U-238-like"
  ladder, so no library and no OpenMC twin.
- **Range:** 1 eV to 5 keV, with E·φ = 1 assumed above 5 keV.
- **Error metric:** worst per-cell capture rate error of a rank-M truncation of ψ(x,μ,E), over any
  absorber and any fuel cell.

Cases: A is one pin at 900 K. B is one pin with a 600-1200 K fuel profile. C is one pin plus a second
absorber concentrated at the rim. D is two different pins in one cell.

Scripts are in `scripts/`. `results/` is gitignored (about 55 MB). Case D's angular SVD needs about
3 GB of RAM.

```bash
cd studies/00-toy-rank/scripts
pixi run python verify.py
pixi run python run_cases.py
pixi run python plots.py
```

### Model limits

- The lowest 13 resonances have approximately realistic parameters; those above 300 eV are sampled
  (Wigner spacing, Porter-Thomas widths). Single-level, s-wave only, with Gamma_n not energy
  dependent.
- All uranium is U-238-like: no U-235 and no fission source.
- Slabs with chord-preserving thicknesses rather than cylinders.
- Isotropic lab-frame scattering, target-at-rest kinematics, no S(α,β), no inelastic scattering.

## Result

Modes needed for 0.1% worst per-cell capture error, angular flux separated as
(x,μ) × E, from the rerun on 2026-11-01:

| Case | Energy points | Modes for 1% | Modes for 0.1% |
| --- | --- | --- | --- |
| A: one pin, 900 K | 35,966 | 10 | 17 |
| B: one pin, temperature profile | 36,061 | 12 | 22 |
| C: one pin + rim absorber | 91,953 | 11 | 19 |
| D: two different pins | 92,215 | 17 | 30 |

So 17-30 modes stand in for 36,000-92,000 energy points. Case D needs the most, but it changes
temperature, absorber, pin count and mesh together, so no run isolates the cause. Splitting energy
into bands raises the pooled mode count: for the scalar flux in case A, 0.1% takes 7 modes with one
band, 20 with four and 39 with eight (15, 45 and 67 in case D). Band-local modes span fewer energy
points, though, and total unknowns were not measured, so this does not settle whether bands cost
more overall. See `figures/fig1_rank.png` and `figures/fig2_modes.png`.

**Solver check.** With no absorption the exact answer is F = E·φ = 1. `verify.py` gives
max|F-1| = 7.2e-6 on the case A grid. That is discretization error rather than round-off: on uniform
lethargy grids the error falls by a factor of 3.99 per halving of du (1.5e-5, 3.8e-6, 9.5e-7 at
du = 0.004, 0.002, 0.001, from `scripts/convergence.py`), which is the trapezoid rule's second order.

## Decision

**Go**, for continuing with a separated representation, with the bound's limits in view: a real PGD
will need more modes, and real data, fission and scattering are all still untested. Band splitting
remains open, since only its mode count was measured. See
[decision 002](../../docs/decisions/002-toy-rank.md).
