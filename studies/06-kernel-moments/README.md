# Study 06: Legendre orders of the scattering kernels

Status: done (milestone M8).

## Question

Spatial PGD modes that aren't isotropic in angle give the energy subproblem a scattering density per
Legendre order. How many orders does the SFR mix's forward peak need, at a fixed streaming
coefficient, before the energy solution stops changing?

## Method

`zerog.energy.moments` gives each nuclide's elastic lab moments <P_l(mu_lab)>, from OpenMC's
centre-of-mass tables as it samples them, and rows of the order-l transfer on inelastic transfer's
coarse hats. In the subproblem each (nuclide, order) is a pseudo-term emitting through its rows;
order 0 keeps the exact window recursion. Inelastic and thermal scattering are order 0 only.

The coefficients are those of an angular mode X(mu) = exp(kappa mu) projected onto itself: removal
density N int X^2 and order-l scattering density N (2l + 1) / 2 X_l^2, with X_l = int P_l X. For
kappa = 3, a strongly forward-peaked mode, the scattering coefficients over removal are 0.332,
0.449, 0.179, 0.036, 0.0044 and 0.0004 for orders 0 to 5. The problem is study 03's SFR mix at
900 K, cut off at 50 eV, with the fission spectrum of a flat flux as a fixed source and no
streaming, on a 1% thinned grid at subdivide 1 (244,110 nodes). Each truncation is compared with
order 5.

```bash
cd studies/06-kernel-moments/scripts
```

```bash
pixi run python run_orders.py 1e-2 1 3 --orders 0,1,2,3
```

Results go to `results/`, gitignored.

## Result

kappa = 3, against the order-5 solution:

| Orders | Flux | Worst rate | Solve | Sweeps |
| --- | --- | --- | --- | --- |
| 0 | -23.6% | U238 fission -26.8% | 354 s | 4 |
| 1 | -7.3% | U238 fission -8.7% | 2573 s | 14 |
| 2 | -0.85% | U238 fission -1.2% | 3210 s | 16 |
| 3 | -0.061% | U238 fission -0.091% | 3307 s | 17 |
| 5 | reference | | 3275 s | 17 |

The worst rates are fission in the threshold fissile nuclides (U238, Am241, Pu240, Pu242), which see
the flux above 1 MeV where elastic scattering is most forward. Lab moments of elastic scattering at
1 and 5 MeV: O16 <P_1> 0.036 and 0.073, Na23 0.385 and 0.497, Fe56 0.037 and 0.732, U238 0.475
and 0.875; at 10 keV every one is below 0.04.

- **Orders:** the error falls about tenfold per order, as the mode's coefficients do. P3 meets the
  0.2% rate gate for this mode (0.091%); P2 misses it by six.
- **kappa = 1** (coefficients 0.29 and 0.016 at orders 1 and 2) was stopped after orders 0 and 1,
  at 349 s and 2109 s; its order-2 coefficient is a fortieth of kappa = 3's.
- **Cost:** each order adds a pseudo-term per nuclide to every sweep, and the rows' emission into
  the hat being marched is lagged, so sweeps rise from 4 to 17 and each costs twice as much at
  order 1. P3 is nine times the order-0 solve here. This is M9's first target: the lagged sweeps
  want the Krylov acceleration the thermal block has, and the pseudo-terms want batching.

## Decision

- The energy subproblem carries elastic orders to P3 for SFR spatial modes; the PGD driver should
  take the order from the mode's own coefficients, since they set the error.
- Order 0 is unchanged from M1 to M7, and every earlier result stands.

Limits passed on:

- Inelastic, thermal and probability-table scattering are order 0 only. Inelastic scattering in the
  SFR's light nuclides is anisotropic in the lab too, so these errors are a lower bound on what
  orders above zero carry. Where probability tables apply, orders above zero take the smooth
  elastic times the band-summed flux, not the bands' shielded elastic, which order 0 uses; with
  <P_1> below 0.04 at 10 keV the difference is small, but it is inconsistent.
- Orders above zero are coarse-hat rows, which smooth a heavy nuclide's narrow window (14 hats for
  U238) at second order in hat width, and spread each outgoing piece uniformly in lethargy, about
  2e-3 of a row's peak against the analytic isotropic kernel. Each hat weights its incoming
  energies by its shape and the cross section at four points, ignoring the flux inside it, which
  is crude in resolved resonances.
- The grid is 1%, coarser than study 04's choice, and only one mode shape was run to completion.
- There is no OpenMC twin: a projected mode problem has no Monte Carlo analogue. The lab moments
  are checked against OpenMC's sampling of its own tables instead (tests/energy/test_moments.py).
- For M9, from the physics review: the march loops over every pseudo-term though their removal
  and window kernels are zero; the extended operators copy the inelastic rows and hold zero rows
  for pseudo-terms; row building allocates two grid-length arrays per hat and order; and the
  kernels build lists, which won't port to Kokkos as they are.
