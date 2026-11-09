# 011: The unresolved range as probability-table bands with a shared scattering source

Date: 2026-11-08

## Context

The plan expected bands as a discrete index on the energy mode with a shared scattering source,
and sampled resonance ladders only if bands missed the gate. With OpenMC's probability tables on,
study 03's SFR mix gains 94.7 ± 9.6 pcm, and infinite dilution misses the k gate by 34 pcm and U238
capture's by a hair. OpenMC v0.16.0 (`Nuclide::calculate_urr_xs`) draws one random number per
nuclide per flight, uses it at both bracketing tables, and draws again after each collision.

## Decision

- **Bands are points in r.** Per flight, the cross sections of every nuclide with tables are a
  function of r = (r_1, ..., r_K), drawn afresh each flight. So F splits into F_c over a fixed set
  of points c, each with its own removal and all fed by one scattering source in proportion to its
  weight. The march solves each node in closed form: the own panel couples the points with rank
  one. `solve_energy`'s signature is unchanged; its solution carries each banded term's
  self-shielded total, elastic, capture and fission.
- **64 scrambled Sobol points**, equal weights, for any number of nuclides. Independent sampling
  needs no product of bands: on two nuclides the Sobol sets are within 0.05% of a 4096-point
  product.
- **Each band reaction is scaled to its exact mean at each node.** A band's values are steps in r,
  so a point set misses the mean by order 1 / points where a stratum straddles a cdf breakpoint,
  percents with fewer points than bands. The scale tends to one with more points.
- **OpenMC's total there.** Inside a range the total is elastic + capture + fission + the flagged
  inelastic reaction, (n,gamma) is absorption less fission, and other reactions drop out, as in
  OpenMC.
- **No sampled ladders.**

## Consequences

- Study 05: with 64 points, the SFR mix is +16.5 pcm from the tables-on twin, and every gated rate
  is within 0.073% (Pu240 capture).
- Bands cost about 16% of solve time over 16 points, and grow with points times banded nuclides
  times nodes in the tables' ranges.
- Fission production, power iteration, the balance and the twin comparison take the shielded
  cross sections; `zerog.energy.urr.shielded` and `zerog.energy.fission.shield_fission` apply
  them.
- Not covered: the PWR mix with tables on; window edges and anisotropic moments, which take the
  shielded cross sections (second order in panel width); the anisotropic own-panel part, which
  takes the smooth elastic; bands of higher Legendre orders (M8); and, for Phase 2, OpenMC
  keeping a flight's r across material crossings, which a shared in-leakage source drops.
- The kernel recomputes each table's index for every combination and allocates per call, which
  M9 should hoist and batch, as the Kokkos port will want anyway.
