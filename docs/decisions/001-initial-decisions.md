# 001: Initial project decisions

Date: 2026-10-31

## Context

ZeroG starts from a development plan for a deterministic full-core transport solver with no
multigroup library. These are the choices the plan settles before any solver code exists. The
reference-solution row was proposed in the plan rather than settled; recording it here adopts it
as the working choice until a later record changes it.

## Decision

| Area | Choice | Consequence |
| --- | --- | --- |
| Language | Python + Numba prototypes | Port profiled hot paths to a C++ core with a Python interface once the method passes its gates |
| Nuclear data | OpenMC HDF5 library, ENDF/B-VIII.0 to start | Pointwise data at 250, 293.6, 600, 900, 1200 and 2500 K, with thermal scattering included |
| Space and angle | DG finite elements + S_N | Reads an unstructured finite element mesh directly |
| Reactor types | PWR for benchmarks, SFR cases early | SFR cases in Phases 1, 2 and 4 prove a fast spectrum works before PWR choices harden |
| Hardware | GPUs eventually | Keep data layouts flat, array-based and batched from the start |
| Codebase | Fresh repo named ZeroG | Seeded with the 1D toy as study 00 |
| Reference (proposed) | OpenMC continuous-energy Monte Carlo, same library | Differences reflect the method, not the data |

## Consequences

- Only ENDF/B-VII.1 ships with a windowed multipole library. For VIII.0, temperatures between the
  tabulated points need interpolation or our own broadening (Phase 3).
- ENDF/B-VIII.1 is available in the same format, so switching library later is cheap.
- Core code assumes no thermal spectrum, square lattice or water.
