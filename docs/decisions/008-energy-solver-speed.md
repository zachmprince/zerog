# 008: Centred moment tree for anisotropic elastic, and a sparse, parallel kernel build

Date: 2026-11-07

## Context

After decision 007, study 02's fast cases spent 6 to 13 minutes building inelastic kernels and
40 to 750 s solving, with solve time growing faster than the node count. Profiles of both stages:

- **Solve.** On heavy-fast from 1 to 14 MeV (45,787 nodes), each node's anisotropic window held
  about 266 separable segments, and 98% of them hit decision 006's `CANCELLATION_LIMIT` and fell
  back to quadrature over all their panels (11 per segment on average, more with subdivision).
  That is the superlinear part. Without the fallback the solve still took a third of its time.
- **Build.** U238 from 4 to 14 MeV deposited 1.5e9 lab segments for 1,025 hats. Time went to
  NumPy logs of segment ends, binary searches, re-solving each correlated law's mu strata for
  every outgoing point, and dense coarse x coarse rows. Flipping law tables per call, the review's
  suspect, cost under 1%.

## Decision

- **Anisotropic elastic: a tree of centred moments.** On a segment, (2 f - 1) / E' is a quadratic
  in z = E_r / E' for any reference E_r. Expanded about z = 1 at an E_r inside the segment, every
  coefficient times the range of (z - 1)^n it multiplies is bounded by the kernel's own size, however
  steep the pdf or close the tables, so nothing cancels. A binary tree over panels holds
  the moments of sigma F (z - 1)^n, each node centred at its top energy and built from its
  children's by an exact shift. A segment's whole panels are O(log panels) nodes inside it; its
  partial end panels, and the panel holding the unknown, take the existing three-point Gauss
  quadrature of f itself. The fallback and `CANCELLATION_LIMIT` are gone. The direct sum stays the
  oracle (`anisotropy="direct"`).
- **Inelastic build.** Laws are flipped and their strata precomputed once per law. Strata of each
  centre-of-mass energy are deposited as they are made, sharing bounds between adjacent strata,
  walking node searches, and skipping logs for segments wholly off the grid. Rows are kept sparse
  from the start, and each row only spreads the intervals it touches. Hats are independent, so
  `term_transfer` splits them over forked processes (by default one per 256 emitting hats, up to
  the CPU count). A row's bits do not depend on the split.
- **Numba calls in inner loops take scalars.** Passing an array to a non-inlined Numba function
  costs an atomic reference count increment and decrement. That was 12 to 25% of both stages,
  so the per-segment code is written out in its caller.

## Consequences

Same machine and settings as study 02 (a 16-core Threadripper 2950X under WSL), with old and new
runs back to back while it was otherwise nearly idle. Solve times are of a second solve in the
same process; the first adds 2 to 5 s of compilation, since `_march` is not cached.

| Case | Nodes | Old solve | New solve |
| --- | --- | --- | --- |
| heavy-fast, 1 to 14 MeV | 45,787 | 51 s | 8.4 s |
| light-fast x1 | 57,715 | 34 s | 5.6 s |
| heavy-fast x1 | 195,442 | 52 s | 8.1 s |
| light-fast x4 | 198,556 | 404 s | 22 s |
| heavy-fast x2 | 380,116 | 173 s | 18 s |

| Kernels | Old, serial | New, serial | New, 32 processes |
| --- | --- | --- | --- |
| heavy-fast x1, 10,832 hats | 777 s | 345 s | 48 s |
| light-fast x1, 10,883 hats | 379 s (study 02) | - | 23 s |
| heavy-fast from 4 MeV, 1,025 hats | 240 s (CPU, loaded) | 76 s (CPU, loaded) | - |

- Fluxes agree with the old solver to 4.5e-13 relative at every node in every case above, with
  old or new kernels. Kernel rows agree to 6.5e-15 of each row's largest entry, from summing
  reactions into one row. The new rows drop round-off tails that the old difference array left
  past each segment, up to 14% of O16's stored values, each below 1e-15 of its row.
- Solve time now grows about linearly in nodes, but almost all of it is above 1 MeV: heavy-fast
  from 1 to 14 MeV alone takes as long as the whole range to 20 eV. The 10 s target at 500k
  nodes is about 2.5 times off for heavy-fast. Per segment, the partial-panel quadratures are
  about half of what is left, and enumerating segments (266 per node there) the rest.
- Kernels are cheap enough to rebuild when the fine grid changes in PGD, if processes are
  available.
