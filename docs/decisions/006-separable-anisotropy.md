# 006: Anisotropic elastic window by running moments

Date: 2026-11-03

## Context

Decision 005 added anisotropic elastic scattering as a direct sum over each node's window, which
grows with the square of the node count: 7742 s for study 02's heavy case at 544k nodes. A work
count by term and energy on the heavy case put 93% of it in O16, whose window of 0.25 in lethargy
spans about 13,000 panels of U238's resonance grid between 1 and 40 keV. Every panel took one
quadrature piece, so tuning `PIECES_PER_PEAK` could not help.

## Decision

- Between two table energies the mix weights are linear in E', and on a segment between both
  tables' mu knots each pdf is linear in mu, which is linear in E / E'. So (2f - 1) / E' there is a
  combination of 1, 1/E' and 1/E'^2, with coefficients that depend on E only through the segment.
  The window integral is a sum over segments of three running moments, M_n = ∫ σF E'^-n du, kept in
  the isotropic recursion's blocks. A node costs O(table knots in its window), not O(panels).
- No Legendre truncation is used, so the result is exact for the interpolants. It matches adaptive
  quadrature to 1e-10 where the direct sum reaches 2e-4. The direct sum's frozen-mean correction is
  not needed, because the separable integral of a frozen f vanishes to round-off.
- Moments in this basis cancel on steep segments and between close table energies that differ, by
  about (slope × 2/(1 - α)) or (E'/ΔT × Δp/p). A segment whose terms sum to more than 64 times its
  isotropic integral is integrated by Gauss quadrature over its panels instead (`CANCELLATION_LIMIT`),
  where f is linear and smooth. Without this, round-off in the 14 MeV case reached 6e-10 of F and
  inelastic sweeps stalled above their 1e-12 tolerance.
- The direct sum stays as the oracle: `solve_energy(..., anisotropy="direct")`.

## Consequences

| Case (study 02) | Nodes | Direct | Separable |
| --- | --- | --- | --- |
| heavy x1 | 136k | 498 s | 0.4 s |
| heavy x4 | 544k | 7742 s (README) | 1.5–2.6 s |
| heavy x16 | 2.2M | - | 6.4 s |
| light x1 | 18k | 22 s | 1.8 s |
| light x8 | 140k | 1435 s (README) | 96 s |
| heavy-fast x1, per sweep | 187k | 400 s | 23 s |
| light-fast x1, all sweeps | 49k | 199 s | 33 s |
| light-fast x4, all sweeps | 190k | 3023 s | 390 s |

Direct times for the fast cases are from runs sharing the machine, so are somewhat slow. Fast-case
results agree with the direct sum's to every printed digit.

Captures agree with the direct sum to 7e-8 (heavy) and 6e-7 (light), within its quadrature error.

- Below about 100 keV the cost is linear in nodes, and the elastic speed target is met.
- Above 100 keV the fallback carries most segments, because tables there are close and differ, and
  cost returns to O(window) on those. This is what remains of the speed gap. A fix would keep the
  moments centred per short block, so coefficients stay small. That is multilevel bookkeeping not
  attempted yet.
- The limit trades time against round-off: 512 runs light x8 in 36 s with 5e-13 noise in F, and
  4096 in 15 s with 5e-12, but with 4e-11 on heavy-fast. 64 keeps every case below 1e-13.
- Inelastic kernel setup (`build_inelastic`, 366 s for heavy-fast) now costs more than the solve,
  and is the next target. It is per grid, so it can be cached.
