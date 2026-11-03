# 005: Energy subproblem interface and discretization

Date: 2026-11-01

## Context

Phase 1 builds the 0D energy solve as the PGD energy subproblem. Its first milestone fixes the
unknown, the quadrature and the signature that later kernels and Phase 4 depend on.

## Decision

- The unknown is the flux per unit lethargy F = E ε(E), piecewise linear in lethargy, with cross
  sections linear in E. Panel integrals of the product are exact, so a pure scatterer with a constant
  cross section gives constant F to round-off on any grid. With a varying one, the exact F is
  proportional to 1/σ, which the grid only approximates.
- The equation is collocated at the nodes, with the implicit part of a node's own panel on the left.
- `solve_energy(ops, removal, scatter, streaming, source)` takes densities by term and order, which
  can be negative. The source is per unit lethargy. Each node's diagonal is returned and not treated.
- Isotropic elastic scattering uses a running integral per term. Window edges split a panel into a
  piece that is still lin-lin, so window integrals are exact for the interpolants. Every panel must
  be narrower than every window, which `build_operators` enforces and `limit_panels` provides.
- The running integral restarts per term once its block is a window wide, so subtraction cancels only
  local magnitudes. Without this, a flux falling faster than E lost all relative accuracy in its
  tail, which small PGD modes and strongly leaking modes would hit.
- Every jump in a source or cross section gets a node pair (`node_pairs`). F is continuous, so a jump
  between nodes is smeared across a panel and makes the flux below it first order.

## Consequences

- **Balance is not round-off.** The Phase 1 plan expected discrete normalization to close balance to
  round-off. Collocation does not do that: the residual is truncation error, second order in panel
  width. In `test_balance_and_capture_converge_through_a_resonance` it is 0.4% of the source at 21
  points across the resonance and 3e-5 at 321, with capture errors of 7% and 0.06%. Keeping node
  values is what lets F follow a resonance, so we accept this.
- **Gate item 4 changes.** "Balance closes" now means a residual of at most 1e-4 of the source in
  every gate run. It is a necessary check only. Its ratio to rate error is not stable, it is signed,
  and interpolating a jump in a source hides lost strength from it. Gate rates are certified by grid
  refinement and by the twin.
- **The planned grid size may be too small.** The M1 review ran Na23, Fe56, U238 and O16 at 294 K from
  100 keV to 1 eV, elastic and capture only, on the library's full union grid (158,951 nodes). The
  residual was -2.4e-3 and the U238 capture error 0.24% against a ×16 grid. Subdividing panels ×2
  gave 0.06% at 318k nodes, and ×4 gave -1.1e-4 at 636k nodes. Even the unthinned library grid can
  therefore miss the gates, because it resolves σ and not F. Study 04 needs flux-driven node
  insertion and has to revisit the 500k figure in the speed gate. This was a one-off review script,
  so study 04 should reproduce it.
- Thresholds in `interpolate`d data need node pairs too. Their effect is not measured yet.
- The edge integral costs several exponentials per term per node, so it is a first target for the
  speed gate.
- **Anisotropic elastic scattering (M2)** adds 2f - 1 by direct sum to the isotropic recursion, down to
  the energy where a term's tables are within 1e-3 of flat. Panels must resolve forward peaks, at
  (peak of f) x h / window of 0.05 (`refine_for_angle`), which adds about 70,000 nodes for the SFR
  and PWR nuclides sampled, mostly above 100 keV.
- **Discrete normalization** is needed for this part, with f frozen at the node's energy, and only for
  windows wholly above the anisotropy cutoff. Quadrature error acts as a net gain per collision that
  compounds over lethargy.
- **The direct sum misses the speed gate** by about 800 times in study 02 (7742 s at 544k nodes,
  two terms), and cost grows with the square of the node count. It stays as the reference until M9 replaces it, which is now the phase's
  largest technical risk.
- **Numba's cache ignores callees in other files.** A cached kernel calling across modules keeps
  running the old callee after an edit, so such callers must not use `cache=True`.
