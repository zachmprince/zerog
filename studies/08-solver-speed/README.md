# Study 08: where the energy solver's time goes, and what exact changes recover

Status: done (Phase 1.5, the speed work after decision 012). The decision is
[013](../../docs/decisions/013-energy-solver-speed.md); the gate reruns' logs are in `results/`.

## Question

Decision 012 closed Phase 1 with the speed gate missed by 23 times: 231 s for a 500k-node
fixed-source PWR solve against 10 s. Without changing any result beyond round-off, with no
hardware acceleration and no C++, how much of that can the energy solver, its eigenvalue loop and
its kernel build recover, and what is the floor?

## Method

`scripts/bench.py` (its docstring has the usage) times study 03's mixes at a 1% grid (SFR 244,110
nodes, PWR 173,295), stage by stage. It saves F, k and the timings as a baseline and diffs later
runs against it. Kernels come from a scratch `ZEROG_KERNEL_CACHE`, deleted afterwards.

```bash
cd studies/08-solver-speed/scripts
```

```bash
pixi run python -u bench.py case sfr --k --save base-sfr
```

```bash
pixi run python -u bench.py case sfr --k --compare base-sfr
```

`bench.py kernel U238 Fe56` marches one nuclide's elastic scattering alone, to time the anisotropic
correction per node. The per-part splits below came from a scratch copy of
`elastic.separable_correction` with counters and switches; it is not kept.

## Result

### Baseline (code at decision 012)

| Case | March | of it anisotropic | Fixed-source solve | k solve | Build |
| --- | --- | --- | --- | --- | --- |
| SFR, 244k | 43.1 s | 42.3 s | 172.7 s, 4 sweeps | 663 s, 15 sweeps | 281 s (inelastic 268 s) |
| PWR, 173k | 11.3 s | 10.6 s | 103.7 s, 6 sweeps + 55 block marches | 247 s, 15 + 108 | 254 s (inelastic 204, thermal 44) |

- **Anisotropic elastic is the sweep.** On the SFR grid each node's window crosses 90 to 130
  separable pieces per nuclide, at 15 to 24 µs per node per nuclide. For each piece:
  - 1.5 to 2 partial-panel Gauss quadratures of f, about half the cost;
  - 1 to 4.5 tree nodes;
  - a knot search, a galloping panel search and a log.

  Fe56 and O16 also cross 40 to 45 table intervals per node. Thinning tables cuts U238's cost
  with its knots, but Fe56's only to about 8 µs, since its intervals carry the rest.
- **The build is a load-balance problem.** Sampled serially, the SFR's inelastic rows cost about
  1,900 CPU-s: Pu240 456, O16 272, U235 253 and U238 240, with Na23, Cr52 and Pu241 under 10
  each. That is 60 s on 32 processes, against 268 s of wall time. Each nuclide got its own pool,
  sized by its emitting hats, so O16 (938 hats, 272 s) ran on 3 processes for about 90 s.

### Exact changes so far

- **Shared cuts** (`anisotropy="separable"`, the default). A knot inside a panel above the
  unknown's costs one quadrature, not two. The cut's top takes `elastic.cut_moments`, and the
  segment below takes the panel's tree leaf less that top. Both are centred at the panel's top and
  summed through `_node_value` with each segment's own f.
  - Measured on the SFR grid in isolation: 1.35 to 1.45 times faster than the tree path, which
    stays as `anisotropy="tree"`.
  - Results agree with the tree path to 4e-13 of the window integral, and the SFR flux to 4e-14.
- **Closed-form edge integral.** The isotropic window's partial panel is now exact in closed form:
  one log and one expm1, with series below s = 0.1. It is 40 to 70% faster per call, and its
  oracle's Gauss rule is off by 6e-13 on 0.6-wide panels.
- **No elastic work for terms without elastic scattering.** This covers the pseudo-terms of orders
  above zero.
- **One sweep per power iteration** (`solve_k(..., sweeps_per_iteration=1)`). The lagged emission
  carries over between iterations, which stop once k, the source shape and the sweep have all
  converged.
  - SFR: 6 sweeps instead of 15, k solve 663 s → 246 s.
  - k changes by 1.5e-9 pcm and the flux by 1.8e-12.
- **One pool for every term's inelastic rows.** Hats go in shares of 32, largest terms first; a
  test checks the rows are bit-identical to a serial build.

### After these changes

The machine was shared with test runs, so march times are a little pessimistic.

| Case | March | Fixed-source solve | k solve | Build |
| --- | --- | --- | --- | --- |
| SFR, 244k | 41.2 s (was 43.1) | 141.5 s (172.7) | 226 s, 6 sweeps (663, 15) | 139 s (281) |
| PWR, 173k | 9.4 s (11.3) | 89.8 s (103.7) | 106 s, 7 + 61 (247, 15 + 108) | 112 s (254) |

- **Fixed-source flux:** agrees with the baseline to 4.2e-14 (SFR) and 4.5e-14 (PWR).
- **k:** agrees to 1.5e-9 pcm and 8e-8 pcm, with the flux to 1.8e-12 and 1.3e-11. The one-sweep
  iteration stops at a different iterate of the same fixed point, so these sit inside the eigen
  loop's own tolerances, not at round-off.
- **Build:** the inelastic rows went from 268 s to 128 s (SFR) and from 204 s to 76 s (PWR), and
  the PWR's thermal rows from 44 s to 32 s.

## Decision

The march, which is what the speed gate measures, improved by only 5 to 15%:

- On the real grid most of the anisotropic cost is enumerating pieces and querying the tree, not
  the cut quadratures that the shared cuts remove.
- Exact integration is O(knots in the window) per node per nuclide, 90 to 130 pieces at about
  100 ns each. Even at 10 to 20 ns per piece, a 500k-node SFR sweep would take 6 to 12 s.
- The 10 s gate for a whole fixed-source solve is therefore out of reach without changing the
  method.

What the remaining exact work could still do: precomputed piece tables and cached knot pointers
(about 10%, measured), and a sliding aggregate in place of the tree query (perhaps 1.3 to 1.5
times, unmeasured).

## Part 2: the Legendre form, an approximation with an error budget

### Question

Once the exact form is set aside, how much faster can the anisotropic window get, and at what
error against the exact path?

### Method

`subproblem.build_operators(..., legendre=(order, tolerance))`, also `build_mix(legendre=...)`.

- **The form.** Each centre-of-mass table is truncated at Legendre order L. Between tables it
  stays linear in E', as the tables mix.
- **What a node costs.** (2 f - 1) / E' is then a polynomial in the cosine x = c E / E' + d, so a
  window is a sum of O(L^2) running moments, N_nm = ∫ σF b_n v^m / E' du.
  - The moments restart per block of one window width, as the isotropic recursion's sums do.
  - Each block's variable is the cosine seen from its bottom, v, so x = ρ v + τ.
  - A node costs two panel evaluations and three combinations, however many knots or table
    energies its window holds.
- **Exact properties.** Truncation keeps a_0 = 1, so the kernel still moves no neutrons. Tables
  that are polynomials in μ (to order L) reproduce the exact path to round-off (tested).
- **Where it applies.** Windows wholly below a nuclide's switch energy take the form: the highest
  table energy under which every table's cumulative distribution in μ is within the tolerance of
  its truncation. Above the switch, and for targets lighter than α = 0.7, the exact form stays.
- **Budget, against the exact path on the same grid:**
  - k within 1 pcm;
  - every nuclide's elastic, capture and fission rate within 1e-4;
  - F in bins of 1/20 lethargy within 1e-4, for bins holding more than 1e-6 of the largest.

Choices made on the way:

- **Cumulative tolerance, not pointwise.** A pointwise bound on the pdf kept the form below about
  200 keV at any order. Lin-lin tables have kinks no polynomial follows, though they move little.
- **Moments centred on the cosine (powers of μ, not of μ − 1).** About μ = 1, round-off grew by
  P_8(5), about 2e7. The march then carried noise near 1e-12 of F, and the 1e-12 sweep test took
  18 to 22 sweeps.
  - Centred on the cosine, the growth is about 4e5.
  - The form's sweeps stop at 1e-10, still six orders under its truncation error.
  - Orders stop at 10.
- **Light targets keep the exact form.** Hydrogen's α is 1.7e-7, not 0, and the map's ρ then
  reached 1/α. The first PWR run produced NaN.

### Result

On study 03's mixes at 1% (code at this commit; "exact" is the default path after part 1):

| Case | Form | March | Fixed-source solve | k solve | k − exact | Worst rate | Worst bin | Worst node |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFR | exact | 41 s | 142 s, 4 sweeps | 226 s, 6 | | | | |
| SFR | L 8, 1e-3 | 23.1 s | 92.6 s | 127 s | +0.002 pcm | 9e-7 | 1.0e-5 | 1.4e-4 |
| SFR | L 8, 1e-2 | 17.9 s | 71.8 s | 100 s | -6e-5 pcm | 9e-7 | 7.7e-5 | 5.7e-4 |
| SFR | L 8, everywhere | 4.0 s | 16.1 s | 30.4 s | -0.003 pcm | 9e-7 | 5.7e-5 | 3.6e-2 |
| PWR | exact | 9.4 s | 90 s, 6 + 55 block | 106 s | | | | |
| PWR | L 8, 1e-2 | 6.8 s | 68.0 s | 77 s | +9e-4 pcm | 2e-7 | 6.3e-5 | 3.9e-4 |
| PWR | L 8, everywhere | 2.6 s | 48.5 s, 5 + 44 block | 61 s | +5e-4 pcm | 2e-7 | 6.4e-5 | 3.8e-2 |

"Everywhere" is a tolerance of 1, which every table meets, so no window keeps the exact form.

- **Every setting meets the budget.**
  - k moves by under 0.01 pcm and rates by under 1e-6.
  - The worst bin is 6e-5 to 8e-5, always between 18 and 20 MeV.
- **The pointwise error.** At L = 8 everywhere it is a ripple from truncated forward peaks, within
  4% above 10 MeV, where 4e-4 of the flux lies. It is under 7e-4 below 10 MeV and under 7e-5
  below 1 MeV.
- **Where the exact form remains.** With a switch, the actinides' forward peaks keep it above 3 to
  5 MeV, which no order up to 10 meets at these tolerances. The SFR grid has 50,000 nodes above
  5 MeV, so that is most of what is left of the march.
- **Per node,** on a 495k-node lethargy grid: 1.1 to 1.5 µs per nuclide, against 10 to 16 µs
  exact.

### Decision (part 2)

- **Recommended setting: `legendre=(8, 1.0)`,** the form everywhere it applies.
  - It meets the budget with a margin of 1.6 on the worst bin.
  - It makes the SFR's march 10 times faster and its fixed-source solve 11 times.
- **The exact form stays the default** and the oracle.
- **What remains of the gate.** At 500k nodes the SFR's fixed-source solve would take about 33 s
  over its 4 sweeps. Of a 4.0 s march, anisotropy is now 2.1 s, and the isotropic recursion (1.2)
  and inelastic transfer (0.8) the rest, so no single part closes the gap. For the PWR, most of
  the solve is the thermal block's GMRES marches.
- **Not yet run with the form (at the time of part 2):** probability tables (study 05's 64-band
  case; since rerun, decision 013) and orders above
  zero.

## Part 3: the thermal block and the march's fixed costs

On the PWR with the Legendre form, the fixed-source solve was 48.5 s: 5 sweeps of a 2.6 s march
and 44 marches of the thermal block at 0.7 s each. Profiled per march (PWR at 1%, 173k nodes,
of which 30k are in the block):

- **Anisotropy in block marches (exact).** A term whose anisotropy bottom lies above a march's
  start node skips anisotropy there, since F is zero on every panel its windows take.
- **Lagged source (exact).** `_lagged_source` skips hats with no lagged emission, which in a block
  march is every hat above the block.
- **Light targets (H1) take the Legendre form below their bottom node.** This is not exact, but it
  stays inside the Legendre budget.

Together they took a full march from 2.58 s to 1.51 s, and a block march from 0.70 s to 0.24 s.

- **H1's exact anisotropy was 1.1 s of the 2.6 s march.** Below its bottom (1.1 MeV), every
  window runs from the bottom to its edge.
  - Moments summed once from the grid's top, with v = E_bottom / E', serve every node there.
  - The cosine is x = (c E / E_bottom) v + d, with |ρ v| + |τ| under 3, however light the target.
  - With linear tables this matches the exact form to round-off (tested).
- **The block's GMRES stopped at 1e-10 of its right-hand side in every sweep.** But the next sweep
  changes that right-hand side anyway.

| `BLOCK_RTOL` | Fixed-source solve | Sweeps | Block marches | Flux vs 1e-10 |
| --- | --- | --- | --- | --- |
| 1e-10 | 24.4 s | 5 | 44 | |
| 1e-4 (now) | 19.1 s | 5 | 32 | 7e-15 |
| 1e-3 | 18.1 s | 5 | 28 | 1e-12 |
| 1e-2 | 16.5 s | 5 | 24 | 3e-11 |
| 1e-1 | 30.6 s | 10 | 38 | 3e-11 |

- **The exact path took 72 s, against 90 s before these changes,** and matches the baseline to
  4.5e-14.
- **`_march` is now cached:** its first call in a process takes 0.3 s instead of 29 s.
  - Numba's cache notices only changes to subproblem.py, so `_forget_stale_march` hashes
    elastic.py and urr.py, whose kernels it inlines or calls.
  - When the hash changes, it deletes the cached code; a test edit forced the recompile as it
    should.
