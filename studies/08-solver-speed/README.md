# Study 08: where the energy solver's time goes, and what exact changes recover

Status: in progress (Phase 1.5, the speed work after decision 012).

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
