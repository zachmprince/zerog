# 013: The energy solver is 5 to 10 times faster, with an opt-in Legendre form for anisotropy

Date: 2026-11-13

## Context

Decision 012 closed Phase 1 with its speed gate missed by 23 times: 231 s for the PWR's 500k-node
fixed-source solve against 10 s. Most of that was the anisotropic elastic correction. The kernel
builds took 6 to 7 minutes and the k solves up to 30 minutes. Study 08 asked how much of this
Python and Numba on one core could recover, first without changing any result, then with an
approximation held to an error budget. Its README has the profiles and every intermediate number.

## Decision

**Exact changes (on by default; results move by round-off, or by the eigen loop's tolerance in
k):**

- **Anisotropic window.** Each cosine knot's cut panel takes one quadrature, not two. The other
  side is the tree leaf less the cut (`elastic.cut_moments`). The old path stays as
  `anisotropy="tree"`.
- **The rest of the march:**
  - the isotropic edge integral in closed form;
  - no elastic work for terms without elastic scattering;
  - no anisotropy in marches that start below a term's anisotropy bottom;
  - `_lagged_source` skips hats with no lagged emission.
- **Sweeps.** The thermal block's GMRES stops at 1e-4 of its right-hand side, not 1e-10. The next
  sweep changes that right-hand side anyway.
- **Eigen loop.** `solve_k` takes one inelastic sweep per power iteration, carrying the lagged
  emission, and stops when k, the source and the sweep have all converged. That is 5 to 7 sweeps
  in place of 15. Results agree with the fully solved loop to its tolerances (1e-9 in k), not to
  round-off.
- **Build.** All terms' inelastic and thermal rows share one fork pool, in small interleaved
  shares. The per-term pools left O16 on 3 of 32 processes for 90 s. Rows are bit-identical.
- **Compile.** `_march` is cached, and a hash of `elastic.py` and `urr.py` drops stale code.

**The Legendre form (opt-in: `build_mix(..., legendre=(8, 1.0))`):**

- **The form.** Each centre-of-mass table is truncated at order L. A window is then O(L²)
  running moments, not O(knots) exact pieces.
- **Exactness.** It keeps a_0 = 1, so the kernel moves no neutrons, and polynomial tables give the
  exact answer to round-off.
- **Where it applies.** Up to each nuclide's switch energy (the cumulative-distribution tolerance;
  1.0 means everywhere) and for α ≥ 0.7. Lighter targets, such as H1, take it only below their
  anisotropy bottom.
- **Error budget, against the exact path:** k within 1 pcm, rates within 1e-4, flux in 1/20
  lethargy bins within 1e-4.
- **Measured:** k moved by at most 0.003 pcm, rates by 9e-7 and bins by 8e-5, all in the bins
  between 18 and 20 MeV. Pointwise, F above 10 MeV carries a ripple of up to 4%, but only 4e-4
  of the flux is there.

## Consequences

The gate runs of decision 012, rerun at this commit (logs in `studies/08-solver-speed/results/`,
gitignored):

| Gate | 012 | Exact now | Legendre (8, 1.0) |
| --- | --- | --- | --- |
| SFR, 837k: fixed-source solve | 456 s | 373 s | 44 s |
| SFR: k solve | 1792 s | 615 s | 94 s |
| PWR, 509k: fixed-source solve | 231 s | 147 s | 31 s |
| PWR: k solve | 573 s | 185 s | 46 s |
| SFR, 64 bands: k solve | 2061 s | 691 s | 191 s |
| Kernel builds (SFR, PWR) | 440 s, 362 s | 144 s, 112 s | (cached) |

- **The twins.**
  - k equals decision 012's to every printed digit in all three cases, in both forms: +11.1,
    +12.7 and +16.5 pcm.
  - Every gated rate and flux bin statistic is unchanged. The forms differ by at most 8.4e-7 in
    any rate (Na23 capture).
  - Probability tables work with the form.
- **Gate 5 is still missed:** 31 s against 10 s for the PWR with the Legendre form, and 147 s
  exactly. What remains is spread out:
  - At 244k SFR nodes, a 4 s march with the form spends 2.1 s on anisotropy, 1.2 s on the
    isotropic recursion and 0.8 s on inelastic transfer.
  - The PWR's solve is half thermal-block marches.
  - No single change of this kind closes the gap. The Phase 4 port remains the way to 10 s.
- **Phase 2 inherits:**
  - `solve_energy` with the same signature, plus `max_sweeps`;
  - `solve_k` with `sweeps_per_iteration`;
  - `build_operators` and `build_mix` with `legendre`.

  Phase 2 can take the Legendre form for speed and keep the exact form as its oracle.
- **Not run with the Legendre form:** Legendre orders above zero (study 06's pseudo-terms take the
  form's windows only through order 0).
- **The form's limits.**
  - Its sweeps stop at 1e-10, since round-off in its moments sits near 1e-12 of F.
  - Orders stop at 10.
  - The actinides' forward peaks would keep the exact form above 3 to 5 MeV at any tolerance under
    1, so "everywhere" is what makes it fast.
