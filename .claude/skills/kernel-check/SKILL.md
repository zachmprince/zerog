---
name: kernel-check
description: ZeroG conventions for physics kernels. Use when writing or reviewing numerical code in src/zerog (units, energy ordering, lethargy weights, Numba-compilable layout, required analytic or openmc.data tests, no LWR-specific assumptions).
user-invocable: false
---

Checklist for any physics kernel in `src/zerog/`:

**Units and ordering**
- eV, cm, barns, atoms/barn-cm, kelvin everywhere. Convert at the I/O boundary only.
- Energy arrays run high to low. Lethargy is the integration variable, so quadrature weights are lethargy widths, not energy widths.

**Kernel form (GPU port must be mechanical)**
- Flat NumPy arrays in, arrays out. Numba-compilable with no Python objects, dicts, classes or strings inside.
- Batched layouts (leading batch axis) rather than per-item Python loops.
- Keep a plain-Python path as the oracle for a later C++/Kokkos port.

**Tests (required before a study uses the kernel)**
- At least one analytic check (e.g. pure scatterer gives 1/E flux, narrow/wide resonance limits, particle balance) or a comparison against `openmc.data` to round-off.
- Tests needing the nuclear library must skip cleanly when `OPENMC_CROSS_SECTIONS` is unset.

**Generality**
- Nothing LWR-specific in core code: no thermal-cutoff constants, square-lattice or water assumptions. SFR (fast spectrum, hexagonal, sodium) must work through the same code.
- Cross sections stay separable, Σ(x,E) = Σₖ Nₖ(x) σₖ(E, T(x)); do not hard-wire a term count.

**Solver interface**
- Krylov solves go through the thin interface (apply operator, apply preconditioner, solve); no direct SciPy or PETSc calls in physics code.

Never commit nuclear data or results larger than a few MB.
