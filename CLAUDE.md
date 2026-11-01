# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ZeroG is a research code for a deterministic full-core neutron transport solver with **no multigroup library**. The angular flux is a PGD (proper generalized decomposition) expansion ψ(x,Ω,E) ≈ Σₘ Xₘ(x,Ω) εₘ(E), with the energy modes εₘ on a pointwise grid built directly from evaluated nuclear data. It starts with PWR benchmarks but must assume nothing LWR-specific (SFR cases are an early second target: no thermal spectrum, square lattices or water assumptions in core code).

Phase 0 is closed (`docs/decisions/004-phase-0-close.md` lists the limits Phase 1 inherits): `src/zerog/data/` holds the nuclear data readers and union grid, and the other subpackages are still empty. The development plan lives in `Pointwise-Energy PGD Transport Development Plan.md` in the repo root. It is deliberately **untracked and not in `.gitignore`**; never `git add -A` blindly, and never commit it.

## Commands

Development happens in WSL (Ubuntu), in `~/projects/zerog` on the Linux filesystem. Environments are managed by `pixi` (conda-forge), with all dependencies declared in `[tool.pixi.*]` in `pyproject.toml`; `[project] dependencies` is intentionally empty. Python is pinned to 3.13 because conda-forge's `openmc` is only built for it. Add dependencies with `pixi add`, not `pip`/`uv`.

```bash
pixi install                     # create/update the environment
pixi run pytest                  # all tests (also: pixi run test)
pixi run pytest tests/test_x.py::test_name   # single test
pixi run ruff check              # lint (also: pixi run lint)
pixi run ruff format             # format (also: pixi run fmt)
```

Nuclear data comes from the `OPENMC_CROSS_SECTIONS` env var (ENDF/B-VIII.0 HDF5 library) and is never committed. GitHub access from WSL is HTTPS via `gh` (installed in `~/.local/bin`).

## Architecture (planned)

One package, `zerog`, with a module per PGD subproblem:

- `data/`: OpenMC HDF5 reader, union energy grid, temperature interpolation, S(α,β), unresolved-range probability tables
- `energy/`: 0D slowing-down/thermalization solve (the PGD energy subproblem), kernel Legendre moments
- `space/`: 1D reference transport, DG-FEM S_N (the PGD spatial subproblem, one-group)
- `pgd/`: driver, subproblem coupling, update step, eigenvalue iteration
- `mesh/`: Gmsh/meshio input, composition and temperature fields per element
- `analysis/`: SVD rank tools, error metrics

Outside the package: `studies/NN-name/` (each answers one question; README with question, method, result, decision), `benchmarks/` (OpenMC twin models and cached reference results), `docs/decisions/` (one short record per decision).

Key design constraints that span modules:

- The energy subproblem takes effective densities per nuclide-temperature term (removal and each scattering order), a streaming coefficient and a source, and returns ε(E). Design it with that interface from the start.
- Cross sections are assumed separable, Σ(x,E) = Σₖ Nₖ(x) σₖ(E,T(x)); the number of terms is the key open risk.
- Spatial subproblems may have effective scattering ratios > 1 or negative, so Krylov solves go through a thin interface (apply operator, apply preconditioner, solve) so SciPy and petsc4py can be swapped without touching physics code.
- Python/Numba is a prototype; profiled hot paths are later ported to a C++/Kokkos core, with the Python version kept as the test oracle.

## Conventions

- **Units everywhere:** eV, cm, barns, atoms/barn-cm, kelvin. Energy arrays run high to low; lethargy is the integration variable.
- **Kernels:** flat NumPy arrays and Numba-compilable functions with no Python objects inside (GPU port should be mechanical). Keep layouts array-based and batched.
- **Tests:** every physics kernel needs an analytic check or a comparison against `openmc.data` before use in a study.
- **Validation:** each study gets an OpenMC twin with the same geometry, data and settings. Turn OpenMC's unresolved-range probability tables off until we implement ours.
- **Data and results:** never commit nuclear data; results larger than a few MB stay out of git.
- Ruff line length is 100.
