# ZeroG

Pointwise-energy PGD transport solver (Python package `zerog`). The angular flux is expanded as ψ(x,Ω,E) ≈ Σₘ Xₘ(x,Ω) εₘ(E), with the energy modes on a pointwise grid built directly from evaluated nuclear data, so no multigroup library is needed.

## Setup

Development targets Linux (WSL Ubuntu on Windows). Environments are managed with [pixi](https://pixi.sh); it installs everything from conda-forge, including OpenMC, which has no PyPI package.

```bash
pixi install
pixi run test
```

Other tasks:

```bash
pixi run lint    # ruff check
pixi run fmt     # ruff format
```

Python is pinned to 3.13 because conda-forge's `openmc` is only built for that version. Add dependencies with `pixi add`.

## Nuclear data

Set `OPENMC_CROSS_SECTIONS` to the path of an OpenMC HDF5 library (ENDF/B-VIII.0 to start; see [openmc.org/data](https://openmc.org/data)). Nuclear data is never committed.
