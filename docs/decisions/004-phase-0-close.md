# 004: Phase 0 closes

Date: 2026-11-01

## Context

Phase 0 asked whether we can read and represent every piece of data the method needs, and reproduce
OpenMC's values. [Decision 003](003-union-grid.md) kept it open until the remaining readers existed.

## Decision

Close Phase 0 and start Phase 1. The data layer now reads, and tests against `openmc.data`:

- Cross sections, all non-redundant reactions, Q values and neutron yields.
- ν and χ for all 88 fissionable nuclides, including Maxwell, partial, total-mode and correlated forms.
- Scattering laws: every neutron law in all 556 nuclides reads, and 18 nuclides covering every form
  are compared table by table.
- S(α,β) for H in H2O, and unresolved-range probability tables.

The union grid size is known ([study 01](../../studies/01-union-grid/README.md)), and cross sections
between tabulated temperatures follow OpenMC's interpolation method.

## Consequences

Phase 1 inherits these limits:

- Tolerances are not chosen, and thinned grids do not certify ν, MT 5 yields or secondary laws.
- Only H in H2O thermal data is read. Other S(α,β) forms, such as coherent elastic, refuse by name.
- `read_fission` works only at tabulated temperatures, and OpenMC twins should stay there.
- Applicability selects among several laws and is not a neutron weight. Yields carry neutron counts.
- Readers return data but no kernels. Interpolating between incoming energies, frame conversion and
  Legendre moments are Phase 1 work, and each needs its own check.
- The Docker image builds, but tests have not been run inside it, and the devcontainer is untested.
