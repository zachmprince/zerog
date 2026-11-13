"""An infinite homogeneous mix at one library temperature, assembled for the energy subproblem.

This is what studies 03 and 04 run: the grid, each nuclide's curves, inelastic, thermal and fission
kernels, and the operators, from a composition. Densities here are the physical ones; the
subproblem takes effective densities per term separately.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from zerog.data.neutron import MT_FISSION, MT_PARTIAL_FISSION, read_reactions
from zerog.energy.elastic import thin_angle
from zerog.energy.fission import build_fission, data_top, fission_nodes
from zerog.energy.grid import interpolate
from zerog.energy.inelastic import build_inelastic
from zerog.energy.subproblem import build_operators
from zerog.energy.terms import (
    apply_thermal,
    awr,
    load_term,
    thermal_grid_settings,
    thermal_required,
    thermal_spec,
    transport_grid,
)
from zerog.energy.thermal import build_thermal
from zerog.energy.urr import build_urr, urr_required

LIBRARY_BOTTOM = 1.0e-5  # eV, the bottom of ENDF/B-VIII.0's neutron grids
ORDER_WORKERS = 4
# Off by default: thinning angle tables to 1e-3 (absolute, in pdf units) saved nothing on study
# 03's SFR mix, and 3e-3 saved a fifth of a sweep for 1e-3 of the flux (M9). The anisotropic
# correction's cost is per node, not per cosine piece.
ANGLE_TOLERANCE = 0.0


@dataclass(frozen=True)
class Mix:
    """A mix's names, densities (atoms/b-cm), grid, curves and kernels."""

    names: list
    label: str
    densities: np.ndarray
    energy: np.ndarray
    coarse: np.ndarray
    top: float
    energy_min: float
    terms: list
    thermal: list  # ThermalTerm or None per nuclide
    ops: object
    fission: object
    timings: dict = field(default_factory=dict)  # seconds, filled by callers that time solves

    def fission_xs(self, i):
        """Nuclide i's fission cross section on the grid, partial reactions included."""
        data = read_reactions(self.names[i], self.label)
        selected = data.xs[np.isin(data.mt, (MT_FISSION, *MT_PARTIAL_FISSION))].sum(axis=0)
        return interpolate(self.energy, data.energy, selected)


def build_mix(
    densities,
    label,
    energy_min=None,
    sab=None,
    rtol=1e-3,
    subdivide=1,
    curves="all",
    required=(),
    light=None,
    cache=None,
    atol=1e-8,
    bands=0,
    orders=0,
    angle_tolerance=ANGLE_TOLERANCE,
    legendre=None,
):
    """Assemble a mix of {name: atoms/b-cm} at a library temperature label.

    energy_min defaults to the library's bottom. sab maps a nuclide to its S(alpha, beta) table.
    Thermal scattering applies to every nuclide whose free gas top lies above energy_min; light
    sets the top, in kT, for targets of awr <= 1 (terms.thermal_spec). rtol, subdivide and curves
    choose the grid (terms.transport_grid), and required energies are added to it. cache is a
    directory for inelastic and thermal kernels. bands, a power of two, turns probability tables on
    with that many band combinations (zerog.energy.urr); 0 leaves them off. orders adds elastic
    Legendre orders 1 .. orders (zerog.energy.moments), built one process per nuclide.
    angle_tolerance thins each elastic angle table's cosines (zerog.energy.elastic.thin_angle),
    the anisotropic correction's main cost; 0 keeps the library's tables. legendre, (order,
    tolerance) or None, is passed to build_operators: the truncated Legendre form of the
    anisotropic windows, an approximation, where each nuclide's tables allow it.
    """
    names = list(densities)
    density = np.array([densities[n] for n in names], dtype=float)
    energy_min = LIBRARY_BOTTOM if energy_min is None else energy_min
    sab = sab or {}
    nuclides = [(n, label) for n in names]
    top = data_top(nuclides)
    specs = [thermal_spec(n, label, sab.get(n), light if awr(n) <= 1.0 else None) for n in names]
    specs = [s if energy_min < s.free_top else None for s in specs]
    thermal = [s for s in specs if s is not None]
    needed = np.concatenate(
        [
            fission_nodes(nuclides, top, energy_min),
            np.asarray(required, dtype=float),
            *[thermal_required(s, energy_min) for s in thermal],
            urr_required(nuclides) if bands else np.empty(0),
        ]
    )
    energy, coarse = transport_grid(
        nuclides,
        rtol,
        atol,
        top,
        energy_min,
        subdivide,
        required=needed[(needed > energy_min) & (needed < top)],
        thermal=thermal_grid_settings(thermal) if thermal else None,
        curves=curves,
        densities=density,
    )
    terms = [load_term(n, label, energy) for n in names]
    terms = [
        t if s is None else apply_thermal(t, s, energy) for t, s in zip(terms, specs, strict=True)
    ]
    terms = [replace(t, angle=thin_angle(t.angle, angle_tolerance)) for t in terms]
    # Order rows first, while the process is small: forked workers copy what they touch.
    order_rows = _order_rows(names, terms, energy, coarse, orders) if orders else None
    kernels = build_inelastic([(n, label, awr(n)) for n in names], energy, coarse, cache=cache)
    if thermal:
        kernels = build_thermal(specs, energy, kernels, cache=cache)
    fission = build_fission(nuclides, energy, coarse, top)
    ops = build_operators(
        energy,
        np.array([t.total for t in terms]),
        np.array([t.elastic for t in terms]),
        [t.alpha for t in terms],
        [t.angle for t in terms],
        kernels,
        build_urr(nuclides, energy, bands) if bands else None,
        order_rows,
        legendre,
    )
    return Mix(names, label, density, energy, coarse, top, energy_min, terms, specs, ops, fission)


def _one_term_rows(term, a, energy, coarse, orders):
    from zerog.energy.moments import elastic_order_rows

    return elastic_order_rows(term.angle, a, energy, term.elastic, coarse, orders)


def _order_rows(names, terms, energy, coarse, orders):
    """Each term's elastic rows for orders 1 .. orders, a process per term where fork exists."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    work = [(t, awr(n), energy, coarse, orders) for n, t in zip(names, terms, strict=True)]
    # Compile the kernels here first, on a three-node coarse grid, so workers don't all compile.
    small = np.array([coarse[0], coarse[0] * 0.5, coarse[0] * 0.25])
    _one_term_rows(terms[0], awr(names[0]), energy, small, orders)
    if "fork" not in multiprocessing.get_all_start_methods():
        return [_one_term_rows(*w) for w in work]
    context = multiprocessing.get_context("fork")
    # Forked workers grow as they touch the parent's pages: twelve forked after the kernels were
    # built ran a 31 GB machine out of memory.
    with ProcessPoolExecutor(min(len(work), ORDER_WORKERS), mp_context=context) as pool:
        return list(pool.map(_one_term_rows, *zip(*work, strict=True)))
