"""Build subproblem terms from the library. Each is one nuclide at one tabulated temperature."""

from dataclasses import dataclass

import numpy as np

from zerog.data.grid import major_curves, thin_indices, thin_reactions
from zerog.data.neutron import MT_CAPTURE, MT_ELASTIC, read_reactions
from zerog.data.scatter import read_scattering_laws
from zerog.energy.elastic import refine_for_angle
from zerog.energy.grid import assemble_grid, interpolate, kinematic_alpha, node_pairs
from zerog.energy.inelastic import coarse_grid
from zerog.energy.thermal import (
    FREE_GAS_THRESHOLD,
    LIGHT_FREE_GAS_THRESHOLD,
    ThermalTerm,
    emission_curve,
    free_gas_range,
    sab_table,
    thermal_coarse,
)


@dataclass(frozen=True)
class TermData:
    """One term's curves on a grid, in barns, and its elastic kinematics."""

    name: str
    temperature: str
    alpha: float
    total: np.ndarray
    elastic: np.ndarray
    capture: np.ndarray
    angle: object


def awr(name):
    """A nuclide's atomic weight ratio, from the library."""
    return _awr(name)


def _awr(name):
    import h5py

    from zerog.data.library import nuclide_path

    with h5py.File(nuclide_path(name), "r") as f:
        return float(f[name].attrs["atomic_weight_ratio"])


# Study 02 measured the coarse spacing on U238 + O16 from 14 MeV: the 7.68 MeV flux bin is off by
# 2.3% at 0.005, 0.9% at 0.0025, 0.25% at 0.00125 and 0.22% at 0.000625, against 0.14% twin noise.
COARSE_DU = 0.00125


def transport_grid(
    nuclides,
    rtol,
    atol,
    energy_max,
    energy_min,
    subdivide=1,
    coarse_du=COARSE_DU,
    required=(),
    thermal=None,
    curves="all",
    densities=None,
):
    """A fine grid as elastic_grid, holding a coarse grid for inelastic transfer. Returns both.

    required energies, such as zerog.energy.fission.fission_nodes, become fine nodes. thermal,
    (reach, kT, awr) from thermal_grid_settings, widens coarse hats below reach as
    zerog.energy.thermal.thermal_coarse does.
    """
    fine = elastic_grid(
        nuclides, rtol, atol, energy_max, energy_min, subdivide, required, curves, densities
    )
    coarse = coarse_grid(fine, coarse_du, _thresholds(nuclides, energy_max, energy_min))
    if thermal is not None:
        thresholds = _thresholds(nuclides, energy_max, energy_min)
        coarse = thermal_coarse(coarse, *thermal, coarse_du, required=thresholds)
    return np.unique(np.concatenate([fine, coarse]))[::-1], coarse


def _thresholds(nuclides, energy_max, energy_min):
    out = []
    for name, label in nuclides:
        data = read_reactions(name, label)
        keep = (data.threshold < energy_max) & (data.threshold > energy_min)
        out.extend(data.threshold[keep].tolist())
    return np.array(out)


GRID_CURVES = ("all", "major", "macroscopic")


def thinned_grids(nuclides, rtol, atol, energy_max, curves="all", densities=None):
    """Thinned grids to merge, by which curves they hold within rtol * |curve| + atol:

    - "all": every non-redundant reaction of each nuclide, valid for any non-negative densities.
    - "major": each nuclide's total, elastic, capture and fission (zerog.data.grid.major_curves).
    - "macroscopic": the mix's total, elastic, capture and fission, summed with densities
      (atoms/b-cm, one per nuclide), on the union of the nuclides' own points. Valid for that
      composition only, and atol is then in 1/cm.

    Thresholds are kept in every case, so reactions stay zero below them.
    """
    data = [read_reactions(*key) for key in nuclides]
    if curves == "all":
        return [thin_reactions(d, rtol, atol, energy_max) for d in data]
    if curves == "major":
        return [_thin_major(d, rtol, atol, energy_max) for d in data]
    if curves != "macroscopic":
        raise ValueError(f"curves must be one of {GRID_CURVES}, not {curves!r}")
    if densities is None or len(densities) != len(data):
        raise ValueError("a macroscopic grid needs one density per nuclide")
    energy = np.unique(np.concatenate([d.energy for d in data]))[::-1]
    energy = energy[energy <= energy_max]
    total = np.zeros((4, energy.shape[0]))
    for d, n in zip(data, densities, strict=True):
        major = major_curves(d)
        for c in range(4):
            total[c] += n * interpolate(energy, d.energy, major[c])
    required = np.zeros(energy.shape[0], dtype=np.bool_)
    thresholds = np.concatenate([d.threshold for d in data])
    thresholds = thresholds[thresholds <= energy_max]
    required[np.clip(np.searchsorted(-energy, -thresholds), 0, energy.shape[0] - 1)] = True
    return [energy[thin_indices(energy, total, required, rtol, atol)]]


def _thin_major(data, rtol, atol, energy_max):
    start = np.searchsorted(-data.energy, -energy_max)
    energy = data.energy[start:]
    required = np.zeros(energy.shape[0], dtype=np.bool_)
    thresholds = data.threshold[data.threshold <= energy_max]
    required[np.searchsorted(-energy, -thresholds)] = True
    major = major_curves(data)[:, start:]
    return energy[thin_indices(energy, np.ascontiguousarray(major), required, rtol, atol)]


def elastic_grid(
    nuclides,
    rtol,
    atol,
    energy_max,
    energy_min,
    subdivide=1,
    required=(),
    curves="all",
    densities=None,
):
    """A grid for elastic scattering between two energies, for (name, label) pairs.

    It holds node pairs at each term's first window edge below energy_max, where a point source
    there jumps, and resolves forward peaks. subdivide then splits every panel, including those
    capped by a peak, since the thinned grid resolves cross sections and not F. curves and
    densities choose the thinned grids, as thinned_grids describes.
    """
    grids = thinned_grids(nuclides, rtol, atol, energy_max, curves, densities)
    alpha = [kinematic_alpha(_awr(name)) for name, _ in nuclides]
    angle = [_elastic_angle(name) for name, _ in nuclides]
    jumps = node_pairs([a * energy_max for a in alpha if a * energy_max > energy_min])
    max_du = 0.9 * min(-np.log(a) for a in alpha)
    extra = np.concatenate([jumps, np.asarray(required, dtype=float)])
    energy = assemble_grid(grids, energy_max, energy_min, max_du, required=extra)
    energy = refine_for_angle(energy, alpha, angle)
    if subdivide > 1:
        widths = np.log(energy[:-1] / energy[1:])
        fractions = np.arange(subdivide)[None, :] / subdivide
        energy = np.append(
            (energy[:-1, None] * np.exp(-widths[:, None] * fractions)).ravel(), energy[-1]
        )
    return energy


def _elastic_angle(name):
    law = next(r for r in read_scattering_laws(name) if r.mt == MT_ELASTIC)
    return law.laws[0].angle


def load_term(name, temperature, energy):
    """A nuclide's total, elastic and capture on a grid, with its angle tables."""
    data = read_reactions(name, temperature)
    if energy[0] > data.energy[0]:
        raise ValueError(f"grid reaches {energy[0]} eV, above {name}'s top {data.energy[0]} eV")

    def curve(mts):
        selected = data.xs[np.isin(data.mt, mts)].sum(axis=0)
        return interpolate(energy, data.energy, selected)

    return TermData(
        name=name,
        temperature=temperature,
        alpha=kinematic_alpha(_awr(name)),
        total=interpolate(energy, data.energy, data.xs.sum(axis=0)),
        elastic=curve((MT_ELASTIC,)),
        capture=curve((MT_CAPTURE,)),
        angle=_elastic_angle(name),
    )


def library_kT(name, label):
    """A nuclide's kT at a library temperature label, in eV."""
    import h5py

    from zerog.data.library import nuclide_path

    with h5py.File(nuclide_path(name), "r") as f:
        return float(f[name]["kTs"][label][()])


def thermal_spec(name, label, sab=None, threshold=None):
    """The ThermalTerm of a nuclide at a library temperature: free gas below threshold * kT, and
    the S(alpha, beta) table sab (such as 'c_H_in_H2O', at the same label) below its energy_max.

    threshold defaults to OpenMC's 400, or LIGHT_FREE_GAS_THRESHOLD for awr <= 1.
    """
    from zerog.data.thermal import read_thermal

    if threshold is None:
        threshold = LIGHT_FREE_GAS_THRESHOLD if _awr(name) <= 1.0 else FREE_GAS_THRESHOLD

    data = read_reactions(name, label)
    kT = library_kT(name, label)
    elastic = data.xs[list(data.mt).index(MT_ELASTIC)]
    if sab is None:
        return ThermalTerm(_awr(name), kT, threshold * kT, data.energy, elastic)
    table = read_thermal(sab, label)
    if threshold * kT <= table.energy_max:
        raise ValueError(
            f"{name}'s free gas top {threshold * kT} eV is at or below {sab}'s energy_max "
            f"{table.energy_max} eV: free gas would never apply"
        )
    if name not in table.nuclides:
        raise ValueError(f"{sab} is for {table.nuclides}, not {name}")
    if abs(table.kT - kT) > 1e-6 * kT:
        raise ValueError(f"{sab} at {label} has kT {table.kT}, {name} has {kT}")
    return ThermalTerm(
        awr=_awr(name),
        kT=kT,
        free_top=threshold * kT,
        curve_energy=data.energy,
        elastic=elastic,
        sab_top=table.energy_max,
        sab=sab_table(table),
        sab_energy=table.energy,
        sab_xs=table.xs,
    )


def thermal_required(spec, energy_min):
    """Fine nodes a thermal term needs: node pairs at its tops, where cross sections jump, and its
    S(alpha, beta) cross section's own energies."""
    # Free gas stops below free_top (OpenMC takes E >= 400 kT at rest), the table at sab_top
    # inclusive (E <= energy_max), so that pair sits just above it.
    out = [node_pairs([spec.free_top])]
    if spec.sab is not None:
        out.append(np.array([spec.sab_top * (1.0 + 1e-10), spec.sab_top]))
    if spec.sab is not None:
        e = spec.sab_energy
        out.append(e[(e < spec.sab_top) & (e > energy_min)])
    return np.concatenate(out)


def thermal_grid_settings(specs):
    """(reach, kT, awr) for transport_grid: thermal rows reach up to the free gas kernel's top
    from the highest thermal top, and the heaviest free gas target has the narrowest kernel."""
    specs = [s for s in specs if s is not None]
    reach = max(free_gas_range(s.free_top, s.awr, s.kT)[1] for s in specs)
    return reach, max(s.kT for s in specs), max(s.awr for s in specs)


def apply_thermal(term, spec, energy):
    """A TermData with thermal scattering: its elastic is the target-at-rest part, above free_top,
    and its total takes the S(alpha, beta) cross section in place of elastic below sab_top."""
    at_rest = np.where(energy >= spec.free_top, term.elastic, 0.0)
    total = term.total
    if spec.sab is not None:
        bound = np.interp(energy[::-1], spec.sab_energy[::-1], spec.sab_xs[::-1])[::-1]
        below = energy <= spec.sab_top
        total = np.where(below, total - term.elastic + bound, total)
    # Check the split: at-rest plus thermal emission is the elastic, or the table, everywhere.
    scattering = at_rest + emission_curve(spec, energy)
    expected = term.elastic
    if spec.sab is not None:
        expected = np.where(energy <= spec.sab_top, bound, term.elastic)
    if not np.allclose(scattering, expected, rtol=1e-12, atol=0.0):
        raise ValueError(f"{term.name}: thermal and at-rest scattering do not add to the whole")
    return TermData(
        name=term.name,
        temperature=term.temperature,
        alpha=term.alpha,
        total=total,
        elastic=at_rest,
        capture=term.capture,
        angle=term.angle,
    )
