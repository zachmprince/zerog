"""Build subproblem terms from the library. Each is one nuclide at one tabulated temperature."""

from dataclasses import dataclass

import numpy as np

from zerog.data.grid import thin_reactions
from zerog.data.neutron import MT_CAPTURE, MT_ELASTIC, read_reactions
from zerog.data.scatter import read_scattering_laws
from zerog.energy.elastic import refine_for_angle
from zerog.energy.grid import assemble_grid, interpolate, kinematic_alpha, node_pairs


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


def _awr(name):
    import h5py

    from zerog.data.library import nuclide_path

    with h5py.File(nuclide_path(name), "r") as f:
        return float(f[name].attrs["atomic_weight_ratio"])


def elastic_grid(nuclides, rtol, atol, energy_max, energy_min, subdivide=1):
    """A grid for elastic scattering between two energies, for (name, label) pairs.

    It holds node pairs at each term's first window edge below energy_max, where a point source
    there jumps, and resolves forward peaks. subdivide then splits every panel, including those
    capped by a peak, since the thinned grid resolves cross sections and not F.
    """
    data = {key: read_reactions(*key) for key in nuclides}
    grids = [thin_reactions(curves, rtol, atol, energy_max) for curves in data.values()]
    alpha = [kinematic_alpha(_awr(name)) for name, _ in nuclides]
    angle = [_elastic_angle(name) for name, _ in nuclides]
    jumps = node_pairs([a * energy_max for a in alpha if a * energy_max > energy_min])
    max_du = 0.9 * min(-np.log(a) for a in alpha)
    energy = assemble_grid(grids, energy_max, energy_min, max_du, required=jumps)
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
