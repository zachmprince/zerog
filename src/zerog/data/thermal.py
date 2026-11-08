"""Read incoherent inelastic thermal scattering from OpenMC HDF5 files.

The library stores energies low to high; everything returned here runs high to low. Values are
otherwise as stored: eV, barns per atom of the table's nuclides (possibly a compound average, as
in c_C6H6), and per-eV spectra. That includes roundoff-sized negative pdf values, around -1e-27.
"""

from dataclasses import dataclass

import h5py
import numpy as np

from zerog.data.library import nuclide_path
from zerog.data.neutron import TabularSpectrum

RULE_LIN_LIN = 2
RULE_DISCRETE = 0


@dataclass(frozen=True)
class ThermalData:
    """One bound material at one tabulated temperature, for incoming energies up to energy[0].

    xs is lin-lin on energy. spectrum holds E -> E' as a lin-lin TabularSpectrum, so its conventions
    apply. Outgoing energies can exceed energy[0]: H in H2O reaches 11.55 eV from a 10 eV table,
    with up to 0.9% of the probability above 10 eV.

    Each outgoing point, flat index j across all of spectrum's tables, owns the cosines
    mu_offsets[j]:mu_offsets[j + 1]. These are discrete lines, not a density: mu_probability is a
    probability per line, not per unit mu. Cosines are not energies, so they run low to high within
    a point. mu_cdf[k] is the probability of mu <= mu[k], including that line, so it starts above 0
    and ends at 1, unlike spectrum.cdf.

    temperature is the library's label, not a value ('294K' is 293.6 K), so use kT for physics.
    energy_max is where OpenMC stops using the table (E <= energy_max), which need not be the top
    of xs.
    """

    name: str
    awr: float
    nuclides: tuple[str, ...]
    temperature: str
    kT: float  # eV
    energy_max: float  # eV
    energy: np.ndarray
    xs: np.ndarray
    spectrum: TabularSpectrum
    mu_offsets: np.ndarray
    mu: np.ndarray
    mu_probability: np.ndarray
    mu_cdf: np.ndarray


def _read_xs(dataset):
    """Read a lin-lin cross section, refusing forms that this reader would misrepresent."""
    kind = dataset.attrs["type"].decode()
    if kind != "Tabulated1D":
        raise NotImplementedError(f"inelastic cross section of type {kind}")
    rules = dataset.attrs["interpolation"]
    if not np.all(rules == RULE_LIN_LIN):
        raise NotImplementedError(f"inelastic cross section with interpolation {rules.tolist()}")
    return dataset[0, :][::-1].copy(), dataset[1, :][::-1].copy()


def read_thermal(name, temperature):
    """Read a thermal material at a library temperature: read_thermal('c_H_in_H2O', '294K').

    Raises NotImplementedError for forms not supported yet: any elastic scattering (coherent,
    incoherent or mixed), and inelastic scattering that is not lin-lin in E and E' with discrete
    cosines.
    """
    with h5py.File(nuclide_path(name, "thermal"), "r") as f:
        group = f[name]
        if temperature not in group["kTs"]:
            raise KeyError(f"{name} has no data at {temperature}: {sorted(group['kTs'])}")

        at_temperature = group[temperature]
        if "elastic" in at_temperature:
            kind = at_temperature["elastic"]["distribution"].attrs["type"].decode()
            raise NotImplementedError(f"{name} has elastic scattering of type {kind}")

        inelastic = at_temperature["inelastic"]
        distribution = inelastic["distribution"]
        kind = distribution.attrs["type"].decode()
        if kind != "incoherent_inelastic":
            raise NotImplementedError(f"{name} has inelastic scattering of type {kind}")

        tables = distribution["energy_out"]
        if np.any(tables.attrs["n_discrete_lines"] != 0):
            raise NotImplementedError(f"{name} has outgoing energies with discrete lines")
        rules = tables.attrs["interpolation"]
        if np.any(rules != RULE_LIN_LIN):
            raise NotImplementedError(
                f"{name} has outgoing energy rules {np.unique(rules).tolist()}"
            )
        regions = distribution["energy"].attrs["interpolation"]
        if np.any(regions[1, :] != RULE_LIN_LIN):
            raise NotImplementedError(f"{name} has incoming energy rules {regions[1, :].tolist()}")
        data = tables[()]
        if np.any(data[3, :] != RULE_DISCRETE):
            raise NotImplementedError(f"{name} has cosine rules {np.unique(data[3, :]).tolist()}")

        starts = tables.attrs["offsets"].astype(np.int64)
        ends = np.append(starts[1:], data.shape[1])
        lengths = (ends - starts)[::-1]
        order = np.concatenate(
            [np.arange(a, b)[::-1] for a, b in zip(starts, ends, strict=True)][::-1]
        )

        cosines = distribution["mu"][()]
        mu_starts = data[4, :].astype(np.int64)
        mu_ends = np.append(mu_starts[1:], cosines.shape[1])
        if np.any(mu_ends <= mu_starts):
            raise ValueError(f"{name} has cosine offsets that do not increase")
        mu_order = np.concatenate([np.arange(mu_starts[j], mu_ends[j]) for j in order])

        energy, xs = _read_xs(inelastic["xs"])
        return ThermalData(
            name=name,
            awr=float(group.attrs["atomic_weight_ratio"]),
            nuclides=tuple(n.decode() for n in group.attrs["nuclides"]),
            temperature=temperature,
            kT=float(group["kTs"][temperature][()]),
            energy_max=float(group.attrs["energy_max"]),
            energy=energy,
            xs=xs,
            spectrum=TabularSpectrum(
                energy=distribution["energy"][()][::-1].copy(),
                breakpoints_ascending=regions[0, :].astype(np.int64),
                interpolation_ascending=regions[1, :].astype(np.int64),
                offsets=np.concatenate([[0], np.cumsum(lengths)]),
                interpolation=rules[::-1].astype(np.int64),
                energy_out=data[0, order],
                pdf=data[1, order],
                cdf=data[2, order],
            ),
            mu_offsets=np.concatenate([[0], np.cumsum((mu_ends - mu_starts)[order])]),
            mu=cosines[0, mu_order],
            mu_probability=cosines[1, mu_order],
            mu_cdf=cosines[2, mu_order],
        )
