"""Read incident-neutron cross sections, nu and chi from OpenMC HDF5 files.

The library stores energies low to high; everything returned here runs high to low unless its name
says otherwise. Values are otherwise as stored: eV, barns, and per-eV spectra.
"""

from dataclasses import dataclass

import h5py
import numpy as np

from zerog.data.interp import evaluate_tabulated
from zerog.data.library import nuclide_path

MT_ELASTIC = 2
MT_FISSION = 18
MT_CAPTURE = 102


@dataclass(frozen=True)
class TabularSpectrum:
    """Outgoing-energy tables per incoming energy, in flat arrays.

    Table i spans offsets[i]:offsets[i + 1] of energy_out, pdf and cdf, with interpolation[i] its
    ENDF rule. Both incoming and outgoing energies run high to low. cdf keeps the library's meaning,
    the probability of emission below that point, so it decreases along each table.

    Flipping moves histogram (rule 1) values relative to their intervals: pdf[k + 1] applies between
    energy_out[k] and energy_out[k + 1], and pdf[0] of each table is unused. Lin-lin tables are
    unaffected.

    The energy-grid interpolation regions are kept as stored, describing energy[::-1], which is also
    the order evaluate_tabulated needs.
    """

    energy: np.ndarray
    breakpoints_ascending: np.ndarray
    interpolation_ascending: np.ndarray
    offsets: np.ndarray
    interpolation: np.ndarray
    energy_out: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray


@dataclass(frozen=True)
class FissionData:
    """Neutron yields on the nuclide's energy grid, and fission spectra.

    Production must use nu_total. nu_prompt plus nu_delayed does not close to it: the evaluations
    differ slightly, and some nuclides have no delayed products at all, for which the gap reaches
    several percent.
    """

    nu_total: np.ndarray
    nu_prompt: np.ndarray
    nu_delayed: np.ndarray  # (delayed groups, energies)
    decay_rate: np.ndarray  # per delayed group, 1/s
    chi_prompt: TabularSpectrum
    chi_delayed: list[TabularSpectrum]


@dataclass(frozen=True)
class NeutronData:
    """One nuclide at one tabulated temperature. Cross sections are zero below threshold.

    elastic, capture and fission do not sum to total: other reactions, such as inelastic scattering
    and (n,alpha), are in total only. temperature is the library's label, not a value ('294K' is
    293.6 K), so use kT for physics.
    """

    name: str
    awr: float
    temperature: str
    kT: float  # eV
    energy: np.ndarray
    total: np.ndarray
    elastic: np.ndarray
    capture: np.ndarray
    fission: np.ndarray


def _read_yield(dataset, energy):
    kind = dataset.attrs["type"].decode()
    if kind == "Tabulated1D":
        return evaluate_tabulated(
            dataset[0, :],
            dataset[1, :],
            dataset.attrs["breakpoints"].astype(np.int64),
            dataset.attrs["interpolation"].astype(np.int64),
            energy,
        )
    raise NotImplementedError(f"yield of type {kind}")


def _read_spectrum(product):
    """Read a product's energy distribution, refusing forms that this reader would misrepresent."""
    if product.attrs["n_distribution"] != 1:
        raise NotImplementedError("product with more than one distribution")
    group = product["distribution_0"]
    if "applicability" in group and not np.all(group["applicability"][1, :] == 1.0):
        raise NotImplementedError("distribution with partial applicability")
    if group.attrs["type"] != b"uncorrelated":
        raise NotImplementedError(f"distribution of type {group.attrs['type'].decode()}")
    energy_group = group["energy"]
    if energy_group.attrs["type"] != b"continuous":
        raise NotImplementedError(f"energy distribution {energy_group.attrs['type'].decode()}")

    tables = energy_group["distribution"]
    if np.any(tables.attrs["n_discrete_lines"] != 0):
        raise NotImplementedError("spectrum with discrete lines")
    data = tables[()]
    starts = tables.attrs["offsets"].astype(np.int64)
    ends = np.append(starts[1:], data.shape[1])
    lengths = (ends - starts)[::-1]
    order = np.concatenate([np.arange(a, b)[::-1] for a, b in zip(starts, ends, strict=True)][::-1])

    regions = energy_group["energy"].attrs["interpolation"]
    return TabularSpectrum(
        energy=energy_group["energy"][()][::-1].copy(),
        breakpoints_ascending=regions[0, :].astype(np.int64),
        interpolation_ascending=regions[1, :].astype(np.int64),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        interpolation=tables.attrs["interpolation"][::-1].astype(np.int64),
        energy_out=data[0, order],
        pdf=data[1, order],
        cdf=data[2, order],
    )


def read_fission(data):
    """Read yields and spectra for a NeutronData, or None if it has no fission reaction.

    Raises NotImplementedError for forms not supported yet, so that cross sections stay readable for
    those nuclides: Maxwell and correlated spectra, neutrons held only in the partial fission
    reactions (MT 19, 20, 21, 38), and emission mode 'total'.
    """
    with h5py.File(nuclide_path(data.name), "r") as f:
        group = f[data.name]
        key = f"reaction_{MT_FISSION:03d}"
        if key not in group["reactions"]:
            return None
        fission = group["reactions"][key]

        prompt = None
        delayed = []
        product_keys = (k for k in fission if k.startswith("product_"))
        for product_key in sorted(product_keys, key=lambda k: int(k[8:])):
            product = fission[product_key]
            if product.attrs["particle"] != b"neutron":
                continue
            mode = product.attrs["emission_mode"]
            if mode == b"prompt":
                prompt = product
            elif mode == b"delayed":
                delayed.append(product)
            else:
                raise NotImplementedError(f"fission neutrons with emission mode {mode.decode()}")
        if prompt is None:
            raise NotImplementedError(f"{data.name} has no prompt neutrons in MT {MT_FISSION}")

        nu_delayed = np.empty((len(delayed), data.energy.shape[0]))
        for g, product in enumerate(delayed):
            nu_delayed[g] = _read_yield(product["yield"], data.energy)
        return FissionData(
            nu_total=_read_yield(group["total_nu"]["yield"], data.energy),
            nu_prompt=_read_yield(prompt["yield"], data.energy),
            nu_delayed=nu_delayed,
            decay_rate=np.array([p.attrs["decay_rate"] for p in delayed]),
            chi_prompt=_read_spectrum(prompt),
            chi_delayed=[_read_spectrum(p) for p in delayed],
        )


def read_neutron(name, temperature):
    """Read a nuclide's cross sections at a library temperature: '294K', '900K', ..."""
    if temperature == "0K":
        raise ValueError("0K holds only elastic scattering, so it has no complete cross sections")

    with h5py.File(nuclide_path(name), "r") as f:
        group = f[name]
        if temperature not in group["kTs"]:
            raise KeyError(f"{name} has no data at {temperature}: {sorted(group['kTs'])}")

        energy_ascending = group["energy"][temperature][()]
        n = energy_ascending.shape[0]
        total = np.zeros(n)
        partial = {}
        for rx in group["reactions"].values():
            xs_group = rx[temperature]
            if "xs" not in xs_group:
                continue
            threshold = xs_group["xs"].attrs["threshold_idx"]
            xs = np.zeros(n)
            xs[threshold:] = xs_group["xs"][()]
            mt = int(rx.attrs["mt"])
            if mt in (MT_ELASTIC, MT_CAPTURE, MT_FISSION):
                partial[mt] = xs
            # The library has no MT 1. OpenMC sums the non-redundant reactions, so do the same.
            if not rx.attrs["redundant"]:
                total += xs

        return NeutronData(
            name=name,
            awr=float(group.attrs["atomic_weight_ratio"]),
            temperature=temperature,
            kT=float(group["kTs"][temperature][()]),
            energy=energy_ascending[::-1].copy(),
            total=total[::-1].copy(),
            elastic=partial[MT_ELASTIC][::-1].copy(),
            capture=partial.get(MT_CAPTURE, np.zeros(n))[::-1].copy(),
            fission=partial.get(MT_FISSION, np.zeros(n))[::-1].copy(),
        )
