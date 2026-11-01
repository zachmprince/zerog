"""Read incident-neutron cross sections, nu and chi from OpenMC HDF5 files.

The library stores energies low to high; everything returned here runs high to low unless its name
says otherwise. Values are otherwise as stored: eV, barns, and per-eV spectra.
"""

from dataclasses import dataclass

import h5py
import numpy as np
from numpy.polynomial.polynomial import polyval

from zerog.data.interp import evaluate_tabulated
from zerog.data.library import nuclide_path

MT_ELASTIC = 2
MT_FISSION = 18
MT_CAPTURE = 102
MT_PARTIAL_FISSION = (19, 20, 21, 38)


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
class MaxwellSpectrum:
    """Maxwell spectrum sqrt(E') exp(-E' / theta(E)), normalized over 0 <= E' <= E - u.

    energy and theta run high to low, with interpolation regions describing energy[::-1].
    """

    energy: np.ndarray
    theta: np.ndarray  # eV
    breakpoints_ascending: np.ndarray
    interpolation_ascending: np.ndarray
    u: float  # eV, restriction energy, which can be negative


@dataclass(frozen=True)
class FissionData:
    """One fission reaction: MT 18, or one of the partial reactions that replace it.

    Yields are on the nuclide's energy grid. A nuclide's production spectrum sums over its
    reactions, each with its own spectra:

        xs * [(nu_total - sum(nu_delayed)) * chi_prompt + sum(nu_delayed * chi_delayed)]

    Use that prompt weight, not nu_prompt. nu_total is the nuclide's, not this reaction's, and
    nu_prompt plus nu_delayed does not close to it: the evaluations differ slightly, and some
    nuclides have no delayed products at all, for which the gap reaches several percent. Partial
    reactions each repeat the same yields, so never sum them across reactions unweighted.

    With emission_mode 'total' the file does not separate prompt and delayed neutrons. nu_prompt and
    chi_prompt then hold all of them, and there are no delayed groups.
    """

    mt: int
    xs: np.ndarray
    emission_mode: str  # 'prompt' or 'total'
    nu_total: np.ndarray
    nu_prompt: np.ndarray
    nu_delayed: np.ndarray  # (delayed groups, energies)
    decay_rate: np.ndarray  # per delayed group, 1/s
    chi_prompt: TabularSpectrum | MaxwellSpectrum
    chi_delayed: list[TabularSpectrum | MaxwellSpectrum]


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


@dataclass(frozen=True)
class ReactionCurves:
    """Non-redundant reaction cross sections of one nuclide, which sum to its total."""

    energy: np.ndarray
    mt: np.ndarray
    threshold: np.ndarray  # eV, per reaction
    q: np.ndarray  # eV, per reaction
    xs: np.ndarray  # (reactions, energies), zero below threshold


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
    if kind == "Polynomial":
        return polyval(energy, dataset[()])
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
    if energy_group.attrs["type"] == b"maxwell":
        theta = energy_group["theta"]
        return MaxwellSpectrum(
            energy=theta[()][0, ::-1].copy(),
            theta=theta[()][1, ::-1].copy(),
            breakpoints_ascending=theta.attrs["breakpoints"].astype(np.int64),
            interpolation_ascending=theta.attrs["interpolation"].astype(np.int64),
            u=float(energy_group.attrs["u"]),
        )
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
    """Read each fission reaction of a NeutronData, as a list that is empty if it has none.

    Raises NotImplementedError for forms not supported yet, so that cross sections stay readable for
    those nuclides: correlated spectra.
    """
    with h5py.File(nuclide_path(data.name), "r") as f:
        group = f[data.name]
        out = []
        for rx in group["reactions"].values():
            mt = int(rx.attrs["mt"])
            if mt in (MT_FISSION, *MT_PARTIAL_FISSION) and not rx.attrs["redundant"]:
                out.append(_read_fission_reaction(group, rx, mt, data))
        return sorted(out, key=lambda reaction: reaction.mt)


def _read_fission_reaction(group, rx, mt, data):
    leading = None
    delayed = []
    product_keys = (k for k in rx if k.startswith("product_"))
    for product_key in sorted(product_keys, key=lambda k: int(k[8:])):
        product = rx[product_key]
        if product.attrs["particle"] != b"neutron":
            continue
        mode = product.attrs["emission_mode"]
        if mode in (b"prompt", b"total"):
            leading = product
        elif mode == b"delayed":
            delayed.append(product)
        else:
            raise NotImplementedError(f"fission neutrons with emission mode {mode.decode()}")
    if leading is None:
        raise NotImplementedError(f"{data.name} has no prompt neutrons in MT {mt}")

    n = data.energy.shape[0]
    dataset = rx[data.temperature]["xs"]
    xs = np.zeros(n)
    xs[dataset.attrs["threshold_idx"] :] = dataset[()]
    nu_leading = _read_yield(leading["yield"], data.energy)
    nu_delayed = np.empty((len(delayed), n))
    for g, product in enumerate(delayed):
        nu_delayed[g] = _read_yield(product["yield"], data.energy)
    return FissionData(
        mt=mt,
        xs=xs[::-1].copy(),
        emission_mode=leading.attrs["emission_mode"].decode(),
        nu_total=(
            _read_yield(group["total_nu"]["yield"], data.energy)
            if "total_nu" in group
            else nu_leading
        ),
        nu_prompt=nu_leading,
        nu_delayed=nu_delayed,
        decay_rate=np.array([p.attrs["decay_rate"] for p in delayed]),
        chi_prompt=_read_spectrum(leading),
        chi_delayed=[_read_spectrum(p) for p in delayed],
    )


def read_reactions(name, temperature):
    """Read every non-redundant reaction with a cross section at a library temperature."""
    if temperature == "0K":
        raise ValueError("0K holds only elastic scattering, so it has no complete cross sections")

    with h5py.File(nuclide_path(name), "r") as f:
        group = f[name]
        if temperature not in group["kTs"]:
            raise KeyError(f"{name} has no data at {temperature}: {sorted(group['kTs'])}")

        energy_ascending = group["energy"][temperature][()]
        n = energy_ascending.shape[0]
        mts = []
        thresholds = []
        qs = []
        curves = []
        for rx in group["reactions"].values():
            if rx.attrs["redundant"] or "xs" not in rx[temperature]:
                continue
            dataset = rx[temperature]["xs"]
            threshold = dataset.attrs["threshold_idx"]
            xs = np.zeros(n)
            xs[threshold:] = dataset[()]
            mts.append(int(rx.attrs["mt"]))
            thresholds.append(energy_ascending[threshold])
            qs.append(float(rx.attrs["Q_value"]))
            curves.append(xs[::-1])

        return ReactionCurves(
            energy=energy_ascending[::-1].copy(),
            mt=np.array(mts, dtype=np.int64),
            threshold=np.array(thresholds),
            q=np.array(qs),
            xs=np.array(curves),
        )


def read_scattering_yields(name, mts, energy):
    """Neutrons emitted per reaction, as (reactions, energies); zero for fission reactions.

    Fission neutrons are in read_fission. A reaction with zero yield removes the neutron. Only MT 5
    has a yield that varies with energy, and a thinned grid does not certify it, or nu.
    """
    out = np.zeros((len(mts), energy.shape[0]))
    with h5py.File(nuclide_path(name), "r") as f:
        for r, mt in enumerate(mts):
            if mt in (MT_FISSION, *MT_PARTIAL_FISSION):
                continue
            rx = f[name]["reactions"][f"reaction_{int(mt):03d}"]
            for key in rx:
                if key.startswith("product_") and rx[key].attrs["particle"] == b"neutron":
                    out[r] += _read_yield(rx[key]["yield"], energy)
    return out


def neutron_balance(data, yields):
    """Split a ReactionCurves total for particle balance, each (energies).

    Returns scattering production, the sum of yield * xs over non-fission reactions; the fission
    cross section; and net loss, the total less both. Net loss is what a balance needs: it counts
    each reaction by its yield, so it can go negative where (n,xn) dominates, and it handles MT 5,
    whose yield is fractional and depends on energy. It is not a sum of whole reactions.
    """
    is_fission = np.isin(data.mt, (MT_FISSION, *MT_PARTIAL_FISSION))
    production = (data.xs * yields)[~is_fission].sum(axis=0)
    fission = data.xs[is_fission].sum(axis=0)
    return production, fission, data.xs.sum(axis=0) - production - fission


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
