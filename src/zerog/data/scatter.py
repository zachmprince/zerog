"""Read secondary-neutron laws from OpenMC HDF5 files, as flat arrays.

Incoming and outgoing energies run high to low, and a table's interpolation rules describe the
ascending order that the library stores. mu stays ascending, since it is not an energy. Laws hold no
yields or cross sections: see zerog.data.neutron for those.

Flipping a table moves histogram (rule 1) values relative to their intervals: pdf[k + 1] applies
between energy_out[k] and energy_out[k + 1], and pdf[0] of each table is unused. The same shift
applies to every per-point row (Kalbach r and a, and correlated mu tables), and most Kalbach-Mann
laws are rule 1. Lin-lin tables are unaffected. Outgoing energies can repeat, and in a few tables
do not decrease, so kernels must tolerate both. No law in ENDF/B-VIII.0
has discrete lines, so tables that do are refused.
"""

from dataclasses import dataclass

import h5py
import numpy as np

from zerog.data.library import nuclide_path

MT_FISSION_ALL = (18, 19, 20, 21, 38)


@dataclass(frozen=True)
class TabularSpectrum:
    """Outgoing-energy tables per incoming energy. Table i spans offsets[i]:offsets[i + 1].

    interpolation[i] is its ENDF rule. cdf keeps the library's meaning, the probability of emission
    below that point, so it decreases along each table. The energy-grid interpolation regions
    describe energy[::-1], which is also the order evaluate_tabulated needs.
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
class EvaporationSpectrum:
    """Evaporation spectrum E' exp(-E' / theta(E)), normalized over 0 <= E' <= E - u."""

    energy: np.ndarray
    theta: np.ndarray  # eV
    breakpoints_ascending: np.ndarray
    interpolation_ascending: np.ndarray
    u: float  # eV


@dataclass(frozen=True)
class LevelEnergy:
    """Discrete level: outgoing energy in the centre of mass is mass_ratio * (E - threshold)."""

    threshold: float  # eV, negative for exothermic levels of metastable targets
    mass_ratio: float


@dataclass(frozen=True)
class AngleTables:
    """Tabulated mu per incoming energy. Table i spans offsets[i]:offsets[i + 1]; mu ascends."""

    energy: np.ndarray
    offsets: np.ndarray
    interpolation: np.ndarray
    mu: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray


@dataclass(frozen=True)
class UncorrelatedLaw:
    """Independent angle and energy. energy None means elastic.

    angle None means isotropic, but occurs only on delayed fission products. Isotropic scattering
    otherwise arrives as a flat table, so None cannot be used to detect it.
    """

    angle: AngleTables | None
    energy: TabularSpectrum | MaxwellSpectrum | EvaporationSpectrum | LevelEnergy | None


@dataclass(frozen=True)
class KalbachMannLaw:
    """Outgoing-energy tables, as in TabularSpectrum, with Kalbach r and a at each point."""

    spectrum: TabularSpectrum
    precompound: np.ndarray
    slope: np.ndarray


@dataclass(frozen=True)
class CorrelatedLaw:
    """Outgoing-energy tables, as in TabularSpectrum, with a mu table at each outgoing point.

    Point k's table spans mu_start[k]:mu_stop[k] of mu, mu_pdf and mu_cdf, with rule mu_rule[k].
    """

    spectrum: TabularSpectrum
    mu_start: np.ndarray
    mu_stop: np.ndarray
    mu_rule: np.ndarray
    mu: np.ndarray
    mu_pdf: np.ndarray
    mu_cdf: np.ndarray


@dataclass(frozen=True)
class NBodyLaw:
    """N-body phase-space distribution, which is defined analytically."""

    n_particles: int
    total_mass: float
    atomic_weight_ratio: float
    q_value: float  # eV


@dataclass(frozen=True)
class Applicability:
    """Probability of selecting a distribution among several, per incoming energy, high to low.

    It is not a neutron weight. A reaction with one law can store p != 1 (O18 MT 16 has 0.5 with
    yield 2), and that p must be ignored: the yield alone carries the neutron count.
    """

    energy: np.ndarray
    probability: np.ndarray
    breakpoints_ascending: np.ndarray
    interpolation_ascending: np.ndarray


@dataclass(frozen=True)
class ReactionLaw:
    """A reaction's neutron laws, with one applicability per law, or None where absent.

    With several laws, applicability selects among them at each energy. With one, ignore it.
    Yields, Q values and masses are in zerog.data.neutron, joined by MT.
    """

    mt: int
    center_of_mass: bool
    applicability: list[Applicability | None]
    laws: list[UncorrelatedLaw | KalbachMannLaw | CorrelatedLaw | NBodyLaw]


def _table_order(starts, total):
    """Flipped table lengths and indices that reverse tables and the points within each."""
    ends = np.append(starts[1:], total)
    order = np.concatenate([np.arange(a, b)[::-1] for a, b in zip(starts, ends, strict=True)][::-1])
    return (ends - starts)[::-1], order


def _regions(attrs_source):
    return attrs_source[0, :].astype(np.int64), attrs_source[1, :].astype(np.int64)


def _flipped_spectrum(energy, regions, tables):
    if np.any(tables.attrs["n_discrete_lines"] != 0):
        raise NotImplementedError("spectrum with discrete lines")
    data = tables[()]
    lengths, order = _table_order(tables.attrs["offsets"].astype(np.int64), data.shape[1])
    breakpoints, rules = _regions(regions)
    spectrum = TabularSpectrum(
        energy=energy[::-1].copy(),
        breakpoints_ascending=breakpoints,
        interpolation_ascending=rules,
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        interpolation=tables.attrs["interpolation"][::-1].astype(np.int64),
        energy_out=data[0, order],
        pdf=data[1, order],
        cdf=data[2, order],
    )
    return spectrum, data, order


def _read_parametric(cls, group):
    theta = group["theta"][()]
    attrs = group["theta"].attrs
    return cls(
        energy=theta[0, ::-1].copy(),
        theta=theta[1, ::-1].copy(),
        breakpoints_ascending=attrs["breakpoints"].astype(np.int64),
        interpolation_ascending=attrs["interpolation"].astype(np.int64),
        u=float(group.attrs["u"]),
    )


def _read_energy(group):
    kind = group.attrs["type"].decode()
    if kind == "continuous":
        return _flipped_spectrum(
            group["energy"][()], group["energy"].attrs["interpolation"], group["distribution"]
        )[0]
    if kind == "maxwell":
        return _read_parametric(MaxwellSpectrum, group)
    if kind == "evaporation":
        return _read_parametric(EvaporationSpectrum, group)
    if kind == "level":
        return LevelEnergy(float(group.attrs["threshold"]), float(group.attrs["mass_ratio"]))
    raise NotImplementedError(f"energy distribution {kind}")


def _read_angle(group):
    mu = group["mu"][()]
    starts = group["mu"].attrs["offsets"].astype(np.int64)
    ends = np.append(starts[1:], mu.shape[1])
    order = np.concatenate([np.arange(a, b) for a, b in zip(starts, ends, strict=True)][::-1])
    return AngleTables(
        energy=group["energy"][()][::-1].copy(),
        offsets=np.concatenate([[0], np.cumsum((ends - starts)[::-1])]),
        interpolation=group["mu"].attrs["interpolation"][::-1].astype(np.int64),
        mu=mu[0, order],
        pdf=mu[1, order],
        cdf=mu[2, order],
    )


def read_law(group):
    """Read one distribution group into a law, refusing forms it would misrepresent."""
    kind = group.attrs["type"].decode()
    if kind == "uncorrelated":
        return UncorrelatedLaw(
            angle=_read_angle(group["angle"]) if "angle" in group else None,
            energy=_read_energy(group["energy"]) if "energy" in group else None,
        )
    if kind == "kalbach-mann":
        spectrum, data, order = _flipped_spectrum(
            group["energy"][()], group["energy"].attrs["interpolation"], group["distribution"]
        )
        return KalbachMannLaw(spectrum, precompound=data[3, order], slope=data[4, order])
    if kind == "correlated":
        spectrum, data, order = _flipped_spectrum(
            group["energy"][()], group["energy"].attrs["interpolation"], group["energy_out"]
        )
        mu = group["mu"][()]
        stored_starts = data[4, :].astype(np.int64)
        stored_stops = np.append(stored_starts[1:], mu.shape[1])
        return CorrelatedLaw(
            spectrum,
            mu_start=stored_starts[order],
            mu_stop=stored_stops[order],
            mu_rule=data[3, order].astype(np.int64),
            mu=mu[0],
            mu_pdf=mu[1],
            mu_cdf=mu[2],
        )
    if kind == "nbody":
        return NBodyLaw(
            n_particles=int(group.attrs["n_particles"]),
            total_mass=float(group.attrs["total_mass"]),
            atomic_weight_ratio=float(group.attrs["atomic_weight_ratio"]),
            q_value=float(group.attrs["q_value"]),
        )
    raise NotImplementedError(f"distribution of type {kind}")


def _read_applicability(group):
    if "applicability" not in group:
        return None
    table = group["applicability"][()]
    attrs = group["applicability"].attrs
    return Applicability(
        energy=table[0, ::-1].copy(),
        probability=table[1, ::-1].copy(),
        breakpoints_ascending=attrs["breakpoints"].astype(np.int64),
        interpolation_ascending=attrs["interpolation"].astype(np.int64),
    )


def read_scattering_laws(name):
    """Read each non-redundant, non-fission reaction that emits a neutron, ordered by MT.

    Laws do not depend on temperature in the library. Reactions with no neutron product are omitted.
    """
    out = []
    with h5py.File(nuclide_path(name), "r") as f:
        for rx in f[name]["reactions"].values():
            mt = int(rx.attrs["mt"])
            if rx.attrs["redundant"] or mt in MT_FISSION_ALL:
                continue
            products = [
                rx[k]
                for k in rx
                if k.startswith("product_") and rx[k].attrs["particle"] == b"neutron"
            ]
            if not products:
                continue
            if len(products) > 1:
                raise NotImplementedError(f"{name} MT {mt} has several neutron products")
            groups = [
                products[0][f"distribution_{i}"]
                for i in range(int(products[0].attrs["n_distribution"]))
            ]
            out.append(
                ReactionLaw(
                    mt=mt,
                    center_of_mass=bool(rx.attrs["center_of_mass"]),
                    applicability=[_read_applicability(g) for g in groups],
                    laws=[read_law(g) for g in groups],
                )
            )
    return sorted(out, key=lambda law: law.mt)
