"""Cross sections between tabulated temperatures, following OpenMC's temperature methods.

OpenMC's 'interpolation' method picks one of the two bounding temperatures at random, with the upper
one's probability (T - T_low) / (T_high - T_low). read_reactions_at returns the expected cross
section per lookup: linear in T on the union of the two grids.

That is not OpenMC's reaction rate in general. A flight appears to keep one sample, which correlates
its length with its cross section, so a first-flight rate moves from this mean when flights are
optically thin towards the harmonic mean when they are thick. Nor is the mix physical Doppler
broadening. So an OpenMC twin between tabulated temperatures is not solving the same problem, until
a slowing-down comparison measures the difference. Results match only with OpenMC's probability
tables off, since it mixes those and S(a,b) tables too.
"""

import h5py
import numpy as np

from zerog.data.library import nuclide_path
from zerog.data.neutron import ReactionCurves, read_reactions

BOLTZMANN = 8.617333262e-5  # eV/K, as in openmc.data.K_BOLTZMANN
NOMINAL_MATCH = 0.5  # K: file values such as 293.594 K stand for a nominal 294 K


def library_temperatures(name):
    """A nuclide's temperature labels and values in kelvin, ascending by temperature."""
    with h5py.File(nuclide_path(name), "r") as f:
        kts = {label: float(dataset[()]) for label, dataset in f[name]["kTs"].items()}
    labels = sorted(kts, key=kts.get)
    return labels, np.array([kts[label] for label in labels]) / BOLTZMANN


def bracket(name, temperature, tolerance=10.0):
    """Bounding labels and the upper one's weight.

    A temperature at a nominal library value uses that table alone. One within tolerance, in kelvin,
    outside the data uses the end table, as OpenMC does. One further out raises ValueError.
    """
    labels, values = library_temperatures(name)
    nominal = np.round(values)
    i = int(np.abs(nominal - temperature).argmin())
    if abs(nominal[i] - temperature) <= NOMINAL_MATCH:
        return labels[i], labels[i], 0.0
    if temperature < values[0] or temperature > values[-1]:
        end = 0 if temperature < values[0] else -1
        if abs(temperature - values[end]) > tolerance:
            raise ValueError(
                f"{name} has data from {values[0]} to {values[-1]} K, not {temperature} K"
            )
        return labels[end], labels[end], 0.0
    upper = np.searchsorted(values, temperature)
    lower = upper - 1
    weight = (temperature - values[lower]) / (values[upper] - values[lower])
    return labels[lower], labels[upper], float(weight)


def nearest_label(name, temperature, tolerance=10.0):
    """The closest label, or ValueError if further than tolerance from its nominal temperature."""
    labels, values = library_temperatures(name)
    nominal = np.round(values)
    i = int(np.abs(nominal - temperature).argmin())
    if abs(nominal[i] - temperature) > tolerance:
        raise ValueError(f"{name} has no data within {tolerance} K of {temperature} K")
    return labels[i]


def read_reactions_at(name, temperature, tolerance=10.0):
    """Expected reaction cross sections per lookup at a temperature in kelvin; see module note."""
    lower_label, upper_label, weight = bracket(name, temperature, tolerance)
    lower = read_reactions(name, lower_label)
    if lower_label == upper_label:
        return lower
    upper = read_reactions(name, upper_label)
    if not np.array_equal(lower.mt, upper.mt):
        raise NotImplementedError(f"{name} reactions differ between bounding temperatures")

    energy = np.union1d(lower.energy, upper.energy)
    xs = np.zeros((lower.mt.shape[0], energy.shape[0]))
    for r in range(lower.mt.shape[0]):
        low = np.interp(energy, lower.energy[::-1], lower.xs[r, ::-1])
        high = np.interp(energy, upper.energy[::-1], upper.xs[r, ::-1])
        xs[r] = (1.0 - weight) * low + weight * high
    return ReactionCurves(
        energy=energy[::-1],
        mt=lower.mt,
        threshold=np.minimum(lower.threshold, upper.threshold),
        xs=xs[:, ::-1],
    )
