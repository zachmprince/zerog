"""Read unresolved-range probability tables from OpenMC HDF5 files.

The library stores energies low to high; here they run high to low, with the tables flipped to
match. Values are otherwise as stored. Using the tables in physics is not done here.
"""

from dataclasses import dataclass

import h5py
import numpy as np

from zerog.data.library import nuclide_path

# Order of the table's second axis, as documented by openmc.data.ProbabilityTables.
CDF = 0
TOTAL = 1
ELASTIC = 2
FISSION = 3
CAPTURE = 4
HEATING = 5


@dataclass(frozen=True)
class ProbabilityTables:
    """Band tables of one nuclide at one tabulated temperature.

    table has shape (energies, 6, bands), with the second axis indexed by the constants above. Bands
    keep the library's order, so the cdf never falls along them and ends at 1. A first band can have
    zero probability (Na22).

    With multiply_smooth, the total, elastic, fission and capture columns are factors on the smooth
    cross sections, averaging to 1 over the bands. Otherwise they are barns, whose band means can
    miss the smooth cross sections by over 10% (Eu156 capture) and whose values can be negative
    (Na22). Partials sum to the total only on average over the bands: single bands of low
    probability miss by 13% in U235 and by over 100% in some nuclides (Zr96).

    The heating column has two meanings. Without multiply_smooth it is eV per collision. With it, it
    is a factor times 1e6, because OpenMC scales this column from MeV whatever it holds, so it is
    not in eV. It can be negative (Zr96).

    interpolation is the ENDF rule between tables, 2 (lin-lin) or 5 (log-log), which needs no change
    for the flip. Every table in ENDF/B-VIII.0 uses 2. inelastic and absorption are stored flags:
    positive is the MT of the smooth reaction to add in the range, and negative means none is
    flagged, though the library can still hold a small residual, up to 3e-4 of the total. Absorption
    0 is undocumented by OpenMC. The data suggest it means several reactions to sum: they reach 30%
    of the total in Na22, so do not read 0 as none.
    """

    name: str
    temperature: str
    energy: np.ndarray
    table: np.ndarray
    interpolation: int
    inelastic: int
    absorption: int
    multiply_smooth: bool


def read_urr(name, temperature):
    """Read a nuclide's tables at a library temperature, or None if it has no tables."""
    with h5py.File(nuclide_path(name), "r") as f:
        group = f[name]
        if temperature not in group["kTs"]:
            raise KeyError(f"{name} has no data at {temperature}: {sorted(group['kTs'])}")
        if "urr" not in group:
            return None

        urr = group["urr"][temperature]
        return ProbabilityTables(
            name=name,
            temperature=temperature,
            energy=urr["energy"][()][::-1].copy(),
            table=urr["table"][()][::-1].copy(),
            interpolation=int(urr.attrs["interpolation"]),
            inelastic=int(urr.attrs["inelastic"]),
            absorption=int(urr.attrs["absorption"]),
            multiply_smooth=bool(urr.attrs["multiply_smooth"]),
        )
