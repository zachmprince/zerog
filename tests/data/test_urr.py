"""Probability-table reader checks against openmc.data, and physics that pins the column order."""

import os
from functools import cache
from itertools import permutations

import numpy as np
import openmc.data
import pytest

from zerog.data.library import nuclide_path
from zerog.data.neutron import read_neutron
from zerog.data.urr import CAPTURE, CDF, ELASTIC, FISSION, HEATING, TOTAL, read_urr

NUCLIDES = ["U235", "U238", "Pu239", "Zr90", "Zr91", "Zr92", "Zr94", "Zr96", "Fe58", "Mn55", "Na22"]
TEMPERATURES = ["294K", "900K"]

pytestmark = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)


@cache
def reference(name):
    return openmc.data.IncidentNeutron.from_hdf5(nuclide_path(name))


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("name", NUCLIDES)
def test_tables_match_openmc(name, temperature):
    ref = reference(name).urr[temperature]
    urr = read_urr(name, temperature)

    assert urr.name == name
    assert urr.temperature == temperature
    np.testing.assert_array_equal(urr.energy[::-1], ref.energy)
    np.testing.assert_array_equal(urr.table[::-1], ref.table)
    assert urr.interpolation == ref.interpolation
    assert urr.inelastic == ref.inelastic_flag
    assert urr.absorption == ref.absorption_flag
    assert urr.multiply_smooth is ref.multiply_smooth


def test_flip_keeps_each_table_with_its_energy():
    urr = read_urr("U238", "294K")
    ref = reference("U238").urr["294K"]
    assert np.all(np.diff(urr.energy) < 0.0)
    assert urr.energy[0] == ref.energy[-1]
    np.testing.assert_array_equal(urr.table[0], ref.table[-1])
    np.testing.assert_array_equal(urr.table[-1], ref.table[0])


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("name", NUCLIDES)
def test_band_cdfs_rise_to_one(name, temperature):
    cdf = read_urr(name, temperature).table[:, CDF, :]
    assert np.all(cdf[:, 0] >= 0.0)  # Na22 has a first band of zero probability
    assert np.all(np.diff(cdf, axis=1) >= 0.0)
    np.testing.assert_allclose(cdf[:, -1], 1.0, rtol=1e-14)


@pytest.mark.parametrize("name", ["O16", "H1", "Na23", "Fe56"])
def test_nuclide_without_tables_is_none(name):
    assert read_urr(name, "294K") is None


@pytest.mark.parametrize("name", ["U238", "O16"])
def test_unknown_temperature_is_rejected(name):
    with pytest.raises(KeyError, match="123K"):
        read_urr(name, "123K")


def test_column_constants_match_openmc_documentation():
    doc = " ".join(openmc.data.ProbabilityTables.__doc__.split())
    assert (CDF, TOTAL, ELASTIC, FISSION, CAPTURE, HEATING) == (0, 1, 2, 3, 4, 5)
    assert "cumulative probability (0), total (1), elastic (2), fission (3)," in doc
    assert r":math:`(n,\gamma)` (4), or heating number (5)" in doc


def test_tables_of_cross_sections_close_directly():
    """A nuclide storing barns rather than factors, so both meanings of the flag are covered."""
    name = "Zr93"
    urr = read_urr(name, "294K")
    ref = reference(name).urr["294K"]
    assert urr.multiply_smooth is False
    np.testing.assert_array_equal(urr.table[::-1], ref.table)

    data = read_neutron(name, "294K")
    p = np.diff(urr.table[:, CDF, :], axis=1, prepend=0.0)
    for column, xs in [(TOTAL, data.total), (ELASTIC, data.elastic), (CAPTURE, data.capture)]:
        smooth = np.interp(urr.energy, data.energy[::-1], xs[::-1])
        np.testing.assert_allclose(np.sum(p * urr.table[:, column, :], axis=1), smooth, rtol=1e-3)
    assert not urr.table[:, FISSION, :].any()


def test_heating_column_of_cross_section_tables_is_kerma_per_collision():
    """Band-mean heating equals smooth heating over total, which pins the last column in eV."""
    name = "Zr93"
    urr = read_urr(name, "294K")
    ref = reference(name)
    energy = urr.energy
    heating = ref[301].xs["294K"](energy) / ref[1].xs["294K"](energy)
    p = np.diff(urr.table[:, CDF, :], axis=1, prepend=0.0)
    np.testing.assert_allclose(np.sum(p * urr.table[:, HEATING, :], axis=1), heating, rtol=5e-3)


def test_heating_column_of_factor_tables_is_scaled_by_openmc():
    """A factor near 1 times 1e6, nowhere near the eV per collision of the smooth data."""
    name = "U235"
    urr = read_urr(name, "294K")
    ref = reference(name)
    heating = ref[301].xs["294K"](urr.energy) / ref[1].xs["294K"](urr.energy)
    p = np.diff(urr.table[:, CDF, :], axis=1, prepend=0.0)
    mean = np.sum(p * urr.table[:, HEATING, :], axis=1)
    np.testing.assert_allclose(mean, 1e6, rtol=0.1)
    assert np.all(heating > 10.0 * mean)


def test_factors_average_to_one_over_bands():
    """Probability-weighted factors recover the smooth cross section, for columns 1 to 4."""
    urr = read_urr("U235", "294K")
    assert urr.multiply_smooth
    p = np.diff(urr.table[:, CDF, :], axis=1, prepend=0.0)
    for column in (TOTAL, ELASTIC, FISSION, CAPTURE):
        mean = np.sum(p * urr.table[:, column, :], axis=1)
        np.testing.assert_allclose(mean, 1.0, atol=1e-4)


@pytest.mark.parametrize("name", ["Zr90", "Zr96"])
def test_fission_factor_is_one_without_fission(name):
    """Only the fission column can be flat for a nuclide that does not fission."""
    table = read_urr(name, "294K").table
    assert np.all(table[:, FISSION, :] == 1.0)
    for column in (TOTAL, ELASTIC, CAPTURE):
        assert np.ptp(table[:, column, :]) > 0.0


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("name", ["U238", "Pu239"])
def test_band_closure_pins_the_partial_columns(name, temperature):
    """Smooth partials times band factors sum to the band total, which pins columns 1 to 4.

    Single bands of low probability miss by over a percent, so the error is probability-weighted.
    Pu239 carries the fission and capture separation, as U238 has almost no smooth fission. U235 is
    left out: its fission and capture factors are too alike to tell apart this way.
    """
    urr = read_urr(name, temperature)
    data = read_neutron(name, temperature)

    def smooth(xs):
        return np.interp(urr.energy, data.energy[::-1], xs[::-1])[:, np.newaxis]

    total = smooth(data.total)
    partials = [smooth(data.elastic), smooth(data.fission), smooth(data.capture)]
    other = total - sum(partials)
    p = np.diff(urr.table[:, CDF, :], axis=1, prepend=0.0)
    band_total = total * urr.table[:, TOTAL, :]

    def weighted_rms(columns):
        band_sum = sum(xs * urr.table[:, c, :] for xs, c in zip(partials, columns, strict=True))
        error = (band_sum + other - band_total) / band_total
        return np.sqrt(np.sum(p * error**2) / np.sum(p))

    correct = weighted_rms((ELASTIC, FISSION, CAPTURE))
    assert correct < 2e-3
    for columns in permutations((ELASTIC, FISSION, CAPTURE)):
        if columns != (ELASTIC, FISSION, CAPTURE):
            assert weighted_rms(columns) > 3.0 * correct
