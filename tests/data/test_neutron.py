"""Reader checks against openmc.data, plus interpolation checks that need no library."""

import os
from functools import cache

import numpy as np
import openmc.data
import pytest

from zerog.data.interp import evaluate_tabulated
from zerog.data.library import nuclide_path
from zerog.data.neutron import read_fission, read_neutron

RTOL = 1e-14
NUCLIDES = [
    "U235",
    "U238",
    "Pu239",
    "O16",
    "H1",
    "Na23",
    "Fe56",
    "Zr90",
    "Zr91",
    "Zr92",
    "Zr94",
    "Zr96",
]
FISSILE = ["U235", "U238", "Pu239", "U233", "Pu240", "Am241", "Np237"]  # Np237 prompt is histogram
TEMPERATURES = ["294K", "900K"]

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)


@cache
def reference(name):
    return openmc.data.IncidentNeutron.from_hdf5(nuclide_path(name))


def on_grid(rx, temperature, energy):
    """A reaction's cross section on the full ascending grid, zero below threshold."""
    tab = rx.xs[temperature]
    out = np.zeros_like(energy)
    out[tab._threshold_idx :] = tab.y
    return out


@pytest.mark.parametrize("interpolation", [1, 2, 3, 4, 5])
def test_interpolation_matches_openmc_function(interpolation):
    x = np.array([1.0, 2.0, 5.0, 10.0])
    y = np.array([3.0, 1.0, 4.0, 2.0])
    breakpoints = np.array([4])
    rules = np.array([interpolation])
    query = np.geomspace(1.0, 10.0, 41)
    expected = openmc.data.Tabulated1D(x, y, breakpoints, rules)(query)
    actual = evaluate_tabulated(x, y, breakpoints, rules, query)
    np.testing.assert_allclose(actual, expected, rtol=RTOL)


def test_interpolation_multiple_regions():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = np.array([1.0, 2.0, 8.0, 4.0])
    breakpoints = np.array([2, 4])
    rules = np.array([2, 5])
    # Lin-lin to x = 2, then log-log: y = x^2 on [2, 4].
    query = np.array([1.5, 3.0])
    actual = evaluate_tabulated(x, y, breakpoints, rules, query)
    np.testing.assert_allclose(actual, [1.5, 4.5], rtol=RTOL)


def test_interpolation_clamps_outside_range():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 5.0])
    query = np.array([0.5, 3.0])
    actual = evaluate_tabulated(x, y, np.array([2]), np.array([2]), query)
    np.testing.assert_array_equal(actual, [3.0, 5.0])


@needs_library
def test_zero_kelvin_is_rejected():
    with pytest.raises(ValueError, match="0K"):
        read_neutron("H1", "0K")


@needs_library
@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("name", NUCLIDES)
def test_cross_sections_match_openmc(name, temperature):
    ref = reference(name)
    data = read_neutron(name, temperature)
    energy = ref.energy[temperature]

    assert data.awr == ref.atomic_weight_ratio
    assert data.kT == ref.kTs[ref.temperatures.index(temperature)]
    assert np.all(np.diff(data.energy) < 0.0)
    np.testing.assert_array_equal(data.energy[::-1], energy)
    np.testing.assert_array_equal(data.elastic[::-1], ref[2].xs[temperature].y)
    np.testing.assert_array_equal(data.capture[::-1], on_grid(ref[102], temperature, energy))
    if 18 in ref.reactions:
        np.testing.assert_array_equal(data.fission[::-1], on_grid(ref[18], temperature, energy))
    else:
        assert not data.fission.any()

    total = np.zeros_like(energy)
    for rx in ref.reactions.values():
        if not rx.redundant and temperature in rx.xs:
            total += on_grid(rx, temperature, energy)
    np.testing.assert_allclose(data.total[::-1], total, rtol=RTOL)
    # OpenMC's own MT 1 comes from ENDF sum rules, so it is an independent route to the total.
    # Tabulated1D's array call snaps points within a relative 1e-5 of a table's ends to the end
    # value (np.isclose's default rtol), and MT 1 is tabulated from those calls. Its scalar call
    # does not snap, so any disagreement must be exactly that artifact summed over reactions.
    sum_rule = ref[1].xs[temperature](energy)
    disagree = np.flatnonzero(~np.isclose(data.total[::-1], sum_rule, rtol=1e-12, atol=0.0))
    assert len(disagree) < 20
    for i in disagree:
        x = float(energy[i])
        artifact = 0.0
        for rx in ref.reactions.values():
            if not rx.redundant and temperature in rx.xs and x >= rx.xs[temperature].x[0]:
                tab = rx.xs[temperature]
                artifact += tab(x) - tab(np.array([x]))[0]
        assert data.total[::-1][i] == pytest.approx(sum_rule[i] + artifact, rel=1e-12)


@needs_library
@pytest.mark.parametrize("name", FISSILE)
def test_nu_matches_openmc(name):
    ref = reference(name)
    data = read_neutron(name, "294K")
    energy = ref.energy["294K"]
    fission = ref[18]
    prompt = next(p for p in fission.products if p.emission_mode == "prompt")
    delayed = [p for p in fission.products if p.emission_mode == "delayed"]
    total = next(p for p in fission.derived_products if p.emission_mode == "total")

    f = read_fission(data)
    np.testing.assert_allclose(f.nu_prompt[::-1], prompt.yield_(energy), rtol=RTOL)
    np.testing.assert_allclose(f.nu_total[::-1], total.yield_(energy), rtol=RTOL)
    assert f.nu_delayed.shape == (len(delayed), len(energy))
    for g, product in enumerate(delayed):
        np.testing.assert_allclose(f.nu_delayed[g, ::-1], product.yield_(energy), rtol=RTOL)
        assert f.decay_rate[g] == product.decay_rate


def assert_chi_matches(chi, product):
    dist = product.distribution[0].energy
    assert np.all(np.diff(chi.energy) < 0.0)
    np.testing.assert_array_equal(chi.energy[::-1], dist.energy)
    np.testing.assert_array_equal(chi.breakpoints_ascending, dist.breakpoints)
    np.testing.assert_array_equal(chi.interpolation_ascending, dist.interpolation)
    n = len(chi.energy)
    assert chi.offsets[0] == 0
    assert chi.offsets[-1] == len(chi.energy_out)
    for i in range(n):
        table = dist.energy_out[n - 1 - i]
        slice_ = slice(chi.offsets[i], chi.offsets[i + 1])
        assert np.all(np.diff(chi.energy_out[slice_]) < 0.0)
        np.testing.assert_array_equal(chi.energy_out[slice_][::-1], table.x)
        np.testing.assert_array_equal(chi.pdf[slice_][::-1], table.p)
        np.testing.assert_array_equal(chi.cdf[slice_][::-1], table.c)
        assert chi.interpolation[i] == {"histogram": 1, "linear-linear": 2}[table.interpolation]


@needs_library
@pytest.mark.parametrize("name", FISSILE)
def test_chi_matches_openmc(name):
    fission = reference(name)[18]
    prompt = next(p for p in fission.products if p.emission_mode == "prompt")
    delayed = [p for p in fission.products if p.emission_mode == "delayed"]

    f = read_fission(read_neutron(name, "294K"))
    assert_chi_matches(f.chi_prompt, prompt)
    assert len(f.chi_delayed) == len(delayed)
    for chi, product in zip(f.chi_delayed, delayed, strict=True):
        assert_chi_matches(chi, product)


@needs_library
def test_non_fissile_has_no_fission_data():
    assert read_fission(read_neutron("H1", "294K")) is None


@needs_library
@pytest.mark.parametrize("name", ["U235", "Np237"])
def test_flipped_tables_integrate_to_their_cdf(name):
    """The documented interval rule for flipped tables, checked against the stored cdf."""
    f = read_fission(read_neutron(name, "294K"))
    for chi in [f.chi_prompt, *f.chi_delayed]:
        for i in range(len(chi.energy)):
            s = slice(chi.offsets[i], chi.offsets[i + 1])
            e, p, c = chi.energy_out[s], chi.pdf[s], chi.cdf[s]
            width = e[:-1] - e[1:]
            if chi.interpolation[i] == 1:
                increment = p[1:] * width
            else:
                increment = 0.5 * (p[:-1] + p[1:]) * width
            np.testing.assert_allclose(c[:-1] - c[1:], increment, rtol=1e-3, atol=1e-7)


@needs_library
def test_no_delayed_products_keeps_shape():
    data = read_neutron("Am242_m1", "294K")
    f = read_fission(data)
    assert f.nu_delayed.shape == (0, len(data.energy))
    assert np.any(f.nu_total > f.nu_prompt)


@needs_library
@pytest.mark.parametrize("name", ["Pu241", "Th232", "U236", "Pu245"])
def test_unsupported_fission_forms_refuse_but_cross_sections_read(name):
    data = read_neutron(name, "294K")
    assert data.fission.any()
    with pytest.raises(NotImplementedError):
        read_fission(data)


def test_interpolation_zero_value_matches_openmc():
    x = np.array([1.0, 2.0])
    y = np.array([0.0, 1.0])
    query = np.array([1.5])
    expected = openmc.data.Tabulated1D(x, y, [2], [5])(query)
    actual = evaluate_tabulated(x, y, np.array([2]), np.array([5]), query)
    np.testing.assert_array_equal(actual, expected)
