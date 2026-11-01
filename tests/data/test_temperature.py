"""Temperature interpolation: exact and analytic checks, and short OpenMC runs."""

import os
import shutil

import numpy as np
import openmc
import openmc.data
import pytest

from zerog.data.grid import thin_reactions
from zerog.data.neutron import read_reactions
from zerog.data.temperature import (
    BOLTZMANN,
    bracket,
    library_temperatures,
    nearest_label,
    read_reactions_at,
)

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)
needs_openmc = pytest.mark.skipif(
    shutil.which("openmc") is None, reason="openmc executable not found"
)


def test_boltzmann_matches_openmc():
    assert BOLTZMANN == openmc.data.K_BOLTZMANN


@needs_library
def test_temperatures_ascend_and_match_their_labels():
    labels, values = library_temperatures("U238")
    assert np.all(np.diff(values) > 0.0)
    for label, value in zip(labels, values, strict=True):
        assert abs(value - float(label[:-1])) < 1.0


@needs_library
@pytest.mark.parametrize(
    ("temperature", "label"),
    [(600.0, "600K"), (294.0, "294K"), (2500.0, "2500K"), (1200.0, "1200K")],
)
def test_nominal_temperature_returns_library_values(temperature, label):
    assert bracket("Fe56", temperature) == (label, label, 0.0)
    actual = read_reactions_at("Fe56", temperature)
    expected = read_reactions("Fe56", label)
    np.testing.assert_array_equal(actual.energy, expected.energy)
    np.testing.assert_array_equal(actual.xs, expected.xs)


@needs_library
@pytest.mark.parametrize("name", ["U238", "Fe56", "Na23"])
def test_midpoint_is_the_mean_on_both_grids(name):
    labels, values = library_temperatures(name)
    i = labels.index("600K")
    middle = 0.5 * (values[i] + values[i + 1])
    lower_label, upper_label, weight = bracket(name, middle)
    assert (lower_label, upper_label) == ("600K", "900K")
    assert weight == pytest.approx(0.5, rel=1e-12)

    data = read_reactions_at(name, middle)
    lower = read_reactions(name, "600K")
    upper = read_reactions(name, "900K")
    assert np.all(np.diff(data.energy) < 0.0)
    assert np.all(np.isin(lower.energy, data.energy))
    assert np.all(np.isin(upper.energy, data.energy))
    for r in range(data.mt.shape[0]):
        low = np.interp(data.energy, lower.energy[::-1], lower.xs[r, ::-1])
        high = np.interp(data.energy, upper.energy[::-1], upper.xs[r, ::-1])
        np.testing.assert_allclose(data.xs[r], 0.5 * (low + high), rtol=1e-14)


@needs_library
def test_mix_of_thinned_bounds_holds_tolerance():
    """Each bound through its own thinned grid, then mixed, bounds the mix for every reaction."""
    rtol, atol = 1e-3, 1e-8
    lower = read_reactions("Fe56", "600K")
    upper = read_reactions("Fe56", "900K")
    lower_grid = thin_reactions(lower, rtol, atol)
    upper_grid = thin_reactions(upper, rtol, atol)
    data = read_reactions_at("Fe56", 750.0)
    weight = bracket("Fe56", 750.0)[2]
    mask = data.energy <= 2.0e7
    energy = data.energy[mask][::-1]
    for r in range(data.mt.shape[0]):
        low_nodes = np.interp(lower_grid[::-1], lower.energy[::-1], lower.xs[r, ::-1])
        high_nodes = np.interp(upper_grid[::-1], upper.energy[::-1], upper.xs[r, ::-1])
        low = np.interp(energy, lower_grid[::-1], low_nodes)
        high = np.interp(energy, upper_grid[::-1], high_nodes)
        exact = data.xs[r, mask][::-1]
        rebuilt = (1.0 - weight) * low + weight * high
        assert np.all(np.abs(rebuilt - exact) <= rtol * exact + atol * (1.0 + 1e-9))


@needs_library
@pytest.mark.parametrize("temperature", [100.0, 3000.0, 239.0, 2511.0])
def test_beyond_tolerance_raises(temperature):
    with pytest.raises(ValueError, match="has data from"):
        read_reactions_at("U238", temperature)


@needs_library
@pytest.mark.parametrize(("temperature", "label"), [(245.0, "250K"), (2505.0, "2500K")])
def test_within_tolerance_uses_the_end_table(temperature, label):
    assert bracket("U238", temperature) == (label, label, 0.0)


@needs_library
def test_nearest_measures_from_the_nominal_temperature():
    # Cases OpenMC 0.16 accepts and rejects for U238.
    assert nearest_label("U238", 303.9) == "294K"
    assert nearest_label("U238", 259.9) == "250K"
    with pytest.raises(ValueError, match="no data within"):
        nearest_label("U238", 283.9)


def first_flight_total(cwd, density, radius, source_energy, temperature):
    """Track-length total over flux for uncollided neutrons, in barns, and its relative error.

    The relative error treats rate and flux as independent, so it overstates the noise.
    """
    material = openmc.Material()
    material.add_nuclide("U238", 1.0)
    material.set_density("atom/b-cm", density)
    material.temperature = temperature
    sphere = openmc.Sphere(r=radius, boundary_type="vacuum")
    geometry = openmc.Geometry([openmc.Cell(fill=material, region=-sphere)])

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = 20000
    settings.batches = 10
    settings.seed = 1
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(), energy=openmc.stats.Discrete([source_energy], [1.0])
    )
    settings.temperature = {"method": "interpolation"}
    settings.ptables = False
    settings.output = {"summary": False}

    energy_filter = openmc.EnergyFilter([source_energy * (1 - 1e-6), source_energy * (1 + 1e-6)])
    flux = openmc.Tally(name="flux")
    flux.filters = [energy_filter]
    flux.scores = ["flux"]
    flux.estimator = "tracklength"
    rate = openmc.Tally(name="rate")
    rate.filters = [energy_filter]
    rate.nuclides = ["U238"]
    rate.scores = ["total"]
    rate.estimator = "tracklength"

    model = openmc.Model(
        geometry, openmc.Materials([material]), settings, openmc.Tallies([flux, rate])
    )
    statepoint = model.run(cwd=cwd, output=False)
    with openmc.StatePoint(statepoint) as sp:
        flux_tally = sp.get_tally(name="flux")
        rate_tally = sp.get_tally(name="rate")
        phi = flux_tally.mean.ravel()[0]
        reaction = rate_tally.mean.ravel()[0]
        relative = np.hypot(
            flux_tally.std_dev.ravel()[0] / phi, rate_tally.std_dev.ravel()[0] / reaction
        )
    return reaction / phi / density, relative


def bound_totals(source_energy):
    def total_at(data):
        return np.interp(source_energy, data.energy[::-1], data.xs.sum(axis=0)[::-1])

    return total_at(read_reactions("U238", "600K")), total_at(read_reactions("U238", "900K"))


# On the 6.67 eV resonance's high side, where the two bounds differ by about 24%.
FLANK_ENERGY = 6.9


@needs_library
@needs_openmc
def test_openmc_thin_flights_give_the_expected_cross_section(tmp_path):
    tallied, relative = first_flight_total(tmp_path, 1e-5, 1.0, FLANK_ENERGY, 750.0)
    data = read_reactions_at("U238", 750.0)
    expected = np.interp(FLANK_ENERGY, data.energy[::-1], data.xs.sum(axis=0)[::-1])
    assert tallied == pytest.approx(expected, rel=4.0 * relative)

    low, high = bound_totals(FLANK_ENERGY)
    # A mix in sqrt(T) has weight 0.525 against linear T's 0.5. The check has to tell them apart.
    wrong_law = 0.475 * low + 0.525 * high
    assert abs(wrong_law - expected) > 4.0 * relative * expected


@needs_library
@needs_openmc
def test_openmc_thick_flights_tend_to_the_harmonic_mean(tmp_path):
    """Records why an OpenMC twin between tabulated temperatures is not the same problem."""
    tallied, _ = first_flight_total(tmp_path, 0.0223, 1000.0, FLANK_ENERGY, 750.0)
    low, high = bound_totals(FLANK_ENERGY)
    mean = 0.5 * (low + high)
    harmonic = 2.0 / (1.0 / low + 1.0 / high)
    assert abs(tallied - harmonic) < abs(tallied - mean)
