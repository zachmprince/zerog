"""Thermal scattering reader checks: a small synthetic file, then table by table against openmc."""

import os
from functools import cache

import h5py
import numpy as np
import openmc.data
import pytest

from zerog.data.library import nuclide_path
from zerog.data.thermal import read_thermal

NAME = "c_H_in_H2O"
TEMPERATURES = [
    "284K",
    "294K",
    "300K",
    "324K",
    "350K",
    "374K",
    "400K",
    "424K",
    "450K",
    "474K",
    "500K",
    "524K",
    "550K",
    "574K",
    "600K",
    "624K",
    "650K",
    "800K",
]
# The file's pdf and cdf agree to round-off over most intervals, but not all: up to 1.5e-5 in one
# interval, and each table's trapezoid integral exceeds 1 by up to 2.7e-5. Specific to H in H2O.
INTERVAL_ATOL = 2e-5
TYPICAL_ATOL = 1e-11
TABLE_EXCESS = 3e-5

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)


@cache
def reference():
    return openmc.data.ThermalScattering.from_hdf5(nuclide_path(NAME, "thermal"))


def write_library(directory, monkeypatch, edit=None):
    """A one-temperature library: two incoming energies, 2 + 3 outgoing, 2 or 3 cosines each."""
    xml = directory / "cross_sections.xml"
    xml.write_text(
        '<cross_sections><library materials="c_Test" path="c_Test.h5" type="thermal"/>'
        "</cross_sections>"
    )
    monkeypatch.setenv("OPENMC_CROSS_SECTIONS", str(xml))

    energy_out = np.array(
        [
            [0.0, 1.0, 0.0, 0.5, 2.0],  # E'
            [1.0, 1.0, 0.8, 0.6, 0.4],  # pdf
            [0.0, 1.0, 0.0, 0.35, 1.0],  # cdf
            [0, 0, 0, 0, 0],  # cosine rule, 0 is discrete
            [0, 2, 4, 6, 8],  # cosine offsets
        ]
    )
    mu = np.array(
        [
            [-0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.0, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3],
            [0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 1 / 3, 2 / 3, 1.0],
        ]
    )
    with h5py.File(directory / "c_Test.h5", "w") as f:
        group = f.create_group("c_Test")
        group.attrs["atomic_weight_ratio"] = 0.5
        group.attrs["energy_max"] = 2.0
        group.attrs["nuclides"] = np.array([b"H1"])
        group["kTs/294K"] = 0.0253

        inelastic = group.create_group("294K/inelastic")
        xs = inelastic.create_dataset("xs", data=np.array([[1.0, 2.0], [30.0, 20.0]]))
        xs.attrs["type"] = np.bytes_("Tabulated1D")
        xs.attrs["breakpoints"] = np.array([2])
        xs.attrs["interpolation"] = np.array([2])

        distribution = inelastic.create_group("distribution")
        distribution.attrs["type"] = np.bytes_("incoherent_inelastic")
        energy = distribution.create_dataset("energy", data=np.array([1.0, 2.0]))
        energy.attrs["interpolation"] = np.array([[2], [2]])
        table = distribution.create_dataset("energy_out", data=energy_out)
        table.attrs["offsets"] = np.array([0, 2])
        table.attrs["interpolation"] = np.array([2, 2])
        table.attrs["n_discrete_lines"] = np.array([0, 0])
        distribution["mu"] = mu
        if edit is not None:
            edit(group["294K"])


def test_synthetic_file_is_flipped_and_rows_stay_with_their_cosines(tmp_path, monkeypatch):
    write_library(tmp_path, monkeypatch)
    data = read_thermal("c_Test", "294K")

    assert data.name == "c_Test"
    assert data.awr == 0.5
    assert data.nuclides == ("H1",)
    assert data.temperature == "294K"
    assert data.kT == 0.0253
    np.testing.assert_array_equal(data.energy, [2.0, 1.0])
    np.testing.assert_array_equal(data.xs, [20.0, 30.0])

    spectrum = data.spectrum
    np.testing.assert_array_equal(spectrum.energy, [2.0, 1.0])
    np.testing.assert_array_equal(spectrum.offsets, [0, 3, 5])
    np.testing.assert_array_equal(spectrum.interpolation, [2, 2])
    np.testing.assert_array_equal(spectrum.energy_out, [2.0, 0.5, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(spectrum.pdf, [0.4, 0.6, 0.8, 1.0, 1.0])
    np.testing.assert_array_equal(spectrum.cdf, [1.0, 0.35, 0.0, 1.0, 0.0])

    np.testing.assert_array_equal(data.mu_offsets, [0, 3, 5, 7, 9, 11])
    np.testing.assert_array_equal(
        data.mu, [-0.5, 0.0, 0.5, -0.4, 0.4, -0.3, 0.3, -0.2, 0.2, -0.1, 0.1]
    )
    np.testing.assert_array_equal(
        data.mu_probability, [1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    )
    np.testing.assert_array_equal(
        data.mu_cdf, [1 / 3, 2 / 3, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
    )


def add_elastic(kind):
    def edit(group):
        group.create_group("elastic/distribution").attrs["type"] = np.bytes_(kind)

    return edit


def set_distribution_type(group):
    group["inelastic/distribution"].attrs["type"] = np.bytes_("incoherent_inelastic_discrete")


def set_xs_type(group):
    group["inelastic/xs"].attrs["type"] = np.bytes_("Polynomial")


def set_xs_rule(group):
    group["inelastic/xs"].attrs["interpolation"] = np.array([5])


def set_discrete_lines(group):
    group["inelastic/distribution/energy_out"].attrs["n_discrete_lines"] = np.array([0, 1])


def set_histogram_outgoing(group):
    group["inelastic/distribution/energy_out"].attrs["interpolation"] = np.array([2, 1])


def set_histogram_incoming(group):
    group["inelastic/distribution/energy"].attrs["interpolation"] = np.array([[2], [1]])


def set_tabular_cosines(group):
    group["inelastic/distribution/energy_out"][3, 3] = 2


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (add_elastic("coherent_elastic"), "coherent_elastic"),
        (add_elastic("incoherent_elastic"), "incoherent_elastic"),
        (add_elastic("mixed_elastic"), "mixed_elastic"),
        (set_distribution_type, "incoherent_inelastic_discrete"),
        (set_xs_type, "Polynomial"),
        (set_xs_rule, "interpolation"),
        (set_discrete_lines, "discrete"),
        (set_histogram_outgoing, "outgoing energy rules"),
        (set_histogram_incoming, "incoming energy rules"),
        (set_tabular_cosines, "cosine"),
    ],
)
def test_other_forms_are_refused_by_name(tmp_path, monkeypatch, edit, message):
    write_library(tmp_path, monkeypatch, edit)
    with pytest.raises(NotImplementedError, match=message):
        read_thermal("c_Test", "294K")


def test_unknown_temperature_names_those_available(tmp_path, monkeypatch):
    write_library(tmp_path, monkeypatch)
    with pytest.raises(KeyError, match="294K"):
        read_thermal("c_Test", "293K")


@needs_library
def test_graphite_coherent_elastic_is_refused():
    with h5py.File(nuclide_path("c_Graphite", "thermal"), "r") as f:
        temperature = sorted(f["c_Graphite/kTs"])[0]
    with pytest.raises(NotImplementedError, match="elastic"):
        read_thermal("c_Graphite", temperature)


@needs_library
def test_lists_every_tabulated_temperature():
    assert sorted(reference().temperatures) == TEMPERATURES


@needs_library
@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_matches_openmc_table_by_table(temperature):
    data = read_thermal(NAME, temperature)
    ref = reference()
    distribution = ref.inelastic.distribution[temperature]
    xs = ref.inelastic.xs[temperature]

    assert data.awr == ref.atomic_weight_ratio
    assert data.nuclides == tuple(ref.nuclides)
    assert data.kT == ref.kTs[ref.temperatures.index(temperature)]
    np.testing.assert_array_equal(data.energy, xs.x[::-1])
    np.testing.assert_array_equal(data.xs, xs.y[::-1])

    spectrum = data.spectrum
    np.testing.assert_array_equal(spectrum.energy, distribution.energy[::-1])
    n = spectrum.energy.shape[0]
    assert spectrum.offsets.shape[0] == n + 1

    row = 0
    for i in range(n):
        a, b = spectrum.offsets[i], spectrum.offsets[i + 1]
        table = distribution.energy_out[n - 1 - i]
        cosines = distribution.mu[n - 1 - i][::-1]
        np.testing.assert_array_equal(spectrum.energy_out[a:b], table.x[::-1])
        np.testing.assert_array_equal(spectrum.pdf[a:b], table.p[::-1])
        np.testing.assert_array_equal(spectrum.cdf[a:b], table.c[::-1])
        assert b - a == len(cosines)

        lo, hi = data.mu_offsets[row], data.mu_offsets[row + b - a]
        np.testing.assert_array_equal(data.mu[lo:hi], np.concatenate([c.x for c in cosines]))
        np.testing.assert_array_equal(
            data.mu_probability[lo:hi], np.concatenate([c.p for c in cosines])
        )
        np.testing.assert_array_equal(data.mu_cdf[lo:hi], np.concatenate([c.c for c in cosines]))
        np.testing.assert_array_equal(
            np.diff(data.mu_offsets[row : row + b - a + 1]), [len(c.x) for c in cosines]
        )
        row += b - a
    assert row == data.mu_offsets.shape[0] - 1


@needs_library
@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_pdf_integrates_to_cdf(temperature):
    spectrum = read_thermal(NAME, temperature).spectrum
    assert np.all(spectrum.interpolation == 2)
    assert spectrum.pdf.min() > -1e-26
    residuals = []
    for a, b in zip(spectrum.offsets[:-1], spectrum.offsets[1:], strict=True):
        energy_out = spectrum.energy_out[a:b]
        pdf = spectrum.pdf[a:b]
        cdf = spectrum.cdf[a:b]
        assert np.all(np.diff(energy_out) < 0.0)
        trapezoids = 0.5 * (energy_out[:-1] - energy_out[1:]) * (pdf[:-1] + pdf[1:])
        residuals.append(cdf[:-1] - cdf[1:] - trapezoids)
        assert 0.0 <= trapezoids.sum() - 1.0 < TABLE_EXCESS
        assert cdf[0] == 1.0
        assert cdf[-1] == 0.0
    residuals = np.abs(np.concatenate(residuals))
    assert residuals.max() < INTERVAL_ATOL
    assert np.median(residuals) < TYPICAL_ATOL


@needs_library
def test_top_table_approaches_free_atom_scattering():
    """At 10 eV hydrogen is nearly free, which would fail if rows lost their own cosines."""
    data = read_thermal(NAME, "294K")
    spectrum = data.spectrum
    b = spectrum.offsets[1]
    energy_out = spectrum.energy_out[:b]
    pdf = spectrum.pdf[:b]
    mean_mu = np.add.reduceat(data.mu * data.mu_probability, data.mu_offsets[:-1])[:b]

    def integrate(values):
        return np.sum(0.5 * (energy_out[:-1] - energy_out[1:]) * (values[:-1] + values[1:]))

    assert data.energy[0] == 10.0
    np.testing.assert_allclose(integrate(pdf * energy_out) / data.energy[0], 0.5, atol=0.01)
    np.testing.assert_allclose(integrate(pdf * mean_mu), 2.0 / (3.0 * data.awr), atol=0.02)


@needs_library
@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_cosines_are_normalized_and_physical(temperature):
    data = read_thermal(NAME, temperature)
    assert np.all(np.abs(data.mu) <= 1.0)
    for a, b in zip(data.mu_offsets[:-1], data.mu_offsets[1:], strict=True):
        # Not strictly: neighbouring forward lines can tie at the file's 8 digits.
        assert np.all(np.diff(data.mu[a:b]) >= 0.0)
        assert np.all(np.diff(data.mu_cdf[a:b]) > 0.0)
    np.testing.assert_allclose(
        np.add.reduceat(data.mu_probability, data.mu_offsets[:-1]), 1.0, rtol=1e-12
    )
    np.testing.assert_allclose(data.mu_cdf[data.mu_offsets[1:] - 1], 1.0, rtol=1e-12)
