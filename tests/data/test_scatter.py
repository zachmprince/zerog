"""Scattering law checks against openmc.data, and analytic kinematics."""

import os
from functools import cache

import h5py
import numpy as np
import openmc.data
import openmc.stats
import pytest

from zerog.data.library import nuclide_path
from zerog.data.neutron import read_fission, read_neutron, read_reactions
from zerog.data.scatter import (
    CorrelatedLaw,
    EvaporationSpectrum,
    KalbachMannLaw,
    LevelEnergy,
    MaxwellSpectrum,
    NBodyLaw,
    TabularSpectrum,
    UncorrelatedLaw,
    read_law,
    read_scattering_laws,
)

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)
# Covers the PWR and SFR materials, and an example of every law form in the library.
NUCLIDES = ["H1", "O16", "U235", "U238", "Pu239", "Na23", "Fe56", "Zr90", "Sn118", "Cr52", "Mo95"]
NUCLIDES += ["H2", "O18", "F19", "Be9", "Am244", "Al27", "Ag107"]
RULES = {"histogram": 1, "linear-linear": 2}


@cache
def reference(name):
    return openmc.data.IncidentNeutron.from_hdf5(nuclide_path(name))


def assert_spectrum_matches(spectrum, dist):
    assert np.all(np.diff(spectrum.energy) < 0.0)
    np.testing.assert_array_equal(spectrum.energy[::-1], dist.energy)
    np.testing.assert_array_equal(spectrum.breakpoints_ascending, dist.breakpoints)
    np.testing.assert_array_equal(spectrum.interpolation_ascending, dist.interpolation)
    n = len(spectrum.energy)
    assert spectrum.offsets[0] == 0
    assert spectrum.offsets[-1] == len(spectrum.energy_out)
    for i in range(n):
        s = slice(spectrum.offsets[i], spectrum.offsets[i + 1])
        table = dist.energy_out[n - 1 - i]
        np.testing.assert_array_equal(spectrum.energy_out[s][::-1], table.x)
        np.testing.assert_array_equal(spectrum.pdf[s][::-1], table.p)
        np.testing.assert_array_equal(spectrum.cdf[s][::-1], table.c)
        assert spectrum.interpolation[i] == RULES[table.interpolation]


def assert_parametric_matches(law, dist):
    np.testing.assert_array_equal(law.energy[::-1], dist.theta.x)
    np.testing.assert_array_equal(law.theta[::-1], dist.theta.y)
    np.testing.assert_array_equal(law.breakpoints_ascending, dist.theta.breakpoints)
    np.testing.assert_array_equal(law.interpolation_ascending, dist.theta.interpolation)
    assert law.u == dist.u
    assert np.all(law.theta > 0.0)


def assert_uncorrelated_matches(law, dist):
    if dist.angle is None:
        assert law.angle is None
    else:
        angle = law.angle
        n = len(angle.energy)
        np.testing.assert_array_equal(angle.energy[::-1], dist.angle.energy)
        for i in range(n):
            s = slice(angle.offsets[i], angle.offsets[i + 1])
            table = dist.angle.mu[n - 1 - i]
            np.testing.assert_array_equal(angle.mu[s], table.x)
            np.testing.assert_array_equal(angle.pdf[s], table.p)
            np.testing.assert_array_equal(angle.cdf[s], table.c)
            assert angle.interpolation[i] == RULES[table.interpolation]
            assert np.all(np.diff(angle.mu[s]) > 0.0)

    energy = law.energy
    if dist.energy is None:
        assert energy is None
    elif isinstance(energy, LevelEnergy):
        assert energy.threshold == dist.energy.threshold
        assert energy.mass_ratio == dist.energy.mass_ratio
    elif isinstance(energy, MaxwellSpectrum | EvaporationSpectrum):
        kind = {
            MaxwellSpectrum: openmc.data.MaxwellEnergy,
            EvaporationSpectrum: openmc.data.Evaporation,
        }
        assert isinstance(dist.energy, kind[type(energy)])
        assert_parametric_matches(energy, dist.energy)
    else:
        assert isinstance(energy, TabularSpectrum)
        assert_spectrum_matches(energy, dist.energy)


def assert_kalbach_matches(law, dist):
    assert_spectrum_matches(law.spectrum, dist)
    n = len(law.spectrum.energy)
    for i in range(n):
        s = slice(law.spectrum.offsets[i], law.spectrum.offsets[i + 1])
        np.testing.assert_array_equal(law.precompound[s][::-1], dist.precompound[n - 1 - i].y)
        np.testing.assert_array_equal(law.slope[s][::-1], dist.slope[n - 1 - i].y)


def assert_correlated_matches(law, dist):
    assert_spectrum_matches(law.spectrum, dist)
    n = len(law.spectrum.energy)
    for i in range(n):
        start = law.spectrum.offsets[i]
        points = law.spectrum.offsets[i + 1] - start
        tables = dist.mu[n - 1 - i]
        assert len(tables) == points
        for j in range(points):
            k = start + points - 1 - j  # flipped within the outgoing table
            s = slice(law.mu_start[k], law.mu_stop[k])
            table = tables[j]
            assert isinstance(table, openmc.stats.Tabular)
            np.testing.assert_array_equal(law.mu[s], table.x)
            np.testing.assert_array_equal(law.mu_pdf[s], table.p)
            np.testing.assert_array_equal(law.mu_cdf[s], table.c)
            assert law.mu_rule[k] == RULES[table.interpolation]


@needs_library
@pytest.mark.parametrize("name", NUCLIDES)
def test_laws_match_openmc(name):
    ref = reference(name)
    laws = read_scattering_laws(name)
    expected = sorted(
        rx.mt
        for rx in ref.reactions.values()
        if not rx.redundant
        and rx.mt not in (18, 19, 20, 21, 38)
        and any(p.particle == "neutron" for p in rx.products)
    )
    assert [law.mt for law in laws] == expected
    for reaction in laws:
        rx = ref[reaction.mt]
        assert reaction.center_of_mass == rx.center_of_mass
        product = next(p for p in rx.products if p.particle == "neutron")
        assert len(reaction.laws) == len(product.distribution)
        for g, (law, dist) in enumerate(zip(reaction.laws, product.distribution, strict=True)):
            if isinstance(law, UncorrelatedLaw):
                assert_uncorrelated_matches(law, dist)
            elif isinstance(law, KalbachMannLaw):
                assert_kalbach_matches(law, dist)
            elif isinstance(law, CorrelatedLaw):
                assert_correlated_matches(law, dist)
            else:
                assert isinstance(law, NBodyLaw)
                assert law.n_particles == dist.n_particles
                assert law.total_mass == dist.total_mass
                assert law.atomic_weight_ratio == dist.atomic_weight_ratio
                assert law.q_value == dist.q_value
            weight = reaction.applicability[g]
            if not product.applicability:
                assert weight is None
                continue
            stored = product.applicability[g]
            np.testing.assert_array_equal(weight.energy[::-1], stored.x)
            np.testing.assert_array_equal(weight.probability[::-1], stored.y)
            np.testing.assert_array_equal(weight.breakpoints_ascending, stored.breakpoints)
            np.testing.assert_array_equal(weight.interpolation_ascending, stored.interpolation)


@needs_library
@pytest.mark.parametrize("name", ["U238", "Fe56", "Na23", "Zr90", "O16", "Ag110_m1"])
def test_level_kinematics_match_q_and_mass(name):
    """E' in the centre of mass is (A / (A + 1))^2 (E - threshold), threshold = -(A + 1) / A Q."""
    awr = read_neutron(name, "294K").awr
    data = read_reactions(name, "294K")
    q = dict(zip(data.mt.tolist(), data.q, strict=True))
    levels = 0
    for reaction in read_scattering_laws(name):
        energy = reaction.laws[0].energy if isinstance(reaction.laws[0], UncorrelatedLaw) else None
        if isinstance(energy, LevelEnergy):
            levels += 1
            assert reaction.center_of_mass
            assert energy.mass_ratio == pytest.approx((awr / (awr + 1.0)) ** 2, rel=1e-6)
            assert energy.threshold == pytest.approx(-(awr + 1.0) / awr * q[reaction.mt], rel=1e-6)
    assert levels > 0


@needs_library
@pytest.mark.parametrize("name", ["U238", "Fe56", "H1", "Na23"])
def test_angle_tables_integrate_to_their_cdf_and_one(name):
    for reaction in read_scattering_laws(name):
        for law in reaction.laws:
            if not isinstance(law, UncorrelatedLaw) or law.angle is None:
                continue
            a = law.angle
            for i in range(len(a.energy)):
                s = slice(a.offsets[i], a.offsets[i + 1])
                mu, pdf, cdf = a.mu[s], a.pdf[s], a.cdf[s]
                assert mu[0] == -1.0
                assert mu[-1] == 1.0
                assert cdf[-1] == pytest.approx(1.0, abs=1e-6)
                if a.interpolation[i] == 1:
                    increment = pdf[:-1] * np.diff(mu)
                else:
                    increment = 0.5 * (pdf[:-1] + pdf[1:]) * np.diff(mu)
                np.testing.assert_allclose(np.diff(cdf), increment, rtol=1e-2, atol=1e-5)


@needs_library
@pytest.mark.parametrize("name", ["U238", "Fe56", "Zr90"])
def test_correlated_and_kalbach_tables_end_at_one(name):
    found = 0
    for reaction in read_scattering_laws(name):
        for law in reaction.laws:
            if not isinstance(law, CorrelatedLaw | KalbachMannLaw):
                continue
            found += 1
            spectrum = law.spectrum
            for i in range(len(spectrum.energy)):
                s = slice(spectrum.offsets[i], spectrum.offsets[i + 1])
                # The top of the ascending table, which is first here, closes the cdf.
                assert spectrum.cdf[s][0] == pytest.approx(1.0, abs=1e-5)
                assert np.all(np.diff(spectrum.cdf[s]) <= 1e-12)
            if isinstance(law, CorrelatedLaw):
                assert np.all(law.mu_stop > law.mu_start)
                assert np.all(np.abs(law.mu) <= 1.0)
    assert found > 0


@needs_library
def test_h1_elastic_is_only_law_and_isotropic_cm_at_low_energy():
    laws = read_scattering_laws("H1")
    assert [law.mt for law in laws] == [2]
    assert laws[0].center_of_mass
    law = laws[0].laws[0]
    assert law.energy is None
    i = len(law.angle.energy) - 1
    s = slice(law.angle.offsets[i], law.angle.offsets[i + 1])
    np.testing.assert_allclose(law.angle.pdf[s], 0.5, rtol=1e-6)


@needs_library
def test_several_laws_select_with_probabilities_summing_to_one():
    found = 0
    for reaction in read_scattering_laws("Am244"):
        if len(reaction.laws) < 2:
            continue
        found += 1
        first = reaction.applicability[0]
        for weight in reaction.applicability:
            np.testing.assert_array_equal(weight.energy, first.energy)
        summed = sum(weight.probability for weight in reaction.applicability)
        np.testing.assert_allclose(summed, 1.0, atol=2e-6)
    assert found > 0


@needs_library
def test_single_law_applicability_is_not_a_neutron_weight():
    """Applying this p would lose half of O18's (n,2n) neutrons, so it must stay documented."""
    reaction = next(law for law in read_scattering_laws("O18") if law.mt == 16)
    assert len(reaction.laws) == 1
    assert reaction.applicability[0].probability.max() == 0.5


def test_discrete_lines_are_refused(tmp_path):
    with h5py.File(tmp_path / "law.h5", "w") as f:
        group = f.create_group("law")
        group.attrs["type"] = np.bytes_("uncorrelated")
        energy = group.create_group("energy")
        energy.attrs["type"] = np.bytes_("continuous")
        grid = energy.create_dataset("energy", data=[1.0, 2.0])
        grid.attrs["interpolation"] = np.array([[2], [2]])
        tables = energy.create_dataset("distribution", data=np.zeros((3, 4)))
        tables.attrs["offsets"] = np.array([0, 2])
        tables.attrs["interpolation"] = np.array([2, 2])
        tables.attrs["n_discrete_lines"] = np.array([1, 0])
        with pytest.raises(NotImplementedError, match="discrete lines"):
            read_law(group)


@needs_library
@pytest.mark.parametrize("name", ["Th232", "Pa231", "Pa233"])
def test_correlated_fission_spectra_keep_everything(name):
    ref = reference(name)
    products = ref[18].products
    prompt = next(p for p in products if p.emission_mode == "prompt")
    delayed = [p for p in products if p.emission_mode == "delayed"]
    energy = ref.energy["294K"]
    assert not ref[18].center_of_mass
    f = read_fission(read_neutron(name, "294K"))[0]
    assert isinstance(f.chi_prompt, CorrelatedLaw)
    assert_correlated_matches(f.chi_prompt, prompt.distribution[0])
    np.testing.assert_allclose(f.nu_prompt[::-1], prompt.yield_(energy), rtol=1e-14)
    assert len(delayed) == 6
    assert f.nu_delayed.shape == (6, len(energy))
    for g, product in enumerate(delayed):
        np.testing.assert_allclose(f.nu_delayed[g, ::-1], product.yield_(energy), rtol=1e-14)
        assert_spectrum_matches(f.chi_delayed[g], product.distribution[0].energy)
