"""Fission emission and k-infinity: exact limits, and OpenMC's own sampling from
benchmarks/04-fission-spectra."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from zerog.data.scatter import MaxwellSpectrum
from zerog.energy.fission import _maxwell_points, truncate
from zerog.energy.inelastic import bin_segments

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)
ORACLE = Path(__file__).parents[2] / "benchmarks" / "04-fission-spectra"


def maxwell(theta, u):
    return MaxwellSpectrum(
        energy=np.array([2.0e7, 1.0e-5]),
        theta=np.array([theta, theta]),
        breakpoints_ascending=np.array([2]),
        interpolation_ascending=np.array([2]),
        u=u,
    )


def test_maxwell_pieces_have_the_maxwell_mean():
    lower, upper, w = _maxwell_points(maxwell(1.3e6, -3.0e7), 2.0e6, 1e-3)
    assert w.sum() == pytest.approx(1.0, rel=1e-12)
    # Each piece is spread uniformly in lethargy, whose mean energy is (upper - lower) / ln(ratio).
    inside = lower > 0.0
    mean = np.sum(
        w[inside] * (upper[inside] - lower[inside]) / np.log(upper[inside] / lower[inside])
    )
    assert mean == pytest.approx(1.5 * 1.3e6, rel=1e-6)


def test_maxwell_is_cut_at_its_restriction_energy():
    theta, e, u = 1.0e6, 2.5e6, 0.5e6
    lower, upper, w = _maxwell_points(maxwell(theta, u), e, 1e-3)
    assert upper.max() == pytest.approx(e - u)
    x = np.linspace(0.0, (e - u) / theta, 200001)
    density = np.sqrt(x) * np.exp(-x)
    expected = np.trapezoid(density * x, x) / np.trapezoid(density, x) * theta
    inside = lower > 0.0
    mean = np.sum(
        w[inside] * (upper[inside] - lower[inside]) / np.log(upper[inside] / lower[inside])
    )
    assert mean == pytest.approx(expected, rel=1e-5)


def test_truncation_keeps_the_share_below_the_top_and_renormalizes():
    lower = np.array([1.0e6, 1.0e7, 3.0e7])
    upper = np.array([2.0e6, 4.0e7, 3.1e7])
    _, hi, w = truncate(lower, upper, np.array([0.5, 0.4, 0.1]), 2.0e7)
    kept = np.array([0.5, 0.4 * math.log(2.0) / math.log(4.0)])
    np.testing.assert_allclose(w, kept / kept.sum())
    np.testing.assert_allclose(hi, [2.0e6, 2.0e7])


def oracle_cases():
    return sorted(p.stem for p in ORACLE.glob("*.json"))


@needs_library
@pytest.mark.parametrize("case", oracle_cases())
def test_fission_spectra_match_openmc_sampling(case):
    from zerog.data.neutron import read_reactions
    from zerog.energy.fission import chi_nodes, data_top, fission_components, spectrum_segments

    data = json.loads((ORACLE / f"{case}.json").read_text())
    name, e0 = data["nuclide"], data["source_energy"]
    edges = np.array(data["edges"])
    twin = np.array(data["mean"])
    curves = read_reactions(name, "294K")
    total = np.interp(e0, curves.energy[::-1], curves.xs.sum(axis=0)[::-1])
    top = data_top([(name, "294K")])

    def binned(spectrum, e):
        return bin_segments(*truncate(*spectrum_segments(spectrum, e, du=1e-3), top), edges)

    # The kernels store spectra at chi nodes and mix the two either side of an incoming energy, so
    # that mix is what is compared, not the spectrum at e0 itself.
    ours = np.zeros(twin.shape[0])
    for energy, production, spectrum in fission_components(name, "294K"):
        share = np.interp(e0, energy[::-1], production[::-1]) / total
        if share <= 0.0:
            continue
        nodes, histogram = chi_nodes(spectrum, top, 1.0e3)
        j = np.searchsorted(-nodes, -e0, side="right") - 1
        a, b = nodes[j], nodes[j + 1]
        assert a >= e0 > b
        if histogram:
            ours += share * binned(spectrum, 0.5 * (a + b))
        else:
            w = (e0 - b) / (a - b)
            ours += share * (w * binned(spectrum, a) + (1.0 - w) * binned(spectrum, b))
    ours *= data["collisions"]
    histories = data["provenance"]["particles"] * data["provenance"]["batches"]
    assert ours.sum() == pytest.approx(twin.sum(), abs=4.0 * np.sqrt(ours.sum() / histories))
    # Each fission neutron scores 1 / histories, so a bin's variance is about its expected count
    # over histories squared.
    seen = (ours > 1e-3 / histories) | (twin > 0.0)
    sd = np.sqrt(np.maximum(ours[seen], 1.0 / histories) / histories)
    chi_squared = float(np.sum(((ours[seen] - twin[seen]) / sd) ** 2))
    assert chi2.sf(chi_squared, int(seen.sum())) > 1e-4, (chi_squared, int(seen.sum()))


def test_maxwell_cut_at_a_positive_restriction_energy_needs_few_nodes():
    # Partial fission (U236, U240) has later chances whose Maxwell is cut at E - u with u > 0.
    from zerog.energy.fission import chi_nodes, spectrum_segments

    spectrum = maxwell(1.0e6, 2.6e6)
    nodes, histogram = chi_nodes(spectrum, 2.0e7, 50.0)
    assert not histogram
    assert nodes.shape[0] < 1000
    assert 2.6e6 in nodes
    assert np.all(np.diff(nodes) < 0.0)
    for e in nodes:
        _, upper, w = truncate(*spectrum_segments(spectrum, e), 2.0e7)
        if e <= 2.6e6:
            assert w.shape[0] == 0 or w.sum() == 0.0
        else:
            assert w.sum() == pytest.approx(1.0, rel=1e-12)
            assert upper.max() <= e - 2.6e6 * (1.0 - 1e-12)


def fission_case(names, top=2.0e7, bottom=1.0e3, nodes=301, coarse_du=0.02, temperature="294K"):
    from zerog.energy.fission import build_fission, fission_nodes
    from zerog.energy.inelastic import coarse_grid

    pairs = [(n, temperature) for n in names]
    fine = np.geomspace(top, bottom, nodes)
    coarse = coarse_grid(fine, coarse_du)
    energy = np.unique(np.concatenate([fine, coarse, fission_nodes(pairs, top, bottom)]))[::-1]
    return energy, coarse, build_fission(pairs, energy, coarse, top, du=0.005)


@needs_library
def test_every_row_places_all_its_neutrons():
    _, _, kernels = fission_case(["U238", "Pu241"])
    np.testing.assert_allclose(kernels.rows.sum(axis=1) + kernels.lost, 1.0, rtol=1e-12)
    # Below 1 keV a prompt spectrum holds about 1e-5 and the softer delayed ones up to 3e-4.
    assert kernels.lost.max() < 1e-3


@needs_library
def test_emission_and_loss_add_up_to_production():
    from zerog.energy.fission import fission_hats, production_rates
    from zerog.energy.grid import rate_weights

    energy, _, kernels = fission_case(["U235", "U238", "Pu241"])
    weights = rate_weights(energy)
    flux = 1.0 + 0.5 * np.sin(np.log(energy))
    density = np.array([0.001, 0.02, 0.003])
    values, lost = fission_hats(kernels, density, energy, flux, weights)
    produced = production_rates(kernels, density, energy, flux, weights).sum()
    assert values.sum() + lost == pytest.approx(produced, rel=1e-12)


@needs_library
def test_node_weights_restrict_production_exactly():
    from zerog.energy.fission import node_weights
    from zerog.energy.grid import panel_rate, rate_weights

    energy, _, kernels = fission_case(["U238"])
    weights = rate_weights(energy)
    flux = 1.0 + 0.5 * np.sin(np.log(energy))
    for k in range(kernels.term.shape[0]):
        w = node_weights(kernels, k, energy, flux, weights)
        expected = panel_rate(kernels.production[k], flux, weights)
        assert w.sum() == pytest.approx(expected, rel=1e-13)
        assert np.all(w >= 0.0)


@needs_library
def test_spectrum_between_chi_nodes_is_their_linear_mix():
    # U238's tables change their top outgoing energy between 4 and 5 MeV, so that interval is split.
    from zerog.energy.fission import chi_nodes, fission_components, spectrum_segments
    from zerog.energy.inelastic import deposit

    _, coarse, _ = fission_case(["U238"])
    _, _, spectrum = fission_components("U238", "294K")[0]
    nodes, histogram = chi_nodes(spectrum, 2.0e7, 1.0e3)
    assert not histogram
    split = nodes[(nodes < 5.0e6) & (nodes > 4.0e6)]
    assert split.shape[0] > 10
    a, b = split[3], split[4]
    e = 0.3 * a + 0.7 * b

    def row(at):
        return deposit(*truncate(*spectrum_segments(spectrum, at, du=0.005), 2.0e7), coarse)[0]

    mixed = 0.3 * row(a) + 0.7 * row(b)
    assert np.abs(row(e) - mixed).max() < 1e-6


def dense_k(apply, n):
    """The largest eigenvalue of a linear map, built column by column."""
    matrix = np.column_stack([apply(np.eye(n)[j]) for j in range(n)])
    values = np.linalg.eigvals(matrix)
    return values[np.argmax(values.real)].real


@needs_library
@pytest.mark.parametrize("scatter", [False, True])
def test_power_iteration_finds_the_dominant_eigenvalue(scatter):
    # Without scattering, F = source / Sigma_t, so the dense map is explicit. With isotropic elastic
    # scattering, each column is a solve; either way power iteration must reach its top eigenvalue.
    from zerog.energy.eigen import solve_k
    from zerog.energy.fission import fission_source
    from zerog.energy.inelastic import build_inelastic
    from zerog.energy.subproblem import build_operators, solve_energy

    names = ["U235", "U238"]
    energy, coarse, fission = fission_case(names, nodes=121, coarse_du=0.05)
    smooth = np.array([3.0 + 0.1 * np.log(2.0e7 / energy), 6.0 + 0.2 * np.log(2.0e7 / energy)])
    sigma_s = 0.5 * smooth if scatter else np.zeros_like(smooth)
    kernels = build_inelastic([None, None], energy, coarse)
    ops = build_operators(energy, smooth, sigma_s, [0.0, 0.0], inelastic=kernels)
    density = np.array([0.002, 0.02])
    solution = solve_k(ops, fission, density, density[:, None], 0.0, density)

    def apply(flux):
        source = fission_source(ops, fission, density, flux)
        return solve_energy(ops, density, density[:, None], 0.0, source).flux

    assert solution.k == pytest.approx(dense_k(apply, energy.shape[0]), rel=1e-8)
    assert solution.iterations < 20
