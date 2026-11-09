"""Probability-table bands: cross sections against OpenMC's formula from openmc.data's tables, and
the banded march against a dense oracle of every band combination."""

import os

import numpy as np
import pytest

from zerog.energy.elastic import elastic_row
from zerog.energy.grid import kinematic_alpha, limit_panels
from zerog.energy.subproblem import build_operators, solve_energy
from zerog.energy.urr import UrrData, band_mean, band_points, band_xs, in_range, node_bands

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)


def openmc_urr_xs(data, temperature, e, r):
    """calculate_urr_xs from openmc.data's objects, for elastic, capture and fission."""
    urr = data.urr[temperature]
    energy = np.asarray(urr.energy)
    table = np.asarray(urr.table)
    i = 0 if e == energy[0] else np.searchsorted(energy, e, side="left") - 1
    low = np.searchsorted(table[i, 0], r, side="right")
    up = np.searchsorted(table[i + 1, 0], r, side="right")
    f = (e - energy[i]) / (energy[i + 1] - energy[i])
    # Columns: cdf, total, elastic, fission, (n,gamma), heating.
    out = np.array([(1 - f) * table[i, col, low] + f * table[i + 1, col, up] for col in (2, 4, 3)])
    if urr.multiply_smooth:
        elastic = data[2].xs[temperature](e)
        fission = sum(
            data[mt].xs[temperature](e) for mt in (18, 19, 20, 21, 38) if mt in data.reactions
        )
        absorption = sum(
            rx.xs[temperature](e)
            for mt, rx in data.reactions.items()
            if (101 < mt <= 117 or 600 <= mt <= 849 or mt in (155, 182, 191, 192, 193, 197))
            and not rx.redundant
        )
        out *= np.array([elastic, absorption, fission])
    return np.maximum(out, 0.0)


@needs_library
@pytest.mark.parametrize("name", ["U238", "Pu241", "U235"])
def test_band_cross_sections_match_openmc_from_openmc_data(name):
    import openmc.data

    from zerog.data.library import nuclide_path
    from zerog.energy.urr import build_urr

    data = openmc.data.IncidentNeutron.from_hdf5(nuclide_path(name))
    table = data.urr["900K"]
    rng = np.random.default_rng(len(name))
    energies = np.sort(rng.uniform(table.energy[0], table.energy[-1], 40))[::-1]
    energies = np.concatenate([energies, np.asarray(table.energy[1:-1])[::-1][:5]])
    urr = build_urr([(name, "900K")], energies, 1)
    for k, e in enumerate(energies):
        assert in_range(urr.energy, urr.start, 0, e)
        for r in rng.uniform(0.0, 1.0, 5):
            ours = band_xs(
                urr.energy,
                urr.start,
                urr.bands,
                urr.cdf,
                urr.elastic,
                urr.fission,
                urr.capture,
                urr.multiply,
                urr.rule,
                0,
                e,
                r,
                urr.smooth[0, :, k].copy(),
            )
            np.testing.assert_allclose(ours, openmc_urr_xs(data, "900K", e, r), rtol=1e-10)
    assert not in_range(urr.energy, urr.start, 0, table.energy[0])
    assert not in_range(urr.energy, urr.start, 0, table.energy[-1])


def synthetic_case(n_points=16, factor_one=False, alpha_band=0.0, du=0.012, bottom=50.0):
    """Two terms on a small grid: term 0 banded, with three bands whose cdf moves between tables,
    and term 1 a smooth heavy scatterer. The grid holds every table energy and node pairs just
    inside the range's ends, as urr_required gives them."""
    from zerog.energy.urr import GAP

    low, high = 300.0, 3000.0
    table_energy = np.array([low, 700.0, 1500.0, high])
    required = np.concatenate([table_energy, [low * (1 + GAP), high * (1 - GAP)]])
    energy = np.unique(np.concatenate([np.geomspace(1e4, bottom, 40), required]))[::-1]
    energy = limit_panels(energy, du)
    n = energy.shape[0]
    if factor_one:
        cdf = np.ones((4, 1))
        values = np.ones((4, 1))
        el, fis, cap = values, values, values
    else:
        cdf = np.array([[0.2, 0.7, 1.0], [0.4, 0.8, 1.0], [0.3, 0.5, 1.0], [0.25, 0.75, 1.0]])
        el = np.array([[2.0, 8.0, 40.0], [3.0, 9.0, 30.0], [1.0, 6.0, 50.0], [2.0, 7.0, 20.0]])
        cap = np.array([[0.1, 1.0, 9.0], [0.2, 2.0, 6.0], [0.1, 0.5, 12.0], [0.3, 1.5, 4.0]])
        fis = np.zeros((4, 3))
    smooth = np.zeros((1, 4, n))
    smooth[0, 0] = 10.0
    smooth[0, 1] = 1.5
    urr = UrrData(
        terms=np.array([0]),
        start=np.array([0, 4]),
        bands=np.array([cdf.shape[1]]),
        energy=table_energy,
        cdf=cdf,
        elastic=el,
        fission=fis,
        capture=cap,
        multiply=np.array([factor_one]),
        rule=np.array([2]),
        smooth=smooth,
        points=band_points(1, n_points),
        weights=np.full(n_points, 1.0 / n_points),
    )
    removal = np.array([np.full(n, 11.5), np.full(n, 4.1)])
    scatter = np.array([np.full(n, 10.0), np.full(n, 4.0)])
    alpha = [alpha_band, kinematic_alpha(55.0)]
    return energy, removal, scatter, alpha, urr


def band_curves(energy, removal, scatter, urr):
    """Every combination's (total, elastic) curve of the banded term, at every node."""
    m = urr.points.shape[0]
    n = energy.shape[0]
    total = np.tile(removal[0], (m, 1))
    elastic = np.tile(scatter[0], (m, 1))
    for k, e in enumerate(energy):
        if not in_range(urr.energy, urr.start, 0, e):
            continue
        smooth = urr.smooth[0, :, k].copy()
        values = node_bands(urr, 0, e, smooth)
        total[:, k] = values.sum(axis=1) + smooth[3]
        elastic[:, k] = values[:, 0]
    assert total.shape == (m, n)
    return total, elastic


def banded_oracle(energy, removal, scatter, alpha, urr, d, streaming, source):
    """The banded system by direct sums, one unknown per node and combination."""
    m = urr.points.shape[0]
    n = energy.shape[0]
    p = 1.0 / m
    total, elastic = band_curves(energy, removal, scatter, urr)
    a = np.zeros((n * m, n * m))
    rows = [[elastic_row(energy, elastic[c], alpha[0], k) for k in range(n)] for c in range(m)]
    other = [elastic_row(energy, scatter[1], alpha[1], k) for k in range(n)]
    for k in range(n):
        for c in range(m):
            r = k * m + c
            a[r, r] = streaming + d[0] * total[c, k] + d[1] * removal[1, k]
            for c2 in range(m):
                # Band c2's emission feeds every band in proportion to its weight.
                a[r, np.arange(n) * m + c2] -= p * (d[0] * rows[c2][k] + d[1] * other[k])
    b = np.repeat(p * source, m)
    return np.linalg.solve(a, b).reshape(n, m)


@pytest.mark.parametrize("d", [(1.0, 1.0), (2.0, 0.5), (-0.3, 1.2)])
def test_banded_march_matches_the_dense_band_oracle(d):
    energy, removal, scatter, alpha, urr = synthetic_case()
    ops = build_operators(energy, removal, scatter, alpha, urr=urr)
    source = np.exp(-(((np.log(energy / energy[0]) + 0.2) / 0.1) ** 2))
    d = np.array(d)
    streaming = 2.0
    solution = solve_energy(ops, d, d[:, None], streaming, source)
    bands = banded_oracle(energy, removal, scatter, alpha, urr, d, streaming, source)
    np.testing.assert_allclose(solution.flux, bands.sum(axis=1), rtol=1e-10, atol=1e-14)
    total, elastic = band_curves(energy, removal, scatter, urr)
    flux = bands.sum(axis=1)
    np.testing.assert_allclose(
        solution.shield[0, 0], (total.T * bands).sum(axis=1) / flux, rtol=1e-10
    )
    np.testing.assert_allclose(
        solution.shield[0, 1], (elastic.T * bands).sum(axis=1) / flux, rtol=1e-10
    )


def test_bands_shield_the_absorption_where_the_tables_apply():
    energy, removal, scatter, alpha, urr = synthetic_case()
    ops = build_operators(energy, removal, scatter, alpha, urr=urr)
    source = np.exp(-(((np.log(energy / energy[0]) + 0.2) / 0.1) ** 2))
    solution = solve_energy(ops, [1.0, 1.0], [[1.0], [1.0]], 2.0, source)
    inside = (energy > 300.0) & (energy < 3000.0)
    capture = solution.shield[0, 2]
    # Self-shielding lowers the effective capture below the band mean, inside the range only.
    points = urr.points[:, 0]
    for k in np.nonzero(inside)[0][::5]:
        mean = np.mean(
            [
                band_xs(
                    urr.energy,
                    urr.start,
                    urr.bands,
                    urr.cdf,
                    urr.elastic,
                    urr.fission,
                    urr.capture,
                    urr.multiply,
                    urr.rule,
                    0,
                    energy[k],
                    r,
                    urr.smooth[0, :, k].copy(),
                )[1]
                for r in points
            ]
        )
        assert capture[k] < mean
    np.testing.assert_array_equal(capture[~inside], urr.smooth[0, 1, ~inside])


def test_unit_factor_tables_reproduce_the_smooth_solve():
    energy, removal, scatter, alpha, urr = synthetic_case(n_points=4, factor_one=True)
    # With factors of one, a band's total is the smooth elastic and capture: make removal agree.
    removal = removal.copy()
    removal[0] = urr.smooth[0, 0] + urr.smooth[0, 1]
    source = np.exp(-(((np.log(energy / energy[0]) + 0.2) / 0.1) ** 2))
    banded = solve_energy(
        build_operators(energy, removal, scatter, alpha, urr=urr),
        [1.0, 1.0],
        [[1.0], [1.0]],
        2.0,
        source,
    )
    smooth = solve_energy(
        build_operators(energy, removal, scatter, alpha), [1.0, 1.0], [[1.0], [1.0]], 2.0, source
    )
    np.testing.assert_allclose(banded.flux, smooth.flux, rtol=1e-12)


def test_banded_balance_closes_at_second_order():
    from zerog.energy.balance import energy_balance

    residuals = []
    for du in (0.012, 0.006):
        energy, removal, scatter, alpha, urr = synthetic_case(du=du)
        ops = build_operators(energy, removal, scatter, alpha, urr=urr)
        source = np.exp(-(((np.log(energy / energy[0]) + 0.2) / 0.1) ** 2))
        d = np.array([1.0, 1.0])
        solution = solve_energy(ops, d, d[:, None], 2.0, source)
        balance = energy_balance(ops, d, d[:, None], 2.0, source, solution.flux, solution.shield)
        residuals.append(abs(balance.residual / balance.source))
        with pytest.raises(ValueError, match="shield"):
            energy_balance(ops, d, d[:, None], 2.0, source, solution.flux)
    assert residuals[0] < 1e-4
    assert residuals[0] / residuals[1] > 3.0


def test_banded_march_with_a_window_edge_converges_to_the_oracle():
    """A heavy banded term's window edges take its shielded cross section, which is second order
    in panel width: the march converges to the dense oracle as panels are halved."""
    errors = []
    for du in (0.012, 0.006):
        energy, removal, scatter, alpha, urr = synthetic_case(
            n_points=4, alpha_band=kinematic_alpha(236.0), du=du, bottom=150.0
        )
        ops = build_operators(energy, removal, scatter, alpha, urr=urr)
        source = np.exp(-(((np.log(energy / energy[0]) + 0.2) / 0.1) ** 2))
        d = np.array([1.0, 1.0])
        solution = solve_energy(ops, d, d[:, None], 2.0, source)
        bands = banded_oracle(energy, removal, scatter, alpha, urr, d, 2.0, source)
        errors.append(np.abs(solution.flux / bands.sum(axis=1) - 1.0).max())
    assert errors[1] < 2e-3
    assert errors[0] / errors[1] > 2.5


def test_a_march_from_inside_a_range_is_the_whole_march():
    """With F zero above start, marching from start must give the full march, also when start
    and the node above it lie inside a range, as the upscatter block's marches may."""
    from zerog.energy.subproblem import march

    energy, removal, scatter, alpha, urr = synthetic_case(n_points=8)
    ops = build_operators(energy, removal, scatter, alpha, urr=urr)
    start = int(np.nonzero((energy < 1200.0) & (energy > 300.0))[0][0])
    source = np.where(np.arange(energy.shape[0]) >= start, 1.0 / energy**0.1, 0.0)
    d = np.array([1.0, 1.0])
    lagged = np.zeros((2, ops.inelastic.coarse.shape[0]))
    whole = march(ops, d, d[:, None], 2.0, source, lagged, 0)
    part = march(ops, d, d[:, None], 2.0, source, lagged, start)
    np.testing.assert_allclose(part[0], whole[0], rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(part[3][:, :, start:], whole[3][:, :, start:], rtol=1e-13)


def test_weighted_combinations_are_a_quadrature_over_bands():
    """At one energy a band's cross sections are steps in r, so any set of points converges as
    1 / points where a stratum straddles a cdf breakpoint. Sobol sets and products of strata
    converge to the same flux."""
    from zerog.energy.urr import product_points, with_points

    energy, removal, scatter, alpha, urr = synthetic_case(n_points=4)
    source = np.exp(-(((np.log(energy / energy[0]) + 0.2) / 0.1) ** 2))
    d = np.array([1.0, 1.0])

    def flux(points, weights):
        ops = build_operators(
            energy, removal, scatter, alpha, urr=with_points(urr, points, weights)
        )
        return solve_energy(ops, d, d[:, None], 2.0, source).flux

    reference = flux(*product_points(1024, 1))
    errors = []
    for m in (4, 16, 64, 256):
        f = flux(band_points(1, m), np.full(m, 1.0 / m))
        errors.append(np.abs(f / reference - 1.0).max())
    assert errors[-1] < 5e-3
    assert errors[-1] < 0.1 * errors[0]
    assert np.abs(flux(*product_points(256, 1)) / reference - 1.0).max() < 5e-3


@needs_library
@pytest.mark.parametrize("name", ["U238", "Pu241"])
def test_band_mean_is_the_exact_mean_over_r(name):
    from zerog.energy.urr import build_urr

    energies = np.geomspace(3.5e4, 2.2e4, 7)
    urr = build_urr([(name, "900K")], energies, 1)
    # Midpoints of fine strata integrate the steps in r to 1 / strata.
    r = (np.arange(20000) + 0.5) / 20000
    args = (urr.energy, urr.start, urr.bands, urr.cdf, urr.elastic, urr.fission, urr.capture)
    for k, e in enumerate(energies):
        smooth = urr.smooth[0, :, k].copy()
        samples = np.array(
            [band_xs(*args, urr.multiply, urr.rule, 0, e, x, smooth) for x in r]
        ).mean(axis=0)
        exact = band_mean(*args, urr.multiply, urr.rule, 0, e, smooth)
        np.testing.assert_allclose(samples, exact, rtol=2e-4)
