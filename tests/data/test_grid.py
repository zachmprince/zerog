"""Union grid and thinning checks: analytic curves, and real data when the library is present."""

import os

import numpy as np
import pytest

from zerog.data.grid import major_curves, thin_indices, thin_nuclide, union_grid
from zerog.data.neutron import read_neutron, read_reactions

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)


def no_required(x):
    return np.zeros(x.shape[0], dtype=np.bool_)


def worst_error(x, curve, kept):
    interpolated = np.interp(x[::-1], x[kept][::-1], curve[kept][::-1])[::-1]
    return np.abs(interpolated - curve)


def test_piecewise_linear_keeps_exactly_its_kinks():
    x = np.linspace(10.0, 0.0, 1001)
    curve = np.interp(x, [0.0, 3.0, 7.0, 10.0], [1.0, 4.0, 2.0, 5.0])
    kept = thin_indices(x, curve[None, :], no_required(x), 1e-10, 0.0)
    np.testing.assert_allclose(x[kept], [10.0, 7.0, 3.0, 0.0])


def test_parabola_meets_tolerance_with_expected_point_count():
    # Lin-lin error on x^2 is h^2 / 4 at a span's middle, so absolute tol needs h <= 2 sqrt(tol).
    x = np.linspace(1.0, 0.0, 100001)
    curve = x**2
    tol = 1e-4
    kept = thin_indices(x, curve[None, :], no_required(x), 0.0, tol)
    assert worst_error(x, curve, kept).max() <= tol * (1.0 + 1e-9)
    ideal = 1.0 / (2.0 * np.sqrt(tol))
    assert ideal <= len(kept) - 1 <= 2.0 * ideal


def test_every_curve_constrains_the_grid():
    x = np.linspace(4.0, 0.0, 401)
    first = np.interp(x, [0.0, 1.0, 4.0], [0.0, 1.0, 0.0])
    second = np.interp(x, [0.0, 3.0, 4.0], [0.0, 1.0, 0.0])
    kept = thin_indices(x, np.array([first, second]), no_required(x), 1e-10, 0.0)
    np.testing.assert_allclose(x[kept], [4.0, 3.0, 1.0, 0.0])


def test_required_points_are_kept():
    x = np.linspace(1.0, 0.0, 101)
    required = no_required(x)
    required[[17, 18, 60]] = True
    kept = thin_indices(x, np.ones((1, 101)), required, 0.0, 0.0)
    np.testing.assert_array_equal(kept, [0, 17, 18, 60, 100])


def test_zero_tolerance_drops_only_collinear_points():
    x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    curve = np.array([1.0, 2.0, 3.0, 7.0, 8.0])
    kept = thin_indices(x, curve[None, :], no_required(x), 0.0, 0.0)
    np.testing.assert_array_equal(kept, [0, 2, 3, 4])


def test_union_is_descending_without_duplicates():
    grid = union_grid([np.array([5.0, 3.0, 1.0]), np.array([4.0, 3.0, 2.0])])
    np.testing.assert_array_equal(grid, [5.0, 4.0, 3.0, 2.0, 1.0])


def test_non_monotone_feasibility_stays_within_tolerance():
    # The span 0 to 4 is feasible but 0 to 2 is not, so galloping stops early and stays valid.
    x = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    curve = np.array([0.0, 1.0, 0.6, 1.5, 2.0])
    kept = thin_indices(x, curve[None, :], no_required(x), 0.0, 0.5)
    np.testing.assert_array_equal(kept, [0, 1, 2, 4])
    assert worst_error(x, curve, kept).max() <= 0.5


def test_added_union_points_leave_the_interpolant_unchanged():
    x = np.linspace(1.0, 0.0, 2001)
    curve = 1.0 + np.sin(9.0 * x) ** 2
    kept = thin_indices(x, curve[None, :], no_required(x), 1e-3, 0.0)
    grid = union_grid([x[kept], np.array([0.123456, 0.654321])])
    values = np.interp(grid[::-1], x[kept][::-1], curve[kept][::-1])
    rebuilt = np.interp(x[::-1], grid[::-1], values)[::-1]
    assert np.all(np.abs(rebuilt - curve) <= 1e-3 * np.abs(curve) * (1.0 + 1e-9))


@needs_library
@pytest.mark.parametrize("name", ["U238", "Fe56", "Na23"])
def test_real_grid_meets_tolerance_for_every_reaction(name):
    rtol, atol = 1e-3, 1e-8
    data = read_reactions(name, "294K")
    grid = thin_nuclide(name, "294K", rtol, atol)
    mask = data.energy <= 2.0e7
    energy = data.energy[mask]
    assert 0 < len(grid) < len(energy)
    assert grid[0] == energy[0]
    assert grid[-1] == energy[-1]
    assert np.all(np.isin(grid, energy))
    for curve in data.xs[:, mask]:
        on_grid = np.interp(grid[::-1], energy[::-1], curve[::-1])
        rebuilt = np.interp(energy[::-1], grid[::-1], on_grid)[::-1]
        assert np.all(np.abs(rebuilt - curve) <= rtol * np.abs(curve) + atol * (1.0 + 1e-9))


@needs_library
def test_thresholds_are_kept():
    data = read_reactions("U238", "294K")
    grid = thin_nuclide("U238", "294K", 1e-2, 1e-6)
    for threshold in data.threshold[data.threshold <= 2.0e7]:
        assert threshold in grid


@needs_library
@pytest.mark.parametrize("name", ["U234", "U235", "Na23", "H1"])
def test_major_curves_match_the_reader(name):
    """U234 holds fission only in partial reactions, whose sum must match its stored MT 18."""
    curves = major_curves(read_reactions(name, "294K"))
    data = read_neutron(name, "294K")
    np.testing.assert_allclose(curves[0], data.total, rtol=1e-14)
    np.testing.assert_array_equal(curves[1], data.elastic)
    np.testing.assert_array_equal(curves[2], data.capture)
    np.testing.assert_allclose(curves[3], data.fission, rtol=1e-6, atol=1e-10)


@needs_library
@pytest.mark.parametrize("name", ["U238", "Fe56", "O16"])
def test_library_thresholds_are_continuous(name):
    """Thinning relies on threshold reactions starting from zero; catch a library with steps."""
    data = read_reactions(name, "294K")
    for threshold, curve in zip(data.threshold, data.xs, strict=True):
        if threshold > data.energy[-1]:
            assert curve[data.energy == threshold][0] == 0.0
