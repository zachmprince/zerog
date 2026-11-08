"""Energy subproblem checks against analytic slowing-down limits and a direct-sum oracle."""

import os

import numpy as np
import pytest
from scipy.integrate import quad

from zerog.energy.balance import energy_balance
from zerog.energy.elastic import elastic_row
from zerog.energy.grid import (
    build_grid,
    centred_weights,
    emission_weights,
    interpolate,
    kinematic_alpha,
    limit_panels,
    node_pairs,
    panel_rate,
    rate_weights,
    square_weights,
)
from zerog.energy.subproblem import build_operators, solve_energy

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)

TOP = 1.0e4


def random_grid(seed, n, bottom, max_du):
    rng = np.random.default_rng(seed)
    inner = np.exp(rng.uniform(np.log(bottom), np.log(TOP), n))
    return limit_panels(np.unique(np.concatenate([inner, [bottom, TOP]]))[::-1], max_du)


def inflow(energy, sigma, alpha):
    """Scattering in from an asymptotic F = 1 above the grid, per unit lethargy."""
    return sigma * np.clip((energy / energy[0] - alpha) / (1.0 - alpha), 0.0, None)


def test_panel_weights_match_quadrature():
    energy = np.array([10.0, 7.0, 7.0 - 1e-6, 1.0])
    sigma = np.array([1.0, -2.0, 3.0, 0.5])
    flux = np.array([2.0, 5.0, -1.0, 4.0])
    rates = rate_weights(energy)
    emissions = emission_weights(energy)
    squares = square_weights(energy)
    centred = centred_weights(energy)
    for i in range(3):
        h = np.log(energy[i] / energy[i + 1])

        def s_of(s, i=i, h=h):
            e = energy[i] * np.exp(-s)
            g = (energy[i] - e) / (energy[i] - energy[i + 1])
            return (
                e,
                sigma[i] * (1 - g) + sigma[i + 1] * g,
                flux[i] + (flux[i + 1] - flux[i]) * s / h,
            )

        rate = quad(lambda s: s_of(s)[1] * s_of(s)[2], 0.0, h, epsabs=1e-13, epsrel=1e-13)[0]
        emission = quad(
            lambda s: s_of(s)[1] * s_of(s)[2] * energy[i] / s_of(s)[0],  # noqa: B023
            0.0,
            h,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        square = quad(
            lambda s: s_of(s)[1] * s_of(s)[2] * (energy[i] / s_of(s)[0]) ** 2,  # noqa: B023
            0.0,
            h,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        r = rates[i]
        e = emissions[i]
        q = squares[i]
        assert sigma[i] * (r[0] * flux[i] + r[1] * flux[i + 1]) + sigma[i + 1] * (
            r[2] * flux[i] + r[3] * flux[i + 1]
        ) == pytest.approx(rate, rel=1e-11, abs=1e-12)
        assert sigma[i] * (e[0] * flux[i] + e[1] * flux[i + 1]) + sigma[i + 1] * (
            e[2] * flux[i] + e[3] * flux[i + 1]
        ) == pytest.approx(emission, rel=1e-11, abs=1e-12)
        assert sigma[i] * (q[0] * flux[i] + q[1] * flux[i + 1]) + sigma[i + 1] * (
            q[2] * flux[i] + q[3] * flux[i + 1]
        ) == pytest.approx(square, rel=1e-11, abs=1e-12)
        for power, c in enumerate(centred[:, i]):
            # (E_i / E - 1) ** p is tiny on a narrow panel, so tolerances scale with h ** p.
            centred_moment = quad(
                lambda s, p=power: s_of(s)[1] * s_of(s)[2] * np.expm1(s) ** p,
                0.0,
                h,
                epsabs=1e-13 * h**power,
                epsrel=1e-13,
            )[0]
            assert sigma[i] * (c[0] * flux[i] + c[1] * flux[i + 1]) + sigma[i + 1] * (
                c[2] * flux[i] + c[3] * flux[i + 1]
            ) == pytest.approx(centred_moment, rel=1e-11, abs=1e-12 * h**power)


@pytest.mark.parametrize("awr", [0.9991673, 15.857, 236.0])
def test_pure_scatterer_is_exactly_one_over_e_on_any_grid(awr):
    alpha = kinematic_alpha(awr)
    energy = random_grid(1, 300, 1.0, min(0.5, 0.9 * -np.log(alpha)))
    sigma = np.full((1, energy.shape[0]), 4.0)
    ops = build_operators(energy, sigma, sigma, [alpha])
    solution = solve_energy(ops, [1.0], [[1.0]], 0.0, inflow(energy, 4.0, alpha))
    np.testing.assert_allclose(solution.flux, 1.0, rtol=1e-11, atol=0.0)


@pytest.mark.parametrize("awr", [2.0, 15.857, 236.0])
def test_placzek_transient_in_the_first_window(awr):
    # A unit source at E0 gives sigma F = (E0 / E)^(alpha / (1 - alpha)) / (1 - alpha) for
    # alpha E0 < E < E0, where only first-collision neutrons start the chain.
    alpha = kinematic_alpha(awr)
    energy = np.geomspace(TOP, alpha * TOP, 2001)
    sigma = np.full((1, energy.shape[0]), 2.0)
    ops = build_operators(energy, sigma, sigma, [alpha])
    source = energy / energy[0] / (1.0 - alpha)
    flux = solve_energy(ops, [1.0], [[1.0]], 0.0, source).flux
    expected = (TOP / energy) ** (alpha / (1.0 - alpha)) / (1.0 - alpha) / 2.0
    np.testing.assert_allclose(flux, expected, rtol=2e-6)


def placzek_case(awr, windows, per_window):
    alpha = kinematic_alpha(awr)
    q = -np.log(alpha)
    energy = TOP * np.exp(-np.linspace(0.0, windows * q, windows * per_window + 1))
    # The source stops at alpha E0, so F needs a node on each side of the jump.
    energy = np.unique(np.concatenate([energy, node_pairs([alpha * TOP])]))[::-1]
    sigma = np.ones((1, energy.shape[0]))
    ops = build_operators(energy, sigma, sigma, [alpha])
    source = np.where(energy >= alpha * TOP, energy / TOP / (1.0 - alpha), 0.0)
    return energy, alpha, q, solve_energy(ops, [1.0], [[1.0]], 0.0, source).flux


def test_placzek_second_window_reaches_the_edge():
    energy, alpha, q, flux = placzek_case(11.9, 2, 800)
    y = np.log(TOP / energy)
    kappa = 1.0 / (1.0 - alpha)
    inside = (y > q * (1.0 + 1e-6)) & (y < 2.0 * q * (1.0 - 1e-6))
    expected = (
        np.exp(alpha * y * kappa) * kappa * (1.0 - np.exp(-kappa * q) * (1.0 + kappa * (y - q)))
    )
    np.testing.assert_allclose(flux[inside], expected[inside], rtol=1e-5)


def test_placzek_asymptote_is_one_over_xi():
    _, alpha, _, flux = placzek_case(11.9, 40, 200)
    xi = 1.0 + alpha * np.log(alpha) / (1.0 - alpha)
    assert flux[-1] * xi == pytest.approx(1.0, rel=1e-6)


def test_hydrogen_window_edge_on_the_full_range():
    alpha = kinematic_alpha(0.9991673)
    energy = np.geomspace(2.0e7, 1.0e-2, 4001)
    sigma = np.full((1, energy.shape[0]), 4.0)
    ops = build_operators(energy, sigma, sigma, [alpha])
    assert energy[-1] / alpha < energy[0]
    flux = solve_energy(ops, [1.0], [[1.0]], 0.0, inflow(energy, 4.0, alpha)).flux
    np.testing.assert_allclose(flux, 1.0, rtol=1e-11, atol=0.0)


def test_relative_accuracy_holds_when_flux_falls_faster_than_e():
    # A strongly removing PGD-like mode. Forward substitution on the direct sum has no cancellation,
    # so the march must keep its relative accuracy far into the tail.
    alpha = kinematic_alpha(236.0)
    energy = np.geomspace(2.0e7, 1.0e-2, 2301)
    n = energy.shape[0]
    sigma = np.ones((1, n)) * (1.0 + 0.3 * np.sin(np.log(energy) * 40.0))
    ops = build_operators(energy, sigma, sigma, [alpha])
    source = np.zeros(n)
    source[0] = 1.0
    streaming = 1.0
    flux = solve_energy(ops, [1.0], [[1.0]], streaming, source).flux
    reference = np.zeros(n)
    reference[0] = 1.0 / (streaming + sigma[0, 0])
    for k in range(1, n):
        row = elastic_row(energy, sigma[0], alpha, k)
        reference[k] = (row[:k] @ reference[:k]) / (streaming + sigma[0, k] - row[k])
    resolved = reference > 1e-280  # below this the flux underflows
    assert reference[resolved][-1] / reference[0] < 1e-270
    np.testing.assert_allclose(flux[resolved], reference[resolved], rtol=1e-11, atol=0.0)


def lorentzian(energy, center, width, peak):
    return peak / (1.0 + ((energy - center) / width) ** 2)


def resonance_error(width_ratio, wide, reach=1.0):
    """Worst error against a limit, with a resonance in a nuclide of mass 236.

    The moderator's emission is a fixed flat source, so neutrons are not depleted through the
    resonance and only the absorber's kernel can move F from the limit.
    """
    center = 1000.0
    alpha = kinematic_alpha(236.0)
    window = -np.log(alpha)
    width = width_ratio * window * center
    grid = np.concatenate(
        [center + width * np.sinh(np.linspace(-6.0, 6.0, 4001)), np.geomspace(TOP, 100.0, 1501)]
    )
    energy = limit_panels(np.unique(grid[(grid <= TOP) & (grid >= 100.0)])[::-1], 0.9 * window)
    peak = lorentzian(energy, center, width, 1.0)
    capture = 50.0 * peak
    scatter = 200.0 * peak
    moderator = np.full_like(peak, 20.0)
    ops = build_operators(
        energy,
        np.array([moderator, capture + scatter]),
        np.array([np.zeros_like(peak), scatter]),
        [0.0, alpha],
    )
    flux = solve_energy(ops, [1.0, 1.0], [[0.0], [1.0]], 0.0, moderator).flux
    expected = moderator / (moderator + capture + (0.0 if wide else scatter))
    near = np.abs(energy - center) < reach * width
    return np.abs(flux[near] / expected[near] - 1.0).max()


def test_narrow_resonance_limit():
    errors = [resonance_error(ratio, wide=False) for ratio in (0.01, 0.001, 0.0001)]
    assert errors[0] / errors[1] == pytest.approx(10.0, rel=0.1)
    assert errors[1] / errors[2] == pytest.approx(10.0, rel=0.1)
    assert errors[2] < 1e-3


def test_narrow_resonance_first_correction_is_the_absorber_emission():
    # At the centre F = (moderator + Y) / total, with Y from the limiting F above the centre. This
    # fixes the size of the kernel, which the scaling above does not.
    alpha = kinematic_alpha(236.0)
    ratio = 0.001
    width = ratio * -np.log(alpha) * 1000.0

    def integrand(e):
        peak = lorentzian(e, 1000.0, width, 1.0)
        return 200.0 * peak * 20.0 / (20.0 + 250.0 * peak) / e**2

    y = 1000.0 / (1.0 - alpha) * quad(integrand, 1000.0, 1000.0 / alpha, points=[1000.0 + width])[0]
    error = resonance_error(ratio, wide=False, reach=1e-3)
    assert error == pytest.approx(y / 20.0, rel=0.02)


def test_wide_resonance_limit():
    errors = [resonance_error(ratio, wide=True, reach=0.1) for ratio in (3.0, 30.0)]
    assert errors[0] / errors[1] > 10.0
    assert errors[1] < 1e-2


def oracle_matrix(ops, removal, scatter, streaming):
    n = ops.energy.shape[0]
    a = np.zeros((n, n))
    a[np.arange(n), np.arange(n)] = streaming + removal @ ops.removal
    for t in range(ops.removal.shape[0]):
        for k in range(1, n):
            a[k] -= scatter[t, 0] * elastic_row(ops.energy, ops.scatter[t], ops.alpha[t], k)
    return a


def two_term_case(seed=3):
    energy = random_grid(seed, 120, 20.0, 0.015)
    rng = np.random.default_rng(seed)
    wiggle = 1.0 + 0.5 * np.sin(rng.uniform(0.5, 2.0) * np.log(energy) * 20.0)
    scatter = np.array([5.0 * wiggle, np.full_like(energy, 3.0)])
    removal = scatter + np.array([0.1 * wiggle, 0.2 * np.sqrt(TOP / energy) * 1e-2])
    ops = build_operators(energy, removal, scatter, [kinematic_alpha(236.0), kinematic_alpha(23.0)])
    return energy, ops


@pytest.mark.parametrize(
    ("d_removal", "d_scatter", "streaming"),
    [
        ([1.0, 2.0], [[1.0], [2.0]], 0.0),
        ([1.0, -0.2], [[-0.5], [1.5]], 3.0),
        ([-0.3, 0.1], [[0.2], [-0.7]], -9.0),
    ],
)
def test_march_matches_direct_sum_including_negative_densities(d_removal, d_scatter, streaming):
    energy, ops = two_term_case()
    d_removal = np.array(d_removal)
    d_scatter = np.array(d_scatter)
    source = np.exp(-(((np.log(energy / energy[0]) + 0.1) / 0.05) ** 2))
    solution = solve_energy(ops, d_removal, d_scatter, streaming, source)
    reference = np.linalg.solve(oracle_matrix(ops, d_removal, d_scatter, streaming), source)
    np.testing.assert_allclose(solution.flux, reference, rtol=1e-9, atol=1e-12)


def test_near_singular_diagonal_is_reported():
    energy = np.geomspace(TOP, 1000.0, 11)
    sigma = np.ones((1, 11))
    ops = build_operators(energy, sigma, np.zeros((1, 11)), [0.0])
    solution = solve_energy(ops, [-1.0], [[0.0]], 1.0 + 1e-14, np.ones(11))
    assert np.abs(solution.diagonal).max() < 1e-12


def test_exactly_zero_diagonal_is_reported_not_raised():
    energy = np.geomspace(TOP, 1000.0, 11)
    ops = build_operators(energy, np.ones((1, 11)), np.zeros((1, 11)), [0.0])
    solution = solve_energy(ops, [-1.0], [[0.0]], 1.0, np.ones(11))
    assert np.all(solution.diagonal == 0.0)
    assert np.isinf(solution.flux[0])


@pytest.mark.parametrize(
    ("removal", "scatter", "alpha"),
    [
        (np.ones((1, 5)), np.ones((1, 11)), [0.5]),
        (np.ones((1, 11)), np.ones((1, 5)), [0.5]),
        (np.ones((2, 11)), np.ones((1, 11)), [0.5]),
        (np.ones((2, 11)), np.ones((2, 11)), [0.5]),
    ],
)
def test_build_operators_refuses_mismatched_shapes(removal, scatter, alpha):
    with pytest.raises(ValueError, match="shape"):
        build_operators(np.geomspace(TOP, 9000.0, 11), removal, scatter, alpha)


@pytest.mark.parametrize(
    ("removal", "scatter", "source"),
    [
        ([1.0], [[1.0]], np.ones(3)),
        ([1.0, 1.0], [[1.0]], np.ones(11)),
        ([1.0], [[1.0], [1.0]], np.ones(11)),
    ],
)
def test_solve_energy_refuses_mismatched_shapes(removal, scatter, source):
    energy = np.geomspace(TOP, 9000.0, 11)
    ops = build_operators(energy, np.ones((1, 11)), np.ones((1, 11)), [0.5])
    with pytest.raises(ValueError, match="shape"):
        solve_energy(ops, removal, scatter, 0.0, source)


def test_solve_energy_refuses_an_unknown_anisotropy_method():
    energy, ops = two_term_case()
    with pytest.raises(ValueError, match="anisotropy"):
        solve_energy(ops, [1.0, 1.0], [[1.0], [1.0]], 0.0, np.ones_like(energy), "legendre")


def test_panel_beyond_the_weights_range_is_refused():
    energy = np.array([TOP, 0.1 * TOP])
    sigma = np.ones((1, 2))
    with pytest.raises(ValueError, match="quadrature"):
        build_operators(energy, sigma, sigma, [0.0])


def test_panel_wider_than_a_window_is_refused():
    energy = np.array([TOP, 0.5 * TOP])
    sigma = np.ones((1, 2))
    with pytest.raises(ValueError, match="elastic window"):
        build_operators(energy, sigma, sigma, [kinematic_alpha(236.0)])


def balance_case(panels):
    # A smooth 1/v absorber in a mass 12 scatterer, with its source well above the bottom.
    alpha = kinematic_alpha(11.9)
    energy = np.geomspace(TOP, 1.0, panels + 1)
    scatter = np.full((1, energy.shape[0]), 4.0)
    removal = scatter + 0.1 * np.sqrt(TOP / energy) * 1e-1
    ops = build_operators(energy, removal, scatter, [alpha])
    source = np.exp(-(((np.log(TOP / energy) - 0.5) / 0.1) ** 2))
    flux = solve_energy(ops, [1.0], [[1.0]], 0.0, source).flux
    return energy, ops, source, flux, energy_balance(ops, [1.0], [[1.0]], 0.0, source, flux)


def test_balance_residual_is_truncation_error_at_second_order():
    residuals = [abs(balance_case(n)[-1].residual) for n in (1000, 2000, 4000)]
    assert residuals[0] / residuals[1] == pytest.approx(4.0, rel=0.05)
    assert residuals[1] / residuals[2] == pytest.approx(4.0, rel=0.05)
    assert residuals[2] < 1e-6 * balance_case(4000)[-1].source


def resonance_balance(points):
    # Decision 005 quotes these numbers: a mass 12 scatterer and a strong resonance at mass 236.
    grid = np.concatenate(
        [1000.0 + 0.5 * np.sinh(np.linspace(-6.0, 6.0, points)), np.geomspace(TOP, 10.0, 4001)]
    )
    alpha = kinematic_alpha(236.0)
    energy = limit_panels(
        np.unique(grid[(grid <= TOP) & (grid >= 10.0)])[::-1], 0.9 * -np.log(alpha)
    )
    peak = lorentzian(energy, 1000.0, 0.5, 1.0)
    flat = np.full_like(energy, 10.0)
    ops = build_operators(
        energy,
        np.array([flat, 9000.0 * peak + flat]),
        np.array([flat, 6000.0 * peak + flat]),
        [kinematic_alpha(11.9), alpha],
    )
    source = np.exp(-(((np.log(TOP / energy) - 0.2) / 0.05) ** 2))
    d = np.array([1.0, 0.05])
    flux = solve_energy(ops, d, d[:, None], 0.0, source).flux
    balance = energy_balance(ops, d, d[:, None], 0.0, source, flux)
    captured = panel_rate(0.05 * 3000.0 * peak, flux, ops.rate_weights)
    return balance.residual / balance.source, captured / balance.source


def test_balance_and_capture_converge_through_a_resonance():
    coarse = resonance_balance(21)
    fine = resonance_balance(321)
    reference = resonance_balance(5121)
    assert abs(coarse[0]) == pytest.approx(4.15e-3, rel=0.01)
    assert abs(fine[0]) == pytest.approx(3.27e-5, rel=0.01)
    assert abs(reference[0]) < 1e-6
    assert coarse[1] / reference[1] - 1.0 == pytest.approx(0.07, abs=0.002)
    assert fine[1] / reference[1] - 1.0 == pytest.approx(6e-4, abs=1e-4)


def test_terms_add_as_their_weighted_sum():
    # Two nuclide-temperature terms with different data, and one term holding their weighted sum.
    energy, ops = two_term_case()
    n = energy.shape[0]
    other = np.vstack([ops.removal[0] * (1.0 + 0.2 * np.cos(np.log(energy) * 9.0))])
    other_scatter = ops.scatter[0] * (1.0 + 0.2 * np.cos(np.log(energy) * 9.0))
    split = build_operators(
        energy,
        np.array([ops.removal[0], other[0]]),
        np.array([ops.scatter[0], other_scatter]),
        [ops.alpha[0], ops.alpha[0]],
    )
    single = build_operators(
        energy,
        0.3 * split.removal[:1] + 0.7 * split.removal[1:],
        0.3 * split.scatter[:1] + 0.7 * split.scatter[1:],
        ops.alpha[:1],
    )
    source = np.exp(-np.arange(n) / 20.0)
    a = solve_energy(split, [0.3, 0.7], [[0.3], [0.7]], 0.0, source).flux
    b = solve_energy(single, [1.0], [[1.0]], 0.0, source).flux
    assert (
        np.abs(a - solve_energy(split, [0.7, 0.3], [[0.7], [0.3]], 0.0, source).flux).max() > 1e-4
    )
    np.testing.assert_allclose(a, b, rtol=1e-11)


def test_panel_rate_of_a_constant_is_the_lethargy_span():
    energy = random_grid(2, 50, 1.0, 0.5)
    rate = panel_rate(np.ones_like(energy), np.ones_like(energy), rate_weights(energy))
    assert rate == pytest.approx(np.log(TOP))


@needs_library
def test_build_grid_keeps_required_points_and_limits_panels():
    energy = build_grid(
        [("Na23", "294K")], 1e-2, 1e-8, 2.0e6, 1.0, 0.05, required=[1234.5, 2.8e3, 0.5]
    )
    assert energy[0] == 2.0e6
    assert energy[-1] == 1.0
    assert np.any(energy == 1234.5)
    assert np.log(energy[:-1] / energy[1:]).max() <= 0.05 * (1.0 + 1e-9)
    assert np.all(np.diff(energy) < 0.0)


def test_interpolate_is_zero_below_its_table_and_refuses_above():
    table_e = np.array([3.0, 1.0])
    table_v = np.array([6.0, 2.0])
    np.testing.assert_allclose(interpolate(np.array([3.0, 2.0, 0.5]), table_e, table_v), [6, 4, 0])
    with pytest.raises(ValueError, match="above"):
        interpolate(np.array([4.0, 2.0]), table_e, table_v)


def test_node_pairs_straddle_each_jump():
    pairs = node_pairs([100.0])
    assert pairs.shape == (2,)
    assert pairs[0] == 100.0
    assert 0.0 < 100.0 - pairs[1] < 1e-7


@needs_library
@pytest.mark.parametrize("curves", ["all", "major", "macroscopic"])
def test_thinned_grids_hold_their_curves_at_every_library_point(curves):
    from zerog.data.grid import major_curves
    from zerog.data.neutron import read_reactions
    from zerog.energy.terms import thinned_grids

    nuclides = [("Fe56", "294K"), ("Na23", "294K")]
    densities = np.array([0.02, 0.01])
    rtol, atol = 1e-3, 1e-8
    grid = np.unique(np.concatenate(thinned_grids(nuclides, rtol, atol, 2e7, curves, densities)))
    grid = grid[::-1]
    data = [read_reactions(*key) for key in nuclides]

    def check(energy, values):
        keep = energy <= grid[0]
        energy, values = energy[keep], values[keep]
        approx = interpolate(energy, grid, interpolate(grid, energy, values))
        assert np.all(np.abs(approx - values) <= rtol * np.abs(values) + atol * (1 + 1e-9))

    if curves == "macroscopic":
        energy = np.unique(np.concatenate([d.energy for d in data]))[::-1]
        energy = energy[energy <= grid[0]]
        total = sum(
            n * interpolate(energy, d.energy, major_curves(d)[0])
            for d, n in zip(data, densities, strict=True)
        )
        check(energy, total)
        with pytest.raises(ValueError, match="density"):
            thinned_grids(nuclides, rtol, atol, 2e7, curves)
    else:
        for d in data:
            tested = d.xs if curves == "all" else major_curves(d)
            for row in tested:
                check(d.energy, row)
