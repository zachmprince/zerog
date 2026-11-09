"""Lab Legendre moments of elastic scattering, analytic and against sampled lab cosines, and the
orders above zero in the energy subproblem."""

import os

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre

from zerog.energy.grid import kinematic_alpha, limit_panels
from zerog.energy.inelastic import build_inelastic, coarse_grid, interpolate_hats, restrict
from zerog.energy.moments import (
    _hat_points,
    elastic_order_rows,
    lab_moments,
    mu_lab,
)
from zerog.energy.subproblem import build_operators, solve_energy

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)


@pytest.mark.parametrize("awr", [1.0, 0.999167, 2.0, 15.86, 236.0])
def test_isotropic_lab_moments_match_analytic_forms(awr):
    moments = lab_moments(None, awr, [1e3], 4)[0]
    assert moments[0] == pytest.approx(1.0, abs=1e-13)
    # The mean lab cosine of isotropic centre-of-mass scattering is 2 / 3A.
    assert moments[1] == pytest.approx(2.0 / (3.0 * awr), rel=2e-6)
    for order in (2, 3, 4):
        exact = quad(
            lambda x: 0.5 * eval_legendre(order, mu_lab(x, awr)),  # noqa: B023
            -1.0,
            1.0,
            points=[-1.0 / awr] if awr > 1.0 else None,
            limit=400,
            epsabs=1e-13,
        )[0]
        assert moments[order] == pytest.approx(exact, abs=2e-7)


def sample_lab_cosines(angle_distribution, awr, e, n, rng):
    """OpenMC's AngleDistribution::sample: table i + 1 with probability f, linear in E."""
    energy = np.asarray(angle_distribution.energy)
    i = np.clip(np.searchsorted(energy, e, side="right") - 1, 0, energy.shape[0] - 2)
    f = (e - energy[i]) / (energy[i + 1] - energy[i])
    upper = rng.random(n) < f
    out = np.empty(n)
    for k, pick in ((i, ~upper), (i + 1, upper)):
        draws = angle_distribution.mu[k].sample(int(pick.sum()), seed=int(rng.integers(2**31)))
        out[pick] = draws[0]  # (samples, weights)
    return mu_lab(out, awr)


@needs_library
@pytest.mark.parametrize(
    ("name", "e"), [("O16", 3.2e6), ("Na23", 1.0e6), ("Fe56", 5.5e6), ("H1", 2e6)]
)
def test_lab_moments_match_sampled_lab_cosines(name, e):
    import openmc.data

    from zerog.data.library import nuclide_path
    from zerog.energy.terms import _elastic_angle, awr

    data = openmc.data.IncidentNeutron.from_hdf5(nuclide_path(name))
    distribution = data[2].products[0].distribution[0].angle
    a = awr(name)
    rng = np.random.default_rng(int(e) % 1000)
    cosines = sample_lab_cosines(distribution, a, e, 400_000, rng)
    ours = lab_moments(_elastic_angle(name), a, [e], 3)[0]
    for order in (1, 2, 3):
        values = eval_legendre(order, cosines)
        sd = values.std() / np.sqrt(values.shape[0])
        assert abs(ours[order] - values.mean()) < 5.0 * sd + 1e-6


@needs_library
def test_order_rows_sum_to_the_hat_averaged_moments():
    from zerog.energy.terms import _elastic_angle

    angle = _elastic_angle("O16")
    curve_energy = np.array([2e7, 1e5])
    curve = np.array([3.0, 3.0])
    coarse = coarse_grid(np.array([2e7, 1e5]), 0.00125)
    rows = elastic_order_rows(angle, 15.86, curve_energy, curve, coarse, 3)
    for j in np.searchsorted(-coarse, -np.array([8e6, 3e6, 1.2e6])):
        e, w = _hat_points(coarse, j, curve_energy, curve)
        expected = (w / w.sum()) @ lab_moments(angle, 15.86, e, 3)
        for order, (r, lost) in enumerate(rows, start=1):
            assert r.sums()[j] + lost[j] == pytest.approx(expected[order], rel=2e-4, abs=1e-7)


def order_case(orders=2):
    energy = limit_panels(np.geomspace(1e6, 1e3, 60), 0.01)
    coarse = coarse_grid(energy, 0.00125)
    energy = np.unique(np.concatenate([energy, coarse]))[::-1]
    n = energy.shape[0]
    scatter = np.array([np.full(n, 3.0), 2.0 + 0.5 * np.sin(np.log(energy))])
    removal = scatter + np.array([np.full(n, 0.05), np.full(n, 0.1)])
    awrs = [15.86, 2.0]
    rows = [
        elastic_order_rows(None, a, energy, s, coarse, orders)
        for a, s in zip(awrs, scatter, strict=True)
    ]
    kernels = build_inelastic([None, None], energy, coarse)
    alpha = [kinematic_alpha(a) for a in awrs]
    ops = build_operators(energy, removal, scatter, alpha, inelastic=kernels, order_rows=rows)
    source = np.exp(-(((np.log(energy / energy[0]) + 0.3) / 0.1) ** 2))
    return ops, source


def test_orders_with_zero_density_reproduce_the_order_zero_solve():
    ops, source = order_case()
    d = np.array([1.0, 0.7])
    zero_order = solve_energy(ops, d, d[:, None], 0.5, source)
    padded = solve_energy(ops, d, np.column_stack([d, 0.0 * d, 0.0 * d]), 0.5, source)
    np.testing.assert_allclose(padded.flux, zero_order.flux, rtol=1e-13, atol=0.0)


def test_order_emission_is_its_rows_applied_to_the_flux():
    """At the solution, the order-l source is the rows times what the elastic restricts to each
    hat, prolonged to the nodes: solving order 0 with that source gives the same flux."""
    ops, source = order_case()
    d = np.array([1.0, 0.7])
    d1 = np.array([0.4, -0.3])
    d2 = np.array([0.2, 0.1])
    solution = solve_energy(ops, d, np.column_stack([d, d1, d2]), 0.5, source)
    ext = ops.extended
    restricted = restrict(ext.inelastic, ops.energy, solution.flux, ops.rate_weights)
    kernels = ext.inelastic
    values = np.zeros(kernels.coarse.shape[0])
    densities = np.concatenate([0.0 * d, d1, d2])
    for t in range(2, densities.shape[0]):
        for j in range(kernels.coarse.shape[0]):
            a, b = kernels.first[t, j], kernels.last[t, j]
            row = kernels.values[kernels.starts[t, j] : kernels.starts[t, j] + b - a]
            values[a:b] += densities[t] * restricted[t, j] * row
    extra = interpolate_hats(ops, values)
    plain = solve_energy(ops, d, d[:, None], 0.5, source + extra)
    np.testing.assert_allclose(plain.flux, solution.flux, rtol=1e-9, atol=1e-14)


def test_balance_with_orders_closes_on_the_projected_equation():
    from zerog.energy.balance import energy_balance

    ops, source = order_case()
    d = np.array([1.0, 0.7])
    scatter = np.column_stack([d, [0.4, -0.3], [0.2, 0.1]])
    solution = solve_energy(ops, d, scatter, 0.5, source)
    balance = energy_balance(ops, d, scatter, 0.5, source, solution.flux)
    assert abs(balance.residual / balance.removed) < 1e-6


def test_isotropic_order_rows_have_the_analytic_shape():
    """For isotropic centre-of-mass scattering at e, E' is uniform on [alpha e, e] and mu_lab a
    known function of E': each hat's value is its integral of P_l(mu_lab) over that density.

    deposit spreads each outgoing piece uniformly in lethargy, as inelastic transfer does, which
    moves a piece's split between two hats at first order in its width: about 2e-3 of the row's
    peak value."""
    awr = 15.86
    energy = np.array([2e6, 1e5])
    coarse = coarse_grid(energy, 0.00125)
    rows = elastic_order_rows(None, awr, energy, np.array([3.0, 3.0]), coarse, 2)
    alpha = kinematic_alpha(awr)
    grid_u = np.log(coarse[0] / coarse)
    j = np.searchsorted(-coarse, -1e6)
    e_points, w = _hat_points(coarse, j, energy, np.array([3.0, 3.0]))
    w = w / w.sum()
    for m in range(j - 3, j + 40):
        for order, (r, _) in enumerate(rows, start=1):

            def integrand(x, m=m, order=order, e=None):
                hat = np.interp(np.log(coarse[0] / x), grid_u[m - 1 : m + 2], [0.0, 1.0, 0.0])
                mu = 1.0 - 2.0 * (1.0 - x / e) / (1.0 - alpha)
                return hat * eval_legendre(order, mu_lab(mu, awr)) / ((1.0 - alpha) * e)

            expected = sum(
                w_p
                * quad(
                    lambda x, e=e_p, m=m, order=order: integrand(x, m, order, e),
                    max(alpha * e_p, coarse[m + 1]),
                    min(e_p, coarse[m - 1]),
                    limit=200,
                )[0]
                for e_p, w_p in zip(e_points, w, strict=True)
                if min(e_p, coarse[m - 1]) > max(alpha * e_p, coarse[m + 1])
            )
            a, b = r.first[j], r.last[j]
            ours = r.values[r.starts[j] + m - a] if a <= m < b else 0.0
            # Edge hats take a partial piece: bound the error by a fraction of the row peak.
            assert ours == pytest.approx(expected, abs=5e-3 * 2.8e-3)
