"""Anisotropic elastic scattering: analytic asymptotes, a quadrature oracle and openmc.data."""

import os

import numpy as np
import pytest
from scipy.integrate import quad

from zerog.data.scatter import AngleTables, read_scattering_laws
from zerog.energy.balance import energy_balance
from zerog.energy.elastic import (
    angle_pdf,
    compile_angle_tables,
    cut_moments,
    lab_cosine,
    moment_tree_add,
    moment_tree_size,
    point_emission,
    refine_for_angle,
    table_moments,
)
from zerog.energy.grid import (
    centred_weights,
    kinematic_alpha,
    node_pairs,
    panel_rate,
    rate_weights,
)
from zerog.energy.subproblem import build_operators, solve_energy

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)

TOP = 1.0e6


def tables_from(mu, pdf, energies=(2.0e7, 1.0e-5)):
    """The same lin-lin table at each energy, normalized."""
    mu = np.asarray(mu, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    pdf = pdf / np.trapezoid(pdf, mu)
    n = len(energies)
    return AngleTables(
        energy=np.array(energies, dtype=float),
        offsets=np.arange(n + 1) * mu.shape[0],
        interpolation=np.full(n, 2, dtype=np.int64),
        mu=np.tile(mu, n),
        pdf=np.tile(pdf, n),
        cdf=np.zeros(n * mu.shape[0]),
    )


LINEAR = ([-1.0, 1.0], [0.5, 1.5])
BACKWARD = ([-1.0, 1.0], [1.5, 0.5])
PEAKED = ([-1.0, 0.8, 0.98, 1.0], [0.05, 0.1, 2.0, 40.0])


def mean_lethargy_gain(alpha, table):
    mu = table.mu[: table.offsets[1]]
    pdf = table.pdf[: table.offsets[1]]

    def integrand(x):
        return -np.log(0.5 * (1.0 + alpha + (1.0 - alpha) * x)) * np.interp(x, mu, pdf)

    return quad(integrand, -1.0, 1.0, points=mu[1:-1].tolist() or None, limit=200)[0]


def asymptote_case(awr, table, resolution, windows=30):
    alpha = kinematic_alpha(awr)
    width = -np.log(alpha)
    energy = TOP * np.exp(-np.linspace(0.0, windows * width, windows * 20 + 1))
    energy = np.unique(np.concatenate([energy, node_pairs([alpha * TOP])]))[::-1]
    energy = refine_for_angle(energy, [alpha], [table], resolution)
    sigma = np.ones((1, energy.shape[0]))
    ops = build_operators(energy, sigma, sigma, [alpha], [table])
    source = point_emission(energy, TOP, alpha, table)
    return alpha, solve_energy(ops, [1.0], [[1.0]], 0.0, source).flux


@pytest.mark.parametrize("shape", [LINEAR, BACKWARD, PEAKED], ids=["linear", "backward", "peaked"])
@pytest.mark.parametrize("awr", [11.9, 55.4])
def test_pure_scatterer_asymptote_is_one_over_mean_lethargy_gain(awr, shape):
    table = tables_from(*shape)
    # At the default resolution a peak of about 54 leaves 1.2e-3; the next test shows it converging.
    alpha, flux = asymptote_case(awr, table, 0.05)
    assert flux[-1] * mean_lethargy_gain(alpha, table) == pytest.approx(1.0, abs=2e-3)


def test_asymptote_error_falls_with_peak_resolution():
    # Without removing the kernel's discrete mean this stalls near 1%, as a net gain per collision.
    table = tables_from(*PEAKED)
    errors = []
    for resolution in (0.2, 0.05, 0.0125):
        alpha, flux = asymptote_case(11.9, table, resolution)
        errors.append(abs(flux[-1] * mean_lethargy_gain(alpha, table) - 1.0))
    assert errors[1] < errors[0] / 10.0
    assert errors[2] < 1e-4


def varying_case(panels):
    # Tables swing from forward to backward within a window, then go flat, as resonant data does.
    alpha = kinematic_alpha(22.8)
    energies = TOP * np.exp(-np.linspace(0.0, 1.0, 41))
    mu = np.array([-1.0, 1.0])
    slopes = 0.4 * np.cos(np.arange(41) * 1.3) * (np.arange(41) < 40)
    table = AngleTables(
        energy=np.append(energies, 1.0e-5),
        offsets=np.arange(43) * 2,
        interpolation=np.full(42, 2, dtype=np.int64),
        mu=np.tile(mu, 42),
        pdf=np.concatenate([[0.5 - b, 0.5 + b] for b in np.append(slopes, 0.0)]),
        cdf=np.zeros(84),
    )
    energy = np.geomspace(TOP, 1.0, panels + 1)
    sigma = np.ones((1, panels + 1))
    ops = build_operators(energy, sigma + 0.01, sigma, [alpha], [table])
    source = np.exp(-(((np.log(TOP / energy) - 0.2) / 0.03) ** 2))
    flux = solve_energy(ops, [1.0], [[1.0]], 0.0, source).flux
    return energy_balance(ops, [1.0], [[1.0]], 0.0, source, flux)


def test_tables_varying_with_incoming_energy_conserve_neutrons():
    # 2 f - 1 integrates to zero over outgoing energy only, not over incoming energy at a fixed
    # outgoing one. Treating the second as quadrature error created neutrons here.
    closures = [abs(b.residual) / b.source for b in map(varying_case, (2000, 8000))]
    assert closures[1] < 1e-4
    assert closures[1] < closures[0]


@pytest.mark.parametrize("awr", [15.9, 22.8, 55.4])
def test_slowing_down_density_survives_the_anisotropy_cutoff(awr):
    # Anisotropy falling linearly in E, with no absorption. Well below the cutoff f is flat, so
    # F xi is exactly 1 there whatever happened above it.
    alpha = kinematic_alpha(awr)
    table = AngleTables(
        energy=np.array([1.0e5, 1.0e-5]),
        offsets=np.array([0, 2, 4]),
        interpolation=np.array([2, 2]),
        mu=np.array([-1.0, 1.0, -1.0, 1.0]),
        pdf=np.array([0.3, 0.7, 0.5, 0.5]),
        cdf=np.zeros(4),
    )
    energy = np.unique(
        np.concatenate([np.geomspace(1.0e5, 1.0, 8001), node_pairs([alpha * 1.0e5])])
    )
    energy = energy[::-1]
    sigma = np.ones((1, energy.shape[0]))
    ops = build_operators(energy, sigma, sigma, [alpha], [table])
    source = point_emission(energy, 1.0e5, alpha, table)
    flux = solve_energy(ops, [1.0], [[1.0]], 0.0, source).flux
    xi = 1.0 + alpha * np.log(alpha) / (1.0 - alpha)
    assert flux[-1] * xi == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("shape", [LINEAR, PEAKED], ids=["linear", "peaked"])
def test_point_emission_has_unit_strength(shape):
    table = tables_from(*shape)
    alpha = kinematic_alpha(55.4)
    energy = np.unique(
        np.concatenate([np.geomspace(TOP, 0.5 * TOP, 40), node_pairs([alpha * TOP])])
    )
    energy = refine_for_angle(energy[::-1], [alpha], [table])
    source = point_emission(energy, TOP, alpha, table)
    assert panel_rate(np.ones_like(energy), source, rate_weights(energy)) == pytest.approx(1.0)


def test_refine_for_angle_limits_panels_to_the_local_peak():
    table = tables_from(*PEAKED)
    alpha = kinematic_alpha(11.9)
    energy = refine_for_angle(np.array([TOP, 0.9 * TOP]), [alpha], [table], 0.05)
    limit = 0.05 * -np.log(alpha) / table.pdf.max()
    assert np.log(energy[:-1] / energy[1:]).max() <= limit * (1.0 + 1e-9)
    assert energy[0] == TOP
    assert energy[-1] == 0.9 * TOP


def test_balance_closes_with_the_bottom_inside_an_anisotropic_window():
    table = tables_from(*LINEAR)
    closures = []
    for panels in (500, 2000):
        energy = np.geomspace(TOP, 0.5 * TOP, panels + 1)
        sigma = np.ones((1, panels + 1))
        ops = build_operators(energy, sigma + 0.01, sigma, [kinematic_alpha(2.0)], [table])
        source = np.exp(-(((np.log(TOP / energy) - 0.1) / 0.02) ** 2))
        flux = solve_energy(ops, [1.0], [[1.0]], 0.0, source).flux
        balance = energy_balance(ops, [1.0], [[1.0]], 0.0, source, flux)
        closures.append(abs(balance.residual) / balance.source)
    assert closures[1] < closures[0] / 3.0
    assert closures[1] < 1e-4


def test_flat_table_reproduces_the_isotropic_recursion():
    alpha = kinematic_alpha(23.0)
    energy = np.geomspace(TOP, 1.0e3, 800)
    wiggle = 1.0 + 0.5 * np.sin(np.log(energy) * 30.0)
    removal = np.array([wiggle + 0.1])
    scatter = np.array([wiggle])
    source = np.exp(-np.arange(800) / 10.0)
    isotropic = solve_energy(
        build_operators(energy, removal, scatter, [alpha]), [1.0], [[1.0]], 0.0, source
    ).flux
    flat = tables_from([-1.0, 1.0], [1.0, 1.0])
    ops = build_operators(energy, removal, scatter, [alpha], [flat])
    assert np.isinf(ops.angle.cutoff[0])
    np.testing.assert_allclose(
        solve_energy(ops, [1.0], [[1.0]], 0.0, source).flux, isotropic, rtol=1e-13
    )


def quad_row(energy, sigma, alpha, table, k):
    """Coefficients of F in Y(energy[k]) by adaptive quadrature; independent of the march."""
    kernels = compile_angle_tables([table], tolerance=-1.0)
    row = np.zeros(energy.shape[0])
    target = energy[k]
    edge = target / alpha

    def density(e):
        x = (2.0 * target / e - 1.0 - alpha) / (1.0 - alpha)
        return 2.0 * angle_pdf(
            kernels.tables,
            kernels.table_energy,
            kernels.points,
            kernels.mu,
            kernels.pdf,
            0,
            e,
            x,
        )

    for i in range(k):
        upper = energy[i]
        lower = energy[i + 1]
        if lower >= edge:
            continue
        h = np.log(upper / lower)
        top = min(upper, edge)

        def basis(e, node, i=i, upper=upper, lower=lower, h=h):
            g = (upper - e) / (upper - lower)
            r = np.log(upper / e) / h
            sigma_e = (1.0 - g) * sigma[i] + g * sigma[i + 1]
            hat = r if node else 1.0 - r
            return sigma_e * hat * density(e) / e**2

        # Kinks fall at every table's knots, mapped to E' here, and at the table energies.
        knots = target * 2.0 / (1.0 + alpha + (1.0 - alpha) * np.unique(table.mu))
        knots = np.concatenate([knots, table.energy])
        inside = sorted({v for v in knots if lower < v < top})
        for node in (0, 1):
            row[i + node] += (
                target
                / (1.0 - alpha)
                * quad(basis, lower, top, args=(node,), points=inside or None, limit=200)[0]
            )
    return row


@pytest.mark.parametrize(
    ("anisotropy", "rtol"), [("separable", 1e-10), ("tree", 1e-10), ("direct", 2e-4)]
)
def test_march_matches_quadrature_oracle_with_negative_density(anisotropy, rtol):
    # The separable form is exact for the interpolants; the direct sum is quadrature.
    alpha = kinematic_alpha(11.9)
    energy = TOP * np.exp(-np.linspace(0.0, 1.2, 61))
    sigma = (1.0 + 0.4 * np.sin(np.arange(61) * 0.7))[None, :]
    table = tables_from(*PEAKED)
    ops = build_operators(energy, sigma + 0.2, sigma, [alpha], [table])
    d_removal = 1.0
    d_scatter = -0.6
    streaming = 0.5
    source = np.exp(-np.arange(61) / 8.0)
    flux = solve_energy(ops, [d_removal], [[d_scatter]], streaming, source, anisotropy).flux
    a = np.diag(streaming + d_removal * ops.removal[0])
    for k in range(1, 61):
        a[k] -= d_scatter * quad_row(energy, sigma[0], alpha, table, k)
    np.testing.assert_allclose(flux, np.linalg.solve(a, source), rtol=rtol)


@pytest.mark.parametrize("anisotropy", ["separable", "tree"])
@pytest.mark.parametrize("awr", [11.9, 236.0])
def test_separable_form_matches_quadrature_with_tables_varying_inside_a_window(awr, anisotropy):
    # Tables on different mu knots, a jump between two at one energy, and the top table held above
    # the data, over three windows. A flat table far below keeps the anisotropy cutoff off the
    # grid, since quad_row has none.
    alpha = kinematic_alpha(awr)
    width = -np.log(alpha)
    top = TOP * np.exp(-0.3 * width)
    shapes = [
        ([-1.0, 0.8, 0.98, 1.0], [0.05, 0.1, 2.0, 40.0]),
        ([-1.0, 0.0, 1.0], [0.2, 0.4, 1.5]),
        ([-1.0, -0.5, 0.5, 0.9, 1.0], [0.8, 0.3, 0.4, 1.0, 3.0]),
        ([-1.0, 1.0], [0.7, 0.3]),
        ([-1.0, 0.3, 1.0], [0.6, 0.4, 0.6]),
        ([-1.0, 1.0], [0.5, 0.5]),
    ]
    energies = np.append(top * np.exp(-width * np.array([0.0, 0.7, 1.1, 1.1, 2.4])), 1.0e-5)
    mus, pdfs = [], []
    for mu, pdf in shapes:
        mus.append(np.array(mu))
        pdfs.append(np.array(pdf) / np.trapezoid(pdf, mu))
    table = AngleTables(
        energy=energies,
        offsets=np.concatenate([[0], np.cumsum([m.shape[0] for m in mus])]),
        interpolation=np.full(6, 2, dtype=np.int64),
        mu=np.concatenate(mus),
        pdf=np.concatenate(pdfs),
        cdf=np.zeros(sum(m.shape[0] for m in mus)),
    )
    energy = TOP * np.exp(-np.linspace(0.0, 3.0 * width, 61))
    sigma = (1.0 + 0.4 * np.sin(np.arange(61) * 0.7))[None, :]
    ops = build_operators(energy, sigma + 0.2, sigma, [alpha], [table])
    assert ops.angle.cutoff[0] < energy[-1]
    source = np.exp(-np.arange(61) / 8.0)
    flux = solve_energy(ops, [1.0], [[0.8]], 0.1, source, anisotropy).flux
    a = np.diag(0.1 + ops.removal[0])
    for k in range(1, 61):
        a[k] -= 0.8 * quad_row(energy, sigma[0], alpha, table, k)
    # A peak of 40 rising over 0.02 of mu cost the earlier running moments 2e-8 at awr 236 in
    # round-off. Centred moments have no such cancellation: 1.5e-11 at awr 11.9, 1e-12 at 236.
    np.testing.assert_allclose(flux, np.linalg.solve(a, source), rtol=1e-9)


def test_moment_tree_nodes_hold_centred_moments_of_their_panels():
    # Each complete node holds the integrals of sigma F (E_r / E - 1)^n du over its panels, with
    # E_r its top energy, built from its children's by shifting their centres.
    rng = np.random.default_rng(7)
    energy = TOP * np.exp(-np.cumsum(np.concatenate([[0.0], rng.uniform(1e-4, 0.05, 36)])))
    n = energy.shape[0]
    sigma = rng.uniform(0.5, 2.0, n)
    flux = rng.uniform(-1.0, 3.0, n)
    size = moment_tree_size(n)
    tree = np.zeros((2 * size, 4))
    for p in range(n - 1):
        moment_tree_add(tree, size, energy, sigma, flux, centred_weights(energy), p)
    levels = int(np.log2(size))

    def integrand(s, p, e_r, power):
        e = energy[p] * np.exp(-s)
        g = (energy[p] - e) / (energy[p] - energy[p + 1])
        r = s / np.log(energy[p] / energy[p + 1])
        value = ((1 - g) * sigma[p] + g * sigma[p + 1]) * ((1 - r) * flux[p] + r * flux[p + 1])
        return value * (e_r / e - 1.0) ** power

    checked = 0
    for v in range(1, 2 * size):
        height = levels - int(np.log2(v))
        first = (v << height) - size
        if first + (1 << height) > n - 1:
            continue  # not complete: the grid ends inside it
        e_r = energy[first]
        for power in range(3):
            expected = sum(
                quad(
                    integrand,
                    0.0,
                    np.log(energy[p] / energy[p + 1]),
                    args=(p, e_r, power),
                    epsabs=0.0,
                    epsrel=1e-12,
                )[0]
                for p in range(first, first + (1 << height))
            )
            assert tree[v, power] == pytest.approx(expected, rel=1e-10, abs=1e-14 * abs(tree[v, 0]))
        checked += 1
    assert checked > 50


@pytest.mark.parametrize("c", [0.0, 1e-9, 3e-4, 0.7, 1.0])
def test_cut_moments_are_centred_moments_of_the_panel_top(c):
    # Over a panel's top c of its width: the integral of sigma F (E_i / E - 1)^n du, n = 0, 1, 2.
    upper = 2.0e6
    h = 0.12
    lower = upper * np.exp(-h)
    sigma0, sigma1, flux0, flux1 = 1.3, 0.4, -0.7, 2.1

    def integrand(s, power):
        e = upper * np.exp(-s)
        g = (upper - e) / (upper - lower)
        value = ((1 - g) * sigma0 + g * sigma1) * (flux0 + (flux1 - flux0) * s / h)
        return value * np.expm1(s) ** power

    moments = cut_moments(upper, lower, h, sigma0, sigma1, flux0, flux1, c * h)
    # The kernel's coefficients times the range of (E_i / E - 1)^n are of one size, so each
    # moment's error counts against the scale of the zeroth's magnitude.
    scale = quad(lambda s: abs(integrand(s, 0)), 0.0, max(c * h, 1e-300))[0]
    for power in range(3):
        expected = quad(integrand, 0.0, c * h, args=(power,), epsabs=0.0, epsrel=1e-13)[0]
        assert moments[power] == pytest.approx(expected, rel=1e-12, abs=1e-13 * scale)
    if c == 1.0:
        # The whole panel is the tree's leaf.
        energy = np.array([upper, lower])
        leaf = centred_weights(energy)[:, 0] @ np.array(
            [sigma0 * flux0, sigma0 * flux1, sigma1 * flux0, sigma1 * flux1]
        )
        np.testing.assert_allclose(moments, leaf, rtol=1e-12, atol=1e-13 * scale)


@needs_library
def test_shared_cuts_match_the_tree_path_with_real_data():
    # Forward peaks above 1 MeV, tables close in energy (Fe56) and a window to the grid's top (H1):
    # sharing each knot's cut panel changes only round-off.
    from zerog.energy.elastic import point_emission
    from zerog.energy.terms import elastic_grid, load_term

    nuclides = [("U238", "294K"), ("Fe56", "294K"), ("H1", "294K")]
    top = 2.0e7
    energy = elastic_grid(nuclides, 1e-3, 1e-8, top, 2.0e5)
    terms = [load_term(name, label, energy) for name, label in nuclides]
    ops = build_operators(
        energy,
        np.array([t.total for t in terms]),
        np.array([t.elastic for t in terms]),
        [t.alpha for t in terms],
        [t.angle for t in terms],
    )
    d = np.array([0.02, 0.03, 0.01])
    source = sum(
        density * point_emission(energy, top, t.alpha, t.angle)
        for density, t in zip(d, terms, strict=True)
    )
    flux = {
        method: solve_energy(ops, d, d[:, None], 0.0, source, method).flux
        for method in ("separable", "tree")
    }
    np.testing.assert_allclose(flux["separable"], flux["tree"], rtol=1e-12)


@needs_library
def test_separable_form_matches_the_direct_sum_with_real_data():
    from zerog.energy.elastic import point_emission
    from zerog.energy.terms import elastic_grid, load_term

    nuclides = [("U238", "294K"), ("O16", "294K")]
    top = 4.0e4
    energy = elastic_grid(nuclides, 1e-3, 1e-8, top, 1.5e4)
    terms = [load_term(name, label, energy) for name, label in nuclides]
    ops = build_operators(
        energy,
        np.array([t.total for t in terms]),
        np.array([t.elastic for t in terms]),
        [t.alpha for t in terms],
        [t.angle for t in terms],
    )
    d = np.array([0.02, 0.04])
    source = sum(
        density * point_emission(energy, top, t.alpha, t.angle)
        for density, t in zip(d, terms, strict=True)
    )
    flux = {
        method: solve_energy(ops, d, d[:, None], 0.0, source, method).flux
        for method in ("separable", "direct")
    }
    # The direct sum's quadrature error, about 1e-6 on this grid, sets the tolerance.
    np.testing.assert_allclose(flux["separable"], flux["direct"], rtol=2e-5)
    capture = {m: panel_rate(terms[0].capture, f, ops.rate_weights) for m, f in flux.items()}
    assert capture["separable"] == pytest.approx(capture["direct"], rel=1e-6)


def test_cutoff_is_where_the_mix_between_sparse_tables_reaches_the_tolerance():
    # One steep table at 1000 eV and one flat at nearly zero: the mix deviates in proportion to E.
    table = AngleTables(
        energy=np.array([1000.0, 1.0e-5]),
        offsets=np.array([0, 2, 4]),
        interpolation=np.array([2, 2]),
        mu=np.array([-1.0, 1.0, -1.0, 1.0]),
        pdf=np.array([0.4, 0.6, 0.5, 0.5]),
        cdf=np.zeros(4),
    )
    assert compile_angle_tables([table], 1e-3).cutoff[0] == pytest.approx(10.0, rel=1e-6)


def test_table_moments_are_exact_for_a_linear_pdf():
    table = tables_from(*LINEAR)
    # f = (1 + mu / 2) / 2, so its P1 moment is 1/6 and all higher ones vanish.
    np.testing.assert_allclose(table_moments(table, 0, 4), [1.0, 1.0 / 6.0, 0.0, 0.0], atol=1e-14)


def test_lab_cosine_matches_two_body_kinematics():
    awr = 11.9
    for mu_cm in (-0.9, -0.2, 0.3, 0.95):
        # Speeds in the centre of mass, for a unit lab speed with a target at rest.
        cm_speed = 1.0 / (awr + 1.0)
        neutron = awr / (awr + 1.0)
        along = neutron * mu_cm + cm_speed
        across = neutron * np.sqrt(1.0 - mu_cm**2)
        ratio = along**2 + across**2
        assert lab_cosine(ratio, awr) == pytest.approx(along / np.sqrt(ratio), rel=1e-13)


@needs_library
@pytest.mark.parametrize("name", ["Fe56", "U238", "O16", "H1"])
def test_tables_and_moments_match_openmc_data(name):
    # This checks reading and the mix formula. The mix is from OpenMC's source, tested by study 02.
    import openmc.data

    from zerog.data.library import nuclide_path

    angle = next(r for r in read_scattering_laws(name) if r.mt == 2).laws[0].angle
    kernels = compile_angle_tables([angle], tolerance=-1.0)
    data = openmc.data.IncidentNeutron.from_hdf5(nuclide_path(name))
    reference = data.reactions[2].products[0].distribution[0].angle
    tables = reference.mu
    energies = reference.energy
    rng = np.random.default_rng(1)
    for i in rng.integers(0, len(energies) - 1, 40):
        if energies[i + 1] <= energies[i]:
            continue
        r = rng.uniform()
        e = energies[i] + r * (energies[i + 1] - energies[i])
        for x in rng.uniform(-1.0, 1.0, 5):
            expected = (1.0 - r) * np.interp(x, tables[i].x, tables[i].p) + r * np.interp(
                x, tables[i + 1].x, tables[i + 1].p
            )
            ours = angle_pdf(
                kernels.tables,
                kernels.table_energy,
                kernels.points,
                kernels.mu,
                kernels.pdf,
                0,
                e,
                x,
            )
            assert ours == pytest.approx(expected, rel=1e-12, abs=1e-13)
    # Moments, against numerical integration of openmc's own table.
    for i in rng.integers(0, len(energies), 10):
        k = len(energies) - 1 - i
        moments = table_moments(angle, k, 4)
        table = tables[i]
        for order in range(4):
            basis = np.polynomial.legendre.Legendre.basis(order)
            exact = sum(
                quad(
                    lambda v: basis(v) * np.interp(v, table.x, table.p),  # noqa: B023
                    table.x[j],
                    table.x[j + 1],
                )[0]
                for j in range(len(table.x) - 1)
            )
            assert moments[order] == pytest.approx(exact, abs=1e-9)


@needs_library
@pytest.mark.parametrize("name", ["U238", "O16", "Fe56"])
def test_thinned_angle_tables_stay_within_tolerance_and_normalized(name):
    from zerog.energy.elastic import thin_angle
    from zerog.energy.terms import _elastic_angle

    angle = _elastic_angle(name)
    tolerance = 1e-3  # absolute, in pdf units
    thin = thin_angle(angle, tolerance)
    assert thin.mu.shape[0] < angle.mu.shape[0]
    for i in range(angle.energy.shape[0]):
        mu = angle.mu[angle.offsets[i] : angle.offsets[i + 1]]
        pdf = angle.pdf[angle.offsets[i] : angle.offsets[i + 1]]
        t_mu = thin.mu[thin.offsets[i] : thin.offsets[i + 1]]
        t_pdf = thin.pdf[thin.offsets[i] : thin.offsets[i + 1]]
        assert np.trapezoid(t_pdf, t_mu) == pytest.approx(1.0, abs=1e-12)
        # Within the tolerance, plus what renormalizing moved: at most twice it, times the pdf.
        back = np.interp(mu, t_mu, t_pdf)
        assert np.abs(back - pdf).max() <= tolerance + 2.5 * tolerance * pdf.max() + 1e-9


@pytest.mark.parametrize("awr", [11.9, 55.4, 236.0])
def test_legendre_form_is_exact_where_the_tables_are_polynomials(awr):
    # Linear pdfs are their own order-1 expansions, and the mix between tables is linear in E' in
    # both forms, so the Legendre windows must reproduce the exact ones to round-off: across
    # several blocks, with a table energy inside a panel and the window reaching the grid's top.
    alpha = kinematic_alpha(awr)
    width = -np.log(alpha)
    middle = TOP * np.exp(-1.5 * width)
    shapes = [
        ([-1.0, 1.0], [0.6, 1.4]),
        ([-1.0, 0.0, 1.0], [0.8, 1.0, 1.2]),
        ([-1.0, 1.0], [1.1, 0.9]),
    ]
    mus, pdfs = [], []
    for mu, pdf in shapes:
        mus.append(np.array(mu))
        pdfs.append(np.array(pdf) / np.trapezoid(pdf, mu))
    table = AngleTables(
        energy=np.array([2.0e7, middle, 1.0e-5]),
        offsets=np.concatenate([[0], np.cumsum([m.shape[0] for m in mus])]),
        interpolation=np.full(3, 2, dtype=np.int64),
        mu=np.concatenate(mus),
        pdf=np.concatenate(pdfs),
        cdf=np.zeros(sum(m.shape[0] for m in mus)),
    )
    n = 600
    energy = TOP * np.exp(-np.linspace(0.0, 6.0 * width, n))
    assert not np.any(np.isclose(energy, middle, rtol=1e-6))
    sigma = (1.0 + 0.4 * np.sin(np.arange(n) * 0.05))[None, :]
    source = np.exp(-np.arange(n) / 50.0)
    exact = build_operators(energy, sigma + 0.2, sigma, [alpha], [table])
    legendre = build_operators(energy, sigma + 0.2, sigma, [alpha], [table], legendre=(2, 1e-9))
    assert legendre.angle.legendre_switch[0] == np.inf
    f_exact = solve_energy(exact, [1.0], [[0.8]], 0.1, source).flux
    f_legendre = solve_energy(legendre, [1.0], [[0.8]], 0.1, source).flux
    np.testing.assert_allclose(f_legendre, f_exact, rtol=1e-13)


def test_legendre_coefficients_conserve_and_flag_tables_by_truncation_error():
    from zerog.energy.elastic import legendre_coefficients, legendre_switch

    # A linear table is exact at order 1; a forward peak is not, at any low order.
    table = AngleTables(
        energy=np.array([2.0e7, 1.0e6, 1.0e3]),
        offsets=np.array([0, 4, 6, 8]),
        interpolation=np.full(3, 2, dtype=np.int64),
        mu=np.array([-1.0, 0.8, 0.98, 1.0, -1.0, 1.0, -1.0, 1.0]),
        pdf=np.concatenate(
            [
                np.array([0.05, 0.1, 2.0, 40.0])
                / np.trapezoid([0.05, 0.1, 2.0, 40.0], [-1, 0.8, 0.98, 1]),
                [0.4, 0.6],
                [0.5, 0.5],
            ]
        ),
        cdf=np.zeros(8),
    )
    b, ok = legendre_coefficients(table, 4, 1e-6)
    np.testing.assert_array_equal(ok, [False, True, True])
    assert legendre_switch(table.energy, ok) == 1.0e6
    # 2 f - 1 = 0.2 mu for the linear table, and zero for the flat one.
    np.testing.assert_allclose(b[1], [0.0, 0.2, 0, 0, 0], atol=1e-14)
    np.testing.assert_allclose(b[2], 0.0, atol=1e-14)
    # Every truncation integrates to zero over mu, so it moves no neutrons.
    integrals = np.array([(1.0 - (-1.0) ** (n + 1)) / (n + 1) for n in range(5)])
    np.testing.assert_allclose(b @ integrals, 0.0, atol=1e-12)


def test_light_targets_keep_the_exact_form():
    # A window as wide as the grid would put rho near 1 / alpha into the running moments' powers,
    # so targets lighter than LEGENDRE_MIN_ALPHA (hydrogen's alpha is 2e-7, not 0) march exactly.
    from zerog.energy.elastic import LEGENDRE_MIN_ALPHA

    table = tables_from(*LINEAR)
    energy = TOP * np.exp(-np.linspace(0.0, 4.0, 200))
    sigma = np.ones((2, 200))
    alphas = [1.7e-7, kinematic_alpha(11.9)]
    assert alphas[0] < LEGENDRE_MIN_ALPHA < alphas[1]
    source = np.exp(-np.arange(200) / 20.0)
    d = np.array([0.5, 0.5])
    flux = {
        legendre: solve_energy(
            build_operators(energy, 1.2 * sigma, sigma, alphas, [table, table], legendre=legendre),
            d,
            d[:, None],
            0.1,
            source,
        ).flux
        for legendre in (None, (2, 1.0))
    }
    assert np.all(np.isfinite(flux[(2, 1.0)]))
    np.testing.assert_allclose(flux[(2, 1.0)], flux[None], rtol=1e-13)
