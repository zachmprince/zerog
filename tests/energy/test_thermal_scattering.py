"""Thermal scattering checks: the free gas kernel against analytic forms and OpenMC's sampler, the
S(alpha, beta) rows against OpenMC's sampling of openmc.data, and the block solve against a
Maxwellian equilibrium."""

import itertools
import os

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2

import zerog.energy.subproblem as subproblem
import zerog.energy.thermal as thermal
from zerog.energy.balance import energy_balance
from zerog.energy.grid import kinematic_alpha
from zerog.energy.inelastic import InelasticKernels, bin_segments, build_inelastic, coarse_grid
from zerog.energy.subproblem import build_operators, solve_energy, upscatter_block
from zerog.energy.thermal import (
    ThermalTerm,
    build_thermal,
    free_gas_kernel,
    free_gas_norm,
    free_gas_range,
    free_gas_row,
    sab_choice,
    sab_segments,
    sab_table,
    thermal_coarse,
)

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)

KT = 0.0517  # eV, about 600 K
CASES = [(0.999167, 0.1), (15.86, 1.0), (236.0, 20.0), (236.0, 0.03), (0.999167, 20.0)]


def cxs_sample(awr, e, kT, n, rng):
    """Outgoing energies from OpenMC's sample_cxs_target_velocity and elastic_scatter, with an
    isotropic mu_cm, in OpenMC's units where E = v^2."""
    beta_vn = np.sqrt(awr * e / kT)
    alpha = 1.0 / (1.0 + np.sqrt(np.pi) * beta_vn / 2.0)
    speeds, cosines = [], []
    need = n
    while need > 0:
        m = 2 * need
        r1, r2 = rng.random(m), rng.random(m)
        c = np.cos(np.pi / 2.0 * rng.random(m))
        bt2 = np.where(rng.random(m) < alpha, -np.log(r1 * r2), -np.log(r1) - np.log(r2) * c * c)
        bt = np.sqrt(bt2)
        mu = rng.uniform(-1.0, 1.0, m)
        accept = np.sqrt(beta_vn**2 + bt2 - 2.0 * beta_vn * bt * mu) / (beta_vn + bt)
        keep = rng.random(m) < accept
        speeds.append(bt[keep][:need])
        cosines.append(mu[keep][:need])
        need -= speeds[-1].shape[0]
    vt = np.concatenate(speeds) * np.sqrt(kT / awr)
    mu = np.concatenate(cosines)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    sine = np.sqrt(1.0 - mu * mu)
    v_t = np.stack([vt * sine * np.cos(phi), vt * sine * np.sin(phi), vt * mu], axis=1)
    v_n = np.array([0.0, 0.0, np.sqrt(e)])
    v_cm = (v_n + awr * v_t) / (awr + 1.0)
    speed = np.linalg.norm(v_n - v_cm, axis=1)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    v = speed[:, None] * direction + v_cm
    return (v * v).sum(axis=1)


@pytest.mark.parametrize(("awr", "e"), CASES)
def test_free_gas_kernel_matches_openmc_cxs_sampling(awr, e):
    rng = np.random.default_rng(int(awr * 100 + e * 1000))
    samples = cxs_sample(awr, e, KT, 1_000_000, rng)
    edges = np.quantile(samples, np.linspace(0.0, 1.0, 41))
    norm = free_gas_norm(e, awr, KT)
    p = np.array(
        [
            quad(lambda x: free_gas_kernel(e, x, awr, KT), a, b, limit=200)[0]
            for a, b in itertools.pairwise(edges)
        ]
    )
    expected = p / norm * samples.shape[0]
    counts = np.histogram(samples, edges)[0]
    statistic = np.sum((counts - expected) ** 2 / expected)
    assert chi2.sf(statistic, edges.shape[0] - 2) > 1e-4


@pytest.mark.parametrize(("awr", "e"), CASES)
def test_free_gas_kernel_integrates_to_its_norm(awr, e):
    low, high = free_gas_range(e, awr, KT)
    alpha = ((awr - 1.0) / (awr + 1.0)) ** 2

    def k(x):
        return free_gas_kernel(e, x, awr, KT)

    total = quad(k, low, e, points=[alpha * e], limit=500)[0] + quad(k, e, high, limit=500)[0]
    assert total == pytest.approx(free_gas_norm(e, awr, KT), rel=1e-8)


@pytest.mark.parametrize("awr", [0.999167, 15.86, 236.0])
def test_free_gas_kernel_satisfies_detailed_balance(awr):
    for e in (0.01, 0.1, 1.0):
        out = np.array([0.3 * e, 0.9 * e, 1.2 * e, e + 3.0 * KT])
        forward = e * np.exp(-e / KT) * free_gas_kernel(e, out, awr, KT)
        back = out * np.exp(-out / KT) * free_gas_kernel(out, e, awr, KT)
        keep = forward > 1e-300
        np.testing.assert_allclose(forward[keep], back[keep], rtol=1e-11)


def test_free_gas_kernel_has_no_overflow_far_below_e():
    # Hydrogen at 400 kT scattering to nearly zero: e^(E / kT) alone would overflow.
    out = free_gas_kernel(400.0 * KT, np.array([1e-5, 1e-3, 1.0]), 0.999167, KT)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)


def small_coarse(top=30.0, bottom=1e-5):
    return coarse_grid(np.array([top, bottom]), 0.00125)


@pytest.mark.parametrize(("awr", "e"), CASES)
def test_free_gas_row_matches_quadrature_and_conserves(awr, e):
    coarse = small_coarse()
    row, below = free_gas_row(e, awr, KT, coarse)
    assert row.sum() + below == pytest.approx(1.0, abs=1e-14)
    grid_u = np.log(coarse[0] / coarse)
    norm = free_gas_norm(e, awr, KT)
    for j in np.argsort(row)[-3:]:

        def integrand(x, j=j):
            hat = np.interp(np.log(coarse[0] / x), grid_u[j - 1 : j + 2], [0.0, 1.0, 0.0])
            return free_gas_kernel(e, x, awr, KT) * hat

        ref = quad(integrand, coarse[j + 1], coarse[j - 1], points=[e, coarse[j]], limit=200)[0]
        assert row[j] == pytest.approx(ref / norm, rel=1e-7)


def openmc_sab_sample(dist, e, n, rng):
    """IncoherentInelasticAE::sample_params' outgoing energy, from openmc.data's tables."""
    energy = np.asarray(dist.energy)
    i, f = 0, 0.0
    if e >= energy[0]:
        i = min(np.searchsorted(energy, e, side="right") - 1, energy.shape[0] - 1)
        if i + 1 < energy.shape[0]:
            f = (e - energy[i]) / (energy[i + 1] - energy[i])
    chosen = i + 1 if f > 0.5 else i
    table = dist.energy_out[chosen]
    x, p, c = np.asarray(table.x), np.asarray(table.p), np.asarray(table.c)
    r1 = rng.random(n)
    j = np.clip(np.searchsorted(c, r1, side="right") - 1, 0, x.shape[0] - 2)
    slope = (p[j + 1] - p[j]) / (x[j + 1] - x[j])
    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.sqrt(np.maximum(0.0, p[j] ** 2 + 2.0 * slope * (r1 - c[j])))
        out = np.where(slope == 0.0, x[j] + (r1 - c[j]) / p[j], x[j] + (root - p[j]) / slope)
    e_l = energy[chosen]
    return np.where(out < 0.5 * e_l, out * (2.0 * e / e_l - 1.0), out + e - e_l)


@needs_library
@pytest.mark.parametrize("e", [3e-3, 0.0517, 0.3, 2.0, 7.7, 9.99])
def test_sab_segments_match_openmc_sampling(e):
    import openmc.data

    from zerog.data.library import nuclide_path
    from zerog.data.thermal import read_thermal

    data = openmc.data.ThermalScattering.from_hdf5(nuclide_path("c_H_in_H2O", "thermal"))
    dist = data.inelastic.distribution["600K"]
    rng = np.random.default_rng(int(e * 1e4))
    samples = openmc_sab_sample(dist, e, 1_000_000, rng)
    bottom = 1e-6
    lower, upper, weights, below = sab_segments(
        sab_table(read_thermal("c_H_in_H2O", "600K")), e, 0.001, bottom
    )
    assert weights.sum() + below == pytest.approx(1.0, abs=1e-12)
    edges = np.unique(np.quantile(samples[samples > bottom], np.linspace(0.0, 1.0, 51)))
    edges[0] = bottom
    edges[-1] = max(edges[-1], upper.max()) * (1.0 + 1e-7)
    expected = bin_segments(lower, upper, weights, edges) * samples.shape[0]
    counts = np.histogram(samples, edges)[0]
    statistic = np.sum((counts - expected) ** 2 / expected)
    assert chi2.sf(statistic, edges.shape[0] - 2) > 1e-4


def test_sab_choice_is_the_nearer_table_as_openmc_picks_it():
    table = thermal.SabTable(
        energy=np.array([1.0, 2.0, 4.0]),
        offsets=np.array([0, 2, 4, 6]),
        e_out=np.zeros(6),
        pdf=np.zeros(6),
        cdf=np.zeros(6),
    )
    assert [sab_choice(table, e) for e in (0.5, 1.0, 1.5, 1.51, 3.0, 3.01, 4.0, 9.0)] == [
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        2,
    ]


def maxwellian_case(awr, scale=1.0, nodes=3000, sigma_b=20.0):
    """A free gas scatterer with no absorption, a flat streaming term a and the source a M.
    Detailed balance makes M, the Maxwellian per unit lethargy, the exact solution."""
    top = 400.0 * KT
    fine = np.geomspace(4.0 * top, 1e-5, nodes)
    fine = np.unique(np.concatenate([fine, [top, top * (1.0 - 1e-10)]]))[::-1]
    coarse = coarse_grid(fine, 0.00125)
    saved = thermal.HATS_PER_WIDTH, thermal.FLUX_DU, thermal.THERMAL_DU_MAX
    thermal.HATS_PER_WIDTH /= scale
    thermal.FLUX_DU *= scale
    thermal.THERMAL_DU_MAX *= scale
    try:
        coarse = thermal_coarse(coarse, free_gas_range(top, awr, KT)[1], KT, awr, 0.00125)
    finally:
        thermal.HATS_PER_WIDTH, thermal.FLUX_DU, thermal.THERMAL_DU_MAX = saved
    energy = np.unique(np.concatenate([fine, coarse]))[::-1]
    sigma = sigma_b * free_gas_norm(energy, awr, KT)
    term = ThermalTerm(awr, KT, top, energy, sigma)
    kernels = build_thermal([term], energy, build_inelastic([None], energy, coarse), workers=1)
    at_rest = np.where(energy >= top, sigma, 0.0)
    ops = build_operators(
        energy, sigma[None], at_rest[None], [kinematic_alpha(awr)], inelastic=kernels
    )
    a = 2.0
    maxwellian = (energy / KT) ** 2 * np.exp(-energy / KT)
    return ops, a, maxwellian


@pytest.mark.parametrize(("awr", "tolerance"), [(0.999167, 3e-4), (15.86, 5e-4), (236.0, 1.5e-3)])
def test_free_gas_block_holds_a_maxwellian(awr, tolerance):
    ops, a, maxwellian = maxwellian_case(awr)
    solution = solve_energy(ops, [1.0], [[1.0]], a, a * maxwellian)
    x = ops.energy / KT
    # Below 0.1 kT the grid's bottom, 1e-5 eV, cuts off upscatter from below.
    near = (x >= 0.1) & (x <= 3.0)
    assert np.abs(solution.flux[near] / maxwellian[near] - 1.0).max() < tolerance
    balance = energy_balance(ops, [1.0], [[1.0]], a, a * maxwellian, solution.flux)
    assert abs(balance.residual / balance.source) < 1e-4
    assert solution.block_marches > 0


def test_maxwellian_error_falls_at_second_order_in_the_hats():
    errors = []
    for scale in (1.0, 0.5):
        ops, a, maxwellian = maxwellian_case(236.0, scale, nodes=int(3000 / scale))
        flux = solve_energy(ops, [1.0], [[1.0]], a, a * maxwellian).flux
        near = (ops.energy / KT >= 0.1) & (ops.energy / KT <= 3.0)
        errors.append(np.abs(flux[near] / maxwellian[near] - 1.0).max())
    assert errors[0] / errors[1] > 3.0


def test_block_solve_matches_plain_sweeps(monkeypatch):
    ops, a, maxwellian = maxwellian_case(15.86, nodes=800, sigma_b=2.0)
    source = a * maxwellian * (1.0 + np.sin(np.log(ops.energy)))
    block = solve_energy(ops, [1.0], [[1.0]], a, source)
    assert upscatter_block(ops.inelastic) is not None
    monkeypatch.setattr(subproblem, "upscatter_block", lambda kernels: None)
    monkeypatch.setattr(subproblem, "MAX_SWEEPS", 2000)
    plain = solve_energy(ops, [1.0], [[1.0]], a, source)
    assert plain.block_marches == 0
    assert plain.sweeps > 10
    np.testing.assert_allclose(block.flux, plain.flux, rtol=1e-9, atol=1e-12 * plain.flux.max())


@needs_library
@pytest.mark.parametrize("name", ["H1", "O16", "U235", "U238", "Zr90"])
def test_elastic_angles_are_isotropic_where_free_gas_applies(name):
    """OpenMC samples mu_cm from the elastic tables at E; the kernels take them as isotropic."""
    from zerog.energy.terms import _elastic_angle, awr

    angle = _elastic_angle(name)
    top = (thermal.LIGHT_FREE_GAS_THRESHOLD if awr(name) <= 1.0 else 400.0) * KT
    if angle is None:
        return
    energy = angle.energy  # descending

    def mean(k):
        a, b = angle.offsets[k], angle.offsets[k + 1]
        mu, pdf = angle.mu[a:b], angle.pdf[a:b]
        return np.trapezoid(mu * pdf, mu) / np.trapezoid(pdf, mu)

    for e in (1e-3, 1.0, top):
        # AngleDistribution::sample takes the upper table with probability f, linear in E.
        hi = max(np.searchsorted(-energy, -e, side="right") - 1, 0)
        lo = min(hi + 1, energy.shape[0] - 1)
        f = 0.0 if lo == hi else (e - energy[lo]) / (energy[hi] - energy[lo])
        assert abs((1.0 - f) * mean(lo) + f * mean(hi)) < 1e-3


def test_hats_with_inelastic_and_thermal_rows_mix_and_conserve():
    """A term that emits inelastic neutrons in the thermal range, as U235's MT 5 does, takes a
    mixed row there whose weights are the two emissions' shares."""
    ops, _, _ = maxwellian_case(15.86, nodes=600)
    base = ops.inelastic
    term = ThermalTerm(15.86, KT, 400.0 * KT, ops.energy, np.full(ops.energy.shape[0], 3.0))
    n = base.coarse.shape[0]
    # An inelastic row on every hat: half to the hat below, half lost.
    first = np.minimum(np.arange(n) + 1, n - 1)[None, :]
    rows = thermal.SparseRows(np.arange(n), first[0], first[0] + 1, np.full(n, 0.5))
    inelastic = InelasticKernels(
        coarse=base.coarse,
        coarse_index=base.coarse_index,
        hat=base.hat,
        emission=np.ones((1, ops.energy.shape[0])),
        starts=rows.starts[None, :],
        first=first,
        last=first + 1,
        values=rows.values,
        lost=np.full((1, n), 0.5),
    )
    mixed = build_thermal([term], ops.energy, inelastic, workers=1)
    pure = build_thermal([term], ops.energy, build_inelastic([None], ops.energy, base.coarse))
    sums = np.array(
        [
            mixed.values[
                mixed.starts[0, j] : mixed.starts[0, j] + mixed.last[0, j] - mixed.first[0, j]
            ].sum()
            for j in range(n)
        ]
    )
    np.testing.assert_allclose(sums + mixed.lost[0], 1.0, atol=1e-13)
    hot = np.nonzero(pure.last[0] > pure.first[0])[0]
    # Emission 3 thermal against 1 inelastic: three quarters of each mixed hat is thermal.
    j = hot[len(hot) // 2]
    expected = 0.75 * pure.lost[0, j] + 0.25 * 0.5
    assert mixed.lost[0, j] == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(mixed.emission, inelastic.emission + pure.emission)


def test_block_marches_from_its_reach_are_the_whole_march(monkeypatch):
    """F is zero above the block's reach for a block-only source, so starting the march there
    must give the same Krylov operator as marching the whole grid."""
    ops, a, maxwellian = maxwellian_case(15.86, nodes=800, sigma_b=2.0)
    source = a * maxwellian
    reference = solve_energy(ops, [1.0], [[1.0]], a, source)
    first_hat, start = upscatter_block(ops.inelastic)
    assert start > 0
    monkeypatch.setattr(subproblem, "upscatter_block", lambda kernels: (first_hat, 0))
    whole = solve_energy(ops, [1.0], [[1.0]], a, source)
    assert whole.block_marches == reference.block_marches
    np.testing.assert_allclose(whole.flux, reference.flux, rtol=1e-12, atol=0.0)


@needs_library
def test_bound_hydrogen_total_takes_the_table_at_and_below_energy_max():
    import openmc.data

    from zerog.data.library import nuclide_path
    from zerog.energy.terms import apply_thermal, load_term, thermal_required, thermal_spec

    spec = thermal_spec("H1", "600K", "c_H_in_H2O")
    energy = np.unique(
        np.concatenate([np.geomspace(30.0, 1e-5, 400), thermal_required(spec, 1e-5)])
    )[::-1]
    term = apply_thermal(load_term("H1", "600K", energy), spec, energy)
    hydrogen = openmc.data.IncidentNeutron.from_hdf5(nuclide_path("H1"))
    sab = openmc.data.ThermalScattering.from_hdf5(nuclide_path("c_H_in_H2O", "thermal"))
    total = hydrogen[1].xs["600K"](energy)
    elastic = hydrogen[2].xs["600K"](energy)
    bound = sab.inelastic.xs["600K"](energy)
    expected = np.where(energy <= 10.0, total - elastic + bound, total)
    np.testing.assert_allclose(term.total, expected, rtol=1e-10)
    # Target at rest only at and above free_top.
    assert np.all(term.elastic[energy < spec.free_top] == 0.0)
    np.testing.assert_allclose(
        term.elastic[energy >= spec.free_top], elastic[energy >= spec.free_top]
    )


@needs_library
def test_bound_hydrogen_block_conserves_neutrons():
    """A solve through S(alpha, beta) and free gas rows: every row places or loses all it emits,
    and the balance closes."""
    from zerog.energy.terms import apply_thermal, load_term, thermal_required, thermal_spec

    spec = thermal_spec("H1", "600K", "c_H_in_H2O", threshold=1000.0)
    top = 4.0 * spec.free_top
    fine = np.unique(np.concatenate([np.geomspace(top, 1e-5, 2000), thermal_required(spec, 1e-5)]))[
        ::-1
    ]
    coarse = coarse_grid(fine, 0.00125)
    coarse = thermal_coarse(
        coarse, free_gas_range(spec.free_top, spec.awr, spec.kT)[1], spec.kT, spec.awr, 0.00125
    )
    energy = np.unique(np.concatenate([fine, coarse]))[::-1]
    term = apply_thermal(load_term("H1", "600K", energy), spec, energy)
    kernels = build_thermal([spec], energy, build_inelastic([None], energy, coarse))
    sums = np.array(
        [
            kernels.values[
                kernels.starts[0, j] : kernels.starts[0, j]
                + kernels.last[0, j]
                - kernels.first[0, j]
            ].sum()
            for j in range(coarse.shape[0])
        ]
    )
    emitting = kernels.last[0] > kernels.first[0]
    np.testing.assert_allclose(sums[emitting] + kernels.lost[0, emitting], 1.0, atol=1e-12)
    ops = build_operators(
        energy, term.total[None], term.elastic[None], [term.alpha], inelastic=kernels
    )
    # A little absorption everywhere, and a source high in the block.
    source = np.exp(-(np.log(energy / 5.0) ** 2) / 0.01)
    solution = solve_energy(ops, [1.0], [[1.0]], 0.05, source)
    balance = energy_balance(ops, [1.0], [[1.0]], 0.05, source, solution.flux)
    assert abs(balance.residual / balance.source) < 1e-3


def test_round_off_upward_deposits_do_not_start_the_block():
    """An inelastic row's round-off above its neighbour is lagged by the sweeps, not solved by
    Krylov: only a real share of a row going up starts the block."""

    energy = np.geomspace(1e6, 1e3, 200)
    coarse = coarse_grid(energy, 0.01)
    energy = np.unique(np.concatenate([energy, coarse]))[::-1]
    base = build_inelastic([None], energy, coarse)
    n = coarse.shape[0]
    j = n // 2
    # One row with 1e-12 of itself two hats up, then the same row with a tenth.
    for share, expect in ((1e-12, False), (0.1, True)):
        values = np.array([share, 0.0, 1.0 - share])
        kernels = InelasticKernels(
            **{
                **base.__dict__,
                "starts": np.zeros((1, n), dtype=np.int64),
                "first": np.where(np.arange(n) == j, j - 2, 0)[None, :],
                "last": np.where(np.arange(n) == j, j + 1, 0)[None, :],
                "values": values,
            }
        )
        assert (upscatter_block(kernels) is not None) == expect
