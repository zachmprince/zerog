"""Inelastic kernels: exact limits, and OpenMC's own sampling from benchmarks/03-secondary-laws."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from zerog.data.neutron import read_reactions
from zerog.data.scatter import AngleTables, LevelEnergy, UncorrelatedLaw, read_scattering_laws
from zerog.energy.inelastic import bin_segments, law_points, reaction_points, to_lab

needs_library = pytest.mark.skipif(
    "OPENMC_CROSS_SECTIONS" not in os.environ, reason="OPENMC_CROSS_SECTIONS is not set"
)
ORACLE = Path(__file__).parents[2] / "benchmarks" / "03-secondary-laws"


def isotropic_level(threshold, mass_ratio):
    angle = AngleTables(
        energy=np.array([2.0e7, 1.0e-5]),
        offsets=np.array([0, 2, 4]),
        interpolation=np.array([2, 2]),
        mu=np.array([-1.0, 1.0, -1.0, 1.0]),
        pdf=np.full(4, 0.5),
        cdf=np.array([0.0, 1.0, 0.0, 1.0]),
    )
    return UncorrelatedLaw(angle=angle, energy=LevelEnergy(threshold, mass_ratio))


def test_isotropic_level_is_uniform_in_lab_energy():
    # With an isotropic centre of mass, lab energy is uniform between its mu = -1 and +1 limits.
    awr = 55.45
    law = isotropic_level(0.85e6, (awr / (awr + 1.0)) ** 2)
    e = 2.0e6
    lower, upper, weights = law_points(law, e, awr, True, n_cosine=64)
    e_cm = law.energy.mass_ratio * (e - law.energy.threshold)
    low, high = to_lab(e, e_cm, -1.0, awr), to_lab(e, e_cm, 1.0, awr)
    assert weights.sum() == pytest.approx(1.0, rel=1e-14)
    assert lower.min() == pytest.approx(low, rel=1e-12)
    assert upper.max() == pytest.approx(high, rel=1e-12)
    # Each stratum is an exact interval; spread in lethargy, it is uniform in E to first order.
    edges = np.linspace(low, high, 9)
    np.testing.assert_allclose(bin_segments(lower, upper, weights, edges), 1.0 / 8.0, rtol=2e-3)


def test_segments_fill_narrow_hats_without_gaps():
    # Hats far narrower than a stratum still each get their share, unlike point masses.
    from zerog.energy.inelastic import deposit

    coarse = np.geomspace(2.0e6, 1.0e6, 2001)
    row, lost = deposit(np.array([1.2e6]), np.array([1.6e6]), np.array([1.0]), coarse)
    assert lost == pytest.approx(0.0, abs=1e-15)
    assert row.sum() == pytest.approx(1.0, rel=1e-13)
    inside = (coarse > 1.21e6) & (coarse < 1.59e6)
    widths = np.log(1.6e6 / 1.2e6)
    hat = np.log(coarse[0] / coarse[1])
    np.testing.assert_allclose(row[inside], hat / widths, rtol=1e-9)


def test_deposit_matches_quadrature_of_each_hat():
    from scipy.integrate import quad

    from zerog.energy.inelastic import deposit, hat_weights

    rng = np.random.default_rng(4)
    coarse = np.geomspace(2.0e7, 1.0e5, 61)
    grid_u = np.log(coarse[0] / coarse)
    upper = np.exp(rng.uniform(np.log(5.0e4), np.log(3.0e7), 40))
    lower = upper * np.exp(-rng.uniform(0.0, 3.0, 40))
    lower[:3] = upper[:3]  # points
    weights = rng.uniform(0.1, 1.0, 40)
    row, lost = deposit(lower, upper, weights, coarse)
    expected = np.zeros(coarse.shape[0])
    for lo, hi, w in zip(lower, upper, weights, strict=True):
        a, b = np.log(coarse[0] / hi), np.log(coarse[0] / lo)
        for i in range(coarse.shape[0]):
            hat = np.zeros(coarse.shape[0])
            hat[i] = 1.0
            if b - a < 1e-14:
                expected[i] += w * np.interp(a, grid_u, hat, left=0.0, right=0.0)
            else:
                points = [v for v in grid_u if a < v < b]
                integrand = lambda u, hat=hat: np.interp(u, grid_u, hat, left=0.0, right=0.0)  # noqa: E731
                expected[i] += (
                    w / (b - a) * quad(integrand, a, b, points=points or None, limit=200)[0]
                )
    np.testing.assert_allclose(row, expected, rtol=1e-9, atol=1e-13)
    assert row.sum() + lost == pytest.approx(weights.sum(), rel=1e-13)
    assert hat_weights(coarse).shape == row.shape


@needs_library
@pytest.mark.parametrize(
    "name", ["O16", "Fe56", "Na23", "Cr52", "Pu241", "U238", "Ni58", "Zr90", "Pu239"]
)
def test_point_weights_sum_to_one_for_every_reaction(name):
    awr = None
    for reaction in read_scattering_laws(name):
        if reaction.mt == 2:
            continue
        for e in (2.0e6, 6.0e6, 14.0e6, 19.9e6):
            _, _, weights = reaction_points(reaction, e, awr or 100.0)
            if weights.shape[0] == 0:
                continue
            assert weights.sum() == pytest.approx(1.0, rel=1e-9), reaction.mt


def oracle_cases():
    return sorted(p.stem for p in ORACLE.glob("*.json"))


def _awr(name):
    import h5py

    from zerog.data.library import nuclide_path

    with h5py.File(nuclide_path(name), "r") as f:
        return float(f[name].attrs["atomic_weight_ratio"])


@needs_library
@pytest.mark.parametrize("case", oracle_cases())
def test_outgoing_spectra_match_openmc_sampling(case):
    data = json.loads((ORACLE / f"{case}.json").read_text())
    name = data["nuclide"]
    e0 = data["source_energy"]
    edges = np.array(data["edges"])
    awr = _awr(name)
    curves = read_reactions(name, "294K")
    total = np.interp(e0, curves.energy[::-1], curves.xs.sum(axis=0)[::-1])
    laws = {r.mt: r for r in read_scattering_laws(name)}
    checked = 0
    for mt in data["mts"]:
        twin = np.array(data["mean"][str(mt)])
        if twin.sum() == 0.0:
            continue
        row = list(curves.mt).index(mt)
        share = np.interp(e0, curves.energy[::-1], curves.xs[row, ::-1]) / total
        # A level is one line, so its strata are cheap and a line ending a sliver past a bin edge
        # needs fine ones there.
        level = any(isinstance(getattr(law, "energy", None), LevelEnergy) for law in laws[mt].laws)
        strata = 4096 if level else 256
        lower, upper, weights = reaction_points(laws[mt], e0, awr, du=1e-3, n_cosine=strata)
        ours = share * bin_segments(lower, upper, weights, edges)
        # Each analog event scores 1 / histories, so a bin's variance is its expected count over
        # histories squared. The tally's batch variance is useless in bins holding a few events.
        histories = data["provenance"]["particles"] * data["provenance"]["batches"]
        assert ours.sum() == pytest.approx(twin.sum(), abs=4.0 * np.sqrt(ours.sum() / histories))
        seen = (ours > 1e-3 / histories) | (twin > 0.0)
        sd = np.sqrt(np.maximum(ours[seen], 1.0 / histories) / histories)
        chi_squared = float(np.sum(((ours[seen] - twin[seen]) / sd) ** 2))
        # 385 reactions are tested, so 1e-4 each keeps a false alarm near 4% for the lot.
        assert chi2.sf(chi_squared, int(seen.sum())) > 1e-4, (mt, chi_squared, int(seen.sum()))
        checked += 1
    assert checked > 0


def fe56_inelastic_case(coarse_du, nodes=401, smooth=False):
    from zerog.energy.inelastic import build_inelastic, coarse_grid
    from zerog.energy.subproblem import build_operators

    fine = np.geomspace(14.0e6, 2.0e5, nodes)
    coarse = coarse_grid(fine, coarse_du)
    energy = np.unique(np.concatenate([fine, coarse]))[::-1]
    curves = read_reactions("Fe56", "294K")
    total = np.interp(energy[::-1], curves.energy[::-1], curves.xs.sum(axis=0)[::-1])[::-1]
    if smooth:
        # This grid ignores Fe56's resonances, so a smooth removal keeps truncation out of balance.
        total = 3.0 + 0.1 * np.log(14.0e6 / energy)
    kernels = build_inelastic([("Fe56", "294K", _awr("Fe56"))], energy, coarse)
    # Elastic scattering is left out, so the inelastic transfer is all that moves neutrons.
    ops = build_operators(
        energy, total[None, :], np.zeros((1, energy.shape[0])), [0.0], inelastic=kernels
    )
    return energy, ops, kernels


def test_coarse_grid_has_exact_ends_and_no_slivers():
    # Required energies may be node pairs a hair apart on purpose; the grid's own nodes may not.
    from zerog.energy.inelastic import coarse_grid

    fine = np.geomspace(14.0e6, 2.0e5, 401)
    coarse = coarse_grid(fine, 0.05, required=[1.0e6])
    assert coarse[0] == fine[0]
    assert coarse[-1] == fine[-1]
    assert np.log(coarse[:-1] / coarse[1:]).min() > 1e-12


@needs_library
def test_march_converges_to_the_discrete_inelastic_system():
    from zerog.energy.inelastic import restrict
    from zerog.energy.subproblem import solve_energy

    energy, ops, kernels = fe56_inelastic_case(0.05)
    n = energy.shape[0]
    density = 0.08
    source = np.exp(-(((np.log(14.0e6 / energy) - 0.05) / 0.02) ** 2))
    solution = solve_energy(ops, [density], [[density]], 0.0, source)
    # The same equations as a dense matrix: restrict, transfer, then interpolate onto the fine grid.
    restriction = np.array(
        [restrict(kernels, energy, np.eye(n)[i], ops.rate_weights)[0] for i in range(n)]
    ).T
    rows = np.zeros((kernels.coarse.shape[0], kernels.coarse.shape[0]))
    for j in range(rows.shape[0]):
        start = kernels.starts[0, j]
        rows[j, kernels.first[0, j] : kernels.last[0, j]] = kernels.values[
            start : start + kernels.last[0, j] - kernels.first[0, j]
        ]
    prolong = np.zeros((n, kernels.coarse.shape[0]))
    prolong[np.arange(n), ops.node_m] = (1.0 - ops.node_theta) / kernels.hat[ops.node_m]
    prolong[np.arange(n), ops.node_m + 1] = ops.node_theta / kernels.hat[ops.node_m + 1]
    matrix = np.diag(density * ops.removal[0]) - density * prolong @ rows.T @ restriction
    np.testing.assert_allclose(solution.flux, np.linalg.solve(matrix, source), rtol=1e-9)
    assert solution.sweeps < 10


@needs_library
def test_inelastic_transfer_conserves_neutrons():
    # The transfer itself is exact, so what is left is collocation truncation, second order.
    from zerog.energy.balance import energy_balance
    from zerog.energy.subproblem import solve_energy

    closures = []
    for nodes in (401, 1601):
        energy, ops, _ = fe56_inelastic_case(0.02, nodes, smooth=True)
        source = np.exp(-(((np.log(14.0e6 / energy) - 0.05) / 0.02) ** 2))
        flux = solve_energy(ops, [0.08], [[0.08]], 0.0, source).flux
        balance = energy_balance(ops, [0.08], [[0.08]], 0.0, source, flux)
        assert balance.lost_below > 0.0
        closures.append(abs(balance.residual) / balance.source)
    assert closures[1] < closures[0] / 8.0
    assert closures[1] < 1e-5


@needs_library
def test_cached_kernels_are_the_kernels(tmp_path):
    from zerog.energy.inelastic import build_inelastic, coarse_grid

    fine = np.geomspace(14.0e6, 2.0e6, 201)
    coarse = coarse_grid(fine, 0.05)
    energy = np.unique(np.concatenate([fine, coarse]))[::-1]
    terms = [("O16", "294K", _awr("O16"))]
    built = build_inelastic(terms, energy, coarse, cache=tmp_path)
    assert len(list(tmp_path.glob("inelastic-*.npz"))) == 1
    loaded = build_inelastic(terms, energy, coarse, cache=tmp_path)
    for name, value in built.__dict__.items():
        np.testing.assert_array_equal(getattr(loaded, name), value)
    # A different grid is a different key.
    other_fine = np.geomspace(14.0e6, 2.5e6, 201)
    other_coarse = coarse_grid(other_fine, 0.05)
    other = np.unique(np.concatenate([other_fine, other_coarse]))[::-1]
    build_inelastic(terms, other, other_coarse, cache=tmp_path)
    assert len(list(tmp_path.glob("inelastic-*.npz"))) == 2


@needs_library
@pytest.mark.parametrize("name", ["U238", "Fe56"])
def test_every_emitting_hat_places_all_its_neutrons(name):
    # Hats straddling thresholds can have centroids below one; their share must not vanish.
    from zerog.energy.inelastic import coarse_grid, term_transfer
    from zerog.energy.terms import _thresholds, awr

    fine = np.geomspace(14.0e6, 1.0e5, 401)
    coarse = coarse_grid(fine, 0.02, _thresholds([(name, "294K")], 14.0e6, 1.0e5))
    rows, lost, _, _ = term_transfer(name, "294K", awr(name), coarse, du=0.005, n_cosine=8)
    sums = rows.sums()
    emitting = sums + lost > 0.0
    assert emitting.sum() > 100
    np.testing.assert_allclose(sums[emitting] + lost[emitting], 1.0, rtol=1e-12)


def _transfer_oracle(name, awr, coarse, du, n_cosine):
    """term_transfer's rows the plain way: every reaction's segment arrays, deposited densely."""
    from zerog.data.neutron import read_reactions, read_scattering_yields
    from zerog.energy.inelastic import _hat_moments, deposit

    curves = read_reactions(name, "294K")
    laws = [r for r in read_scattering_laws(name) if r.mt != 2]
    mts = np.array([law.mt for law in laws], dtype=np.int64)
    index = np.array([list(curves.mt).index(mt) for mt in mts])
    produced = curves.xs[index] * read_scattering_yields(name, mts, curves.energy)
    moments = [_hat_moments(coarse, curves.energy, curve) for curve in produced]
    rows = np.zeros((coarse.shape[0], coarse.shape[0]))
    lost = np.zeros(coarse.shape[0])
    for j in range(coarse.shape[0]):
        total = sum(m[0][j] for m in moments)
        if total <= 0.0:
            continue
        for law, (integral, centroid) in zip(laws, moments, strict=True):
            if integral[j] <= 0.0:
                continue
            share = integral[j] / total
            lo, hi, w = reaction_points(law, centroid[j], awr, du, n_cosine)
            if w.shape[0] == 0 and j > 0:
                lo, hi, w = reaction_points(law, coarse[j - 1], awr, du, n_cosine)
            if w.shape[0] == 0:
                lost[j] += share
                continue
            row, off = deposit(lo, hi, w * share, coarse)
            rows[j] += row
            lost[j] += off
    return rows, lost


@needs_library
@pytest.mark.parametrize(("name", "bottom"), [("U238", 4.0e6), ("O16", 1.0e5), ("Na23", 1.0e5)])
def test_transfer_rows_match_depositing_each_reactions_segments(name, bottom):
    # term_transfer deposits strata as it makes them and sums every reaction of a hat into one
    # row, so it differs from the plain deposit only in the order of summation. Workers split hats
    # and must not change a bit. Levels and laws in the centre of mass, correlated (U238) and
    # Kalbach-Mann (O16), go by fans of strata; Na23's lab-frame evaporation by plain segments.
    from zerog.energy.inelastic import coarse_grid, term_transfer
    from zerog.energy.terms import _thresholds, awr

    fine = np.geomspace(14.0e6, bottom, 201)
    coarse = coarse_grid(fine, 0.02, _thresholds([(name, "294K")], 14.0e6, bottom))
    rows, lost, _, _ = term_transfer(name, "294K", awr(name), coarse, 0.005, 8, workers=1)
    expected, expected_lost = _transfer_oracle(name, awr(name), coarse, 0.005, 8)
    dense = rows.dense()
    scale = np.maximum(np.abs(expected).max(axis=1, keepdims=True), 1e-300)
    assert np.all(np.abs(dense - expected) <= 1e-13 * scale)
    np.testing.assert_allclose(lost, expected_lost, rtol=0.0, atol=1e-13)
    parallel, parallel_lost, _, _ = term_transfer(
        name, "294K", awr(name), coarse, 0.005, 8, workers=2
    )
    for field in ("starts", "first", "last", "values"):
        np.testing.assert_array_equal(getattr(parallel, field), getattr(rows, field))
    np.testing.assert_array_equal(parallel_lost, lost)


@needs_library
def test_shared_pool_builds_the_same_kernels_as_one_term_at_a_time():
    # build_inelastic deals every term's hats into one pool; no bit may depend on that.
    from zerog.energy.inelastic import build_inelastic, coarse_grid, term_transfer
    from zerog.energy.terms import _thresholds, awr

    names = ["U238", "O16", "Na23"]
    pairs = [(n, "294K") for n in names]
    fine = np.geomspace(14.0e6, 4.0e6, 101)
    coarse = coarse_grid(fine, 0.02, _thresholds(pairs, 14.0e6, 4.0e6))
    energy = np.unique(np.concatenate([fine, coarse]))[::-1]
    terms = [(n, "294K", awr(n)) for n in names]
    shared = build_inelastic([terms[0], None, *terms[1:]], energy, coarse, 0.005, 8, workers=3)
    for t, name in zip([0, 2, 3], names, strict=True):
        rows, lost, _, _ = term_transfer(name, "294K", awr(name), coarse, 0.005, 8, workers=1)
        a = shared.starts[t, 0]
        stored = shared.values[a : a + rows.values.shape[0]]
        np.testing.assert_array_equal(shared.starts[t] - a, rows.starts)
        np.testing.assert_array_equal(shared.first[t], rows.first)
        np.testing.assert_array_equal(shared.last[t], rows.last)
        np.testing.assert_array_equal(stored, rows.values)
        np.testing.assert_array_equal(shared.lost[t], lost)
    assert not shared.first[1].any()
    assert not shared.last[1].any()


@needs_library
def test_kernels_from_another_grid_are_refused():
    from dataclasses import replace

    from zerog.energy.subproblem import build_operators

    energy, _, kernels = fe56_inelastic_case(0.05)
    # One extra node shifts every coarse index past it, so the stored indices no longer match.
    extra = np.sort(np.concatenate([energy, [0.5 * (energy[3] + energy[4])]]))[::-1]
    n = extra.shape[0]
    moved = replace(kernels, emission=np.zeros((1, n)))
    with pytest.raises(ValueError, match="grid"):
        build_operators(extra, np.ones((1, n)), np.zeros((1, n)), [0.0], inelastic=moved)


@needs_library
def test_a_warm_start_reproduces_the_cold_solve():
    from zerog.energy.inelastic import restrict
    from zerog.energy.subproblem import solve_energy

    energy, ops, kernels = fe56_inelastic_case(0.05)
    source = np.exp(-(((np.log(14.0e6 / energy) - 0.05) / 0.02) ** 2))
    cold = solve_energy(ops, [0.08], [[0.08]], 0.0, source)
    guess = restrict(kernels, energy, 1.01 * cold.flux, ops.rate_weights)
    warm = solve_energy(ops, [0.08], [[0.08]], 0.0, source, initial=guess)
    np.testing.assert_allclose(warm.flux, cold.flux, rtol=1e-11, atol=1e-11 * cold.flux.max())
    assert warm.sweeps < cold.sweeps
