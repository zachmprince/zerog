"""Thermal scattering, free gas and S(alpha, beta), as rows on inelastic transfer's coarse hats.

Both move neutrons up as well as down, so their rows reach above the emitting hat, and the march
leaves that part lagged; solve_energy then converges the block of hats they span by Krylov. Each
follows OpenMC v0.16.0's sampling, with its random numbers replaced by quadrature:

- sample_target_velocity (physics.cpp) takes the target at rest at E >= free_gas_threshold * kT
  (400 kT by default) unless awr <= 1, and otherwise samples it with the constant cross section
  approximation (sample_cxs_target_velocity), since resonance scattering methods are off by
  default. The expected kernel of that sampler, with mu_cm isotropic, is the free gas kernel of
  free_gas_kernel, normalized: the collision rate is the tabulated elastic cross section at E.
  OpenMC samples mu_cm from the elastic angular distribution at E, which these kernels take as
  isotropic; see tests/energy/test_thermal_scattering.py for how far below 400 kT that holds.
- Hydrogen (awr <= 1) is free gas at every energy above its S(alpha, beta) table. Here it is
  free gas below LIGHT_FREE_GAS_THRESHOLD * kT and target at rest above. Free gas smears the top
  of hydrogen's kernel by about sqrt(4 kT E), wider than U238's low resonances, so this matters:
  in study 03's PWR mix, raising the threshold from 400 to 4e3, 4e4 and 4e5 kT moved k by -36,
  -4.8 and -0.2 pcm.
- Material::calculate_neutron_xs uses an S(alpha, beta) table at E <= its energy_max, where its
  cross section replaces the nuclide's elastic one (Nuclide::calculate_sab_xs).
- IncoherentInelasticAE::sample_params takes the nearer incoming table l, where get_energy_index
  gives i and f, l = i + 1 if f > 0.5; samples E' from its lin-lin pdf bin by bin, so that bin j
  holds c[j + 1] - c[j] of the probability; then scales E' by 2 E / E_l - 1 below E_l / 2 and
  shifts it by E - E_l above.

A row is the hat-weighted average of single-collision rows over the incoming hat, by Gauss
quadrature on each half-hat in lethargy, weighted by hat and cross section. That assumes F is flat
across a hat, as restriction does. Intervals are split where the physics changes: at each top and
at each S(alpha, beta) midpoint between tables.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.special import erf, erfc, erfcx

from zerog.energy.inelastic import (
    SparseRows,
    _assemble_rows,
    _start_method,
    _worker_count,
    deposit,
    point_spacing,
)

FREE_GAS_THRESHOLD = 400.0  # OpenMC's settings::free_gas_threshold, in kT
LIGHT_FREE_GAS_THRESHOLD = 4.0e4  # in kT, for awr <= 1, which OpenMC never takes at rest
TARGET_SPREAD = 8.0  # target speeds beyond this many thermal speeds carry e^-64 of the Maxwellian
INCOMING_POINTS = 2  # Gauss points per piece of a half-hat
OUTGOING_POINTS = 3  # Gauss points per coarse interval of the free gas kernel
TAIL_POINTS = 32  # Gauss points for the free gas kernel below the grid
HATS_PER_WORKER = 64
CDF_TOLERANCE = 1e-9  # how far an S(alpha, beta) cdf may start from 0 or end from 1

_IN_S, _IN_W = np.polynomial.legendre.leggauss(INCOMING_POINTS)
_IN_S, _IN_W = 0.5 * (_IN_S + 1.0), 0.5 * _IN_W
_OUT_S, _OUT_W = np.polynomial.legendre.leggauss(OUTGOING_POINTS)
_OUT_S, _OUT_W = 0.5 * (_OUT_S + 1.0), 0.5 * _OUT_W
_TAIL_S, _TAIL_W = np.polynomial.legendre.leggauss(TAIL_POINTS)
_TAIL_S, _TAIL_W = 0.5 * (_TAIL_S + 1.0), 0.5 * _TAIL_W


def _sum_erf(x, y):
    """erf(x) + erf(y), without cancellation when the signs differ."""
    out = erf(x) + erf(y)
    out = np.where((x < 0.0) & (y > 0.0), erfc(-x) - erfc(y), out)
    return np.where((x > 0.0) & (y < 0.0), erfc(-y) - erfc(x), out)


def free_gas_kernel(e, e_out, awr, kT):
    """The free gas kernel per unit E' for isotropic centre-of-mass scattering, for a unit free
    atom cross section: it integrates over E' to free_gas_norm.

    For E' <= E it is eta^2 / 2E times
        erf(eta s' - rho s) + erf(eta s' + rho s)
        + e^(s^2 - s'^2) [erf(eta s - rho s') - erf(eta s + rho s')],
    with s^2 = E / kT, eta = (A + 1) / 2 sqrt(A) and rho = (A - 1) / 2 sqrt(A); above E the
    signs of the second erf in each pair flip. The exponential is folded into erfcx, so it never
    overflows; the bracket it multiplies is bounded by e^-(rho s - eta s')^2.
    """
    e_out = np.asarray(e_out, dtype=float)
    eta = (awr + 1.0) / (2.0 * np.sqrt(awr))
    rho = (awr - 1.0) / (2.0 * np.sqrt(awr))
    s = np.sqrt(e / kT)
    so = np.sqrt(e_out / kT)
    x = (e - e_out) / kT
    a, b = eta * so - rho * s, eta * so + rho * s
    c, d = eta * s - rho * so, eta * s + rho * so
    down = e_out <= e
    first = np.where(down, _sum_erf(a, b), _sum_erf(a, -b))
    # Below E, c and d are positive: e^x [erf(c) - erf(d)] = e^x [erfc(d) - erfc(c)].
    cd = np.maximum(c, 0.0)
    dd = np.maximum(d, 0.0)
    second_down = np.exp(x - dd * dd) * erfcx(dd) - np.exp(x - cd * cd) * erfcx(cd)
    second_up = np.exp(np.minimum(x, 0.0)) * _sum_erf(c, d)
    return eta * eta / (2.0 * e) * (first + np.where(down, second_down, second_up))


def free_gas_norm(e, awr, kT):
    """The free gas kernel's integral: the effective cross section of a unit free atom one."""
    a = np.sqrt(awr * e / kT)
    return (1.0 + 0.5 / (a * a)) * erf(a) + np.exp(-a * a) / (a * np.sqrt(np.pi))


def free_gas_range(e, awr, kT):
    """Outgoing energies outside which the free gas kernel is below e^-64 of its size.

    Lab speeds after a collision lie within 2 A V / (A + 1) of the bounds for a target at rest,
    for target speeds up to V; V is TARGET_SPREAD thermal speeds sqrt(kT / A), in OpenMC's units
    where E = v^2.
    """
    v = np.sqrt(e)
    spread = 2.0 * awr * TARGET_SPREAD * np.sqrt(kT / awr) / (awr + 1.0)
    low = max(0.0, (awr - 1.0) / (awr + 1.0) * v - spread)
    return low * low, (v + spread) ** 2


def free_gas_row(e, awr, kT, coarse):
    """Hat values of the free gas kernel from one collision at e, per neutron, and the part
    landing below the grid.

    Each coarse interval in reach takes OUTGOING_POINTS Gauss points in lethargy, split at e where
    the kernel has a kink. The part below the grid is integrated in energy, and the row is scaled so
    that the two sum to one: the scale differs from one by the quadrature's error.
    """
    grid_u = np.log(coarse[0] / coarse)
    n = coarse.shape[0]
    low, high = free_gas_range(e, awr, kT)
    u_low = np.log(coarse[0] / max(low, coarse[-1])) if low > 0.0 else grid_u[-1]
    u_high = np.log(coarse[0] / min(high, coarse[0]))
    first = max(0, np.searchsorted(grid_u, u_high, side="right") - 1)
    last = min(n - 2, np.searchsorted(grid_u, u_low, side="left"))
    left = grid_u[first : last + 1]
    right = grid_u[first + 1 : last + 2]
    u_e = np.log(coarse[0] / e)
    split = (left < u_e) & (u_e < right)
    # Pieces: each interval, with the one holding e cut in two.
    piece_left = np.concatenate([left[~split], left[split], np.full(split.sum(), u_e)])
    piece_right = np.concatenate([right[~split], np.full(split.sum(), u_e), right[split]])
    index = np.arange(first, last + 1)
    owner = np.concatenate([index[~split], index[split], index[split]])
    width = piece_right - piece_left
    u = piece_left[:, None] + width[:, None] * _OUT_S[None, :]
    e_out = coarse[0] * np.exp(-u)
    density = free_gas_kernel(e, e_out, awr, kT) * e_out  # per unit lethargy
    density *= width[:, None] * _OUT_W[None, :]
    theta = (u - grid_u[owner][:, None]) / (grid_u[owner + 1] - grid_u[owner])[:, None]
    row = np.zeros(n)
    np.add.at(row, owner, (density * (1.0 - theta)).sum(axis=1))
    np.add.at(row, owner + 1, (density * theta).sum(axis=1))
    norm = free_gas_norm(e, awr, kT)
    below = 0.0
    if low < coarse[-1]:
        x = low + (coarse[-1] - low) * _TAIL_S
        below = float((free_gas_kernel(e, x, awr, kT) * _TAIL_W).sum() * (coarse[-1] - low))
        below /= norm
    total = row.sum()
    if total > 0.0:
        row *= (1.0 - below) / total
    return row, below


@dataclass(frozen=True)
class SabTable:
    """An S(alpha, beta) inelastic table in the library's ascending order, for sab_row."""

    energy: np.ndarray  # incoming, ascending
    offsets: np.ndarray
    e_out: np.ndarray  # ascending within each table
    pdf: np.ndarray
    cdf: np.ndarray


def sab_table(thermal):
    """A zerog.data.thermal.ThermalData's inelastic spectrum, ascending, for sab_row."""
    spectrum = thermal.spectrum
    lengths = np.diff(spectrum.offsets)[::-1]
    # sab_segments places c[0]..c[-1] of each table; OpenMC also samples r1 outside that range,
    # into the end bins. Refuse tables where that would matter.
    for i in range(spectrum.energy.shape[0]):
        c = spectrum.cdf[spectrum.offsets[i] : spectrum.offsets[i + 1]]  # descending
        if abs(c[-1]) > CDF_TOLERANCE or abs(1.0 - c[0]) > CDF_TOLERANCE:
            raise ValueError(
                f"{thermal.name} table at {spectrum.energy[i]} eV has a cdf from {c[-1]} to {c[0]}"
            )
    return SabTable(
        energy=spectrum.energy[::-1].copy(),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        e_out=spectrum.energy_out[::-1].copy(),
        pdf=spectrum.pdf[::-1].copy(),
        cdf=spectrum.cdf[::-1].copy(),
    )


def sab_choice(table, e):
    """The incoming table OpenMC samples from at e, by get_energy_index and the nearer one."""
    energy = table.energy
    if e < energy[0]:
        return 0
    i = min(np.searchsorted(energy, e, side="right") - 1, energy.shape[0] - 1)
    if i + 1 >= energy.shape[0]:
        return i
    f = (e - energy[i]) / (energy[i + 1] - energy[i])
    return i + 1 if f > 0.5 else i


def sab_midpoints(table):
    """Incoming energies where the nearer table changes."""
    return 0.5 * (table.energy[:-1] + table.energy[1:])


def _linear_cdf(x, a, p_a, slope):
    return p_a * (x - a) + 0.5 * slope * (x - a) ** 2


def sab_segments(table, e, du, bottom):
    """Weighted lab segments of one collision at e, and the part landing below bottom.

    A bin's probability c[j + 1] - c[j] is spread over [E_j, E*] by its lin-lin pdf, where E* is
    where OpenMC's inversion lands for r1 = c[j + 1]: the bin's top, if pdf and cdf agree.
    """
    chosen = sab_choice(table, e)
    e_l = table.energy[chosen]
    a0, a1 = table.offsets[chosen], table.offsets[chosen + 1]
    x = table.e_out[a0:a1]
    p = table.pdf[a0:a1]
    c = table.cdf[a0:a1]
    scale = 2.0 * e / e_l - 1.0
    if scale <= 0.0:
        raise ValueError(f"S(alpha, beta) scaling {scale} at {e} eV from the table at {e_l} eV")
    lower, upper, weights = [], [], []
    below = 0.0
    for j in range(x.shape[0] - 1):
        prob = c[j + 1] - c[j]
        if prob <= 0.0:
            continue
        a, p_a = x[j], max(p[j], 0.0)
        slope = (p[j + 1] - p[j]) / (x[j + 1] - x[j])
        if slope == 0.0:
            end = a + prob / p_a if p_a > 0.0 else x[j + 1]
        else:
            end = a + (np.sqrt(max(0.0, p_a * p_a + 2.0 * slope * prob)) - p_a) / slope
        if not end > a:
            continue
        # Cut at E_l / 2, where the mapping changes, then map each part.
        cuts = [a, end]
        if a < 0.5 * e_l < end:
            cuts = [a, 0.5 * e_l, end]
        for lo, hi in itertools.pairwise(cuts):
            if lo < 0.5 * e_l:
                m_lo, m_hi = lo * scale, hi * scale
            else:
                m_lo, m_hi = lo + e - e_l, hi + e - e_l
            # The density in the original variable, pieced geometrically in the mapped one.
            if m_hi <= bottom:
                below += prob_of(lo, hi, a, p_a, slope, prob, end)
                continue
            if m_lo < bottom:
                inverse = (bottom - m_lo) / (m_hi - m_lo) * (hi - lo) + lo
                below += prob_of(lo, inverse, a, p_a, slope, prob, end)
                lo, m_lo = inverse, bottom
            pieces = max(1, int(np.ceil(np.log(m_hi / m_lo) / du)))
            mapped = m_lo * np.exp(np.log(m_hi / m_lo) * np.arange(pieces + 1) / pieces)
            original = lo + (mapped - m_lo) / (m_hi - m_lo) * (hi - lo)
            original[-1] = hi
            cdf = _linear_cdf(original, a, p_a, slope)
            w = np.diff(cdf) * prob / _linear_cdf(end, a, p_a, slope)
            lower.append(mapped[:-1])
            upper.append(mapped[1:])
            weights.append(w)
    if not weights:
        return np.empty(0), np.empty(0), np.empty(0), below
    return np.concatenate(lower), np.concatenate(upper), np.concatenate(weights), below


def prob_of(lo, hi, a, p_a, slope, prob, end):
    """The probability of [lo, hi] in a bin of probability prob spread over [a, end]."""
    return (
        prob
        * (_linear_cdf(hi, a, p_a, slope) - _linear_cdf(lo, a, p_a, slope))
        / (_linear_cdf(end, a, p_a, slope))
    )


def sab_row(table, e, coarse, du=None):
    """Hat values of one S(alpha, beta) collision at e, per neutron, and the part off the grid."""
    du = point_spacing(coarse) if du is None else du
    lower, upper, weights, below = sab_segments(table, e, du, coarse[-1])
    if weights.shape[0] == 0:
        return np.zeros(coarse.shape[0]), below
    row, off = deposit(lower, upper, weights, coarse)
    return row, below + off


@dataclass(frozen=True)
class ThermalTerm:
    """How one term scatters below its thermal tops, for build_thermal.

    Free gas applies below free_top and above sab_top, the S(alpha, beta) table at and below
    sab_top, as OpenMC takes E <= energy_max. elastic is the term's elastic cross section on
    curve_energy, descending; sab_xs replaces it at and below sab_top. Without a table, sab_top
    is 0.
    """

    awr: float
    kT: float
    free_top: float
    curve_energy: np.ndarray
    elastic: np.ndarray
    sab_top: float = 0.0
    sab: SabTable | None = None
    sab_energy: np.ndarray | None = None
    sab_xs: np.ndarray | None = None


def emission_curve(term, energy):
    """The term's thermal emission cross section on a descending grid: elastic below free_top
    and above sab_top, the table's at and below sab_top. Zero at free_top itself, where the target
    is at rest; the grid should hold node pairs at each top (terms.thermal_required)."""
    elastic = np.interp(energy[::-1], term.curve_energy[::-1], term.elastic[::-1])[::-1]
    out = np.where((energy < term.free_top) & (energy > term.sab_top), elastic, 0.0)
    if term.sab is not None:
        bound = np.interp(energy[::-1], term.sab_energy[::-1], term.sab_xs[::-1])[::-1]
        out = np.where(energy <= term.sab_top, bound, out)
    return out


def _incoming_points(term, coarse, j):
    """Quadrature energies and weights over hat j, weighted by hat and emission cross section."""
    grid_u = np.log(coarse[0] / coarse)
    top = coarse[0]
    breaks = [term.free_top]
    if term.sab is not None:
        breaks += [term.sab_top, *sab_midpoints(term.sab)]
    breaks = np.log(top / np.asarray([b for b in breaks if b > 0.0]))
    u_out, w_out = [], []
    for side in (j - 1, j):
        if side < 0 or side + 1 >= coarse.shape[0]:
            continue
        a, b = grid_u[side], grid_u[side + 1]
        cuts = np.unique(np.concatenate([[a, b], breaks[(breaks > a) & (breaks < b)]]))
        for lo, hi in itertools.pairwise(cuts):
            u = lo + (hi - lo) * _IN_S
            # Hat j rises over interval j - 1 and falls over interval j.
            shape = (u - a) / (b - a) if side == j - 1 else (b - u) / (b - a)
            u_out.append(u)
            w_out.append((hi - lo) * _IN_W * shape)
    u = np.concatenate(u_out)
    e = top * np.exp(-u)
    sigma = emission_curve(term, e) if e.shape[0] else e
    return e, np.concatenate(w_out) * sigma


def collision_row(term, e, coarse, du=None):
    """Hat values of one thermal collision at e, and the part off the grid."""
    if term.sab is not None and e <= term.sab_top:
        return sab_row(term.sab, e, coarse, du)
    return free_gas_row(e, term.awr, term.kT, coarse)


def _thermal_rows(term, coarse, du, hats):
    first = np.zeros(len(hats), dtype=np.int64)
    last = np.zeros(len(hats), dtype=np.int64)
    lost = np.zeros(len(hats))
    values = []
    for h, j in enumerate(hats):
        e, w = _incoming_points(term, coarse, j)
        keep = w > 0.0
        row = np.zeros(coarse.shape[0])
        off = 0.0
        for e_p, w_p in zip(e[keep], w[keep] / w[keep].sum(), strict=True):
            r, o = collision_row(term, e_p, coarse, du)
            row += w_p * r
            off += w_p * o
        nonzero = np.nonzero(row)[0]
        lost[h] = off
        if nonzero.shape[0] == 0:
            values.append(np.empty(0))
            continue
        first[h], last[h] = nonzero[0], nonzero[-1] + 1
        values.append(row[first[h] : last[h]])
    return first, last, values, lost


def _thermal_hats(term, coarse):
    n = coarse.shape[0]
    top = max(term.free_top, term.sab_top)
    # Hat j spans down to coarse[j + 1], so it emits if that is below the top.
    return np.nonzero(coarse[np.minimum(np.arange(n) + 1, n - 1)] < top)[0]


def thermal_transfer(term, coarse, du=None, workers=None):
    """One term's thermal rows on a coarse grid as SparseRows, and the part of each landing off
    the grid. Hats are split over processes as in inelastic.term_transfer."""
    du = point_spacing(coarse) if du is None else du
    hats = _thermal_hats(term, coarse)
    # _worker_count gives one process per 256 of what it is passed.
    count = _worker_count(workers, hats.shape[0] * 256 // HATS_PER_WORKER)
    if count == 1:
        results = [(hats, _thermal_rows(term, coarse, du, hats))]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        shares = [hats[w::count] for w in range(count)]
        context = multiprocessing.get_context(_start_method(workers))
        with ProcessPoolExecutor(count, mp_context=context) as pool:
            futures = [pool.submit(_thermal_rows, term, coarse, du, s) for s in shares]
            results = [(s, f.result()) for s, f in zip(shares, futures, strict=True)]
    return _assemble_rows(coarse.shape[0], results)


# Read by forked workers of _thermal_transfers, which sets it before forking: (terms, coarse, du).
_SHARED = None
SHARE_HATS = 16  # hats per task in _thermal_transfers' shared pool


def _shared_task(t, hats):
    terms, coarse, du = _SHARED
    return _thermal_rows(terms[t], coarse, du, hats)


def _thermal_transfers(terms, coarse, du=None, workers=None):
    """thermal_transfer for each term (None for none), with one pool over all their hats, dealt
    in small interleaved shares: S(alpha, beta) rows cost twenty times free gas ones. Rows don't
    depend on the split."""
    global _SHARED
    du = point_spacing(coarse) if du is None else du
    present = [t for t, term in enumerate(terms) if term is not None]
    hats = {t: _thermal_hats(terms[t], coarse) for t in present}
    count = _worker_count(workers, sum(h.shape[0] for h in hats.values()) * 256 // HATS_PER_WORKER)
    if count == 1 or _start_method(workers) != "fork":
        return [
            None if term is None else thermal_transfer(term, coarse, du, workers) for term in terms
        ]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    tasks = []
    for t in present:
        shares = max(1, hats[t].shape[0] // SHARE_HATS)
        tasks.extend((t, hats[t][w::shares]) for w in range(shares))
    _SHARED = (terms, coarse, du)
    try:
        with ProcessPoolExecutor(count, mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(_shared_task, t, share) for t, share in tasks]
            results = {t: [] for t in present}
            for (t, share), future in zip(tasks, futures, strict=True):
                results[t].append((share, future.result()))
    finally:
        _SHARED = None
    return [
        None if term is None else _assemble_rows(coarse.shape[0], results[t])
        for t, term in enumerate(terms)
    ]


# Coarse hats below the thermal tops. Free gas kernels are narrowest for the heaviest target near
# the top, about sqrt(4 kT / A E) in lethargy, and widen as E falls. A Maxwellian's F falls as
# e^-(E / kT) in its tail, and restriction and prolongation each smooth a kernel by a hat's
# width, which shifts the equilibrium by du^2 (E / kT)^2 / 6 times the scattering to leakage
# ratio. So hats are the smaller of a fifth of a kernel's width and FLUX_DU / (1 + E / kT).
HATS_PER_WIDTH = 5
FLUX_DU = 0.05
THERMAL_DU_MAX = 0.01  # below kT, where F bends on a scale of one lethargy


def kernel_width(e, kT, awr):
    """The free gas kernel's lethargy width for a target of awr at e, about sqrt(4 kT / A E)."""
    return np.sqrt(4.0 * kT / (awr * e))


def thermal_coarse(coarse, reach, kT, awr, du_min, required=()):
    """A coarse grid whose nodes below reach are spaced as the comment above says, for the heaviest
    target awr, between du_min and THERMAL_DU_MAX. Nodes above reach are kept, and required
    energies, such as reaction thresholds, are added back below it."""
    bottom = coarse[-1]
    nodes = [e for e in coarse if e > reach]
    e = min(reach, coarse[0])
    out = [e]
    while True:
        du = min(kernel_width(e, kT, awr) / HATS_PER_WIDTH, FLUX_DU / (1.0 + e / kT))
        du = np.clip(du, du_min, THERMAL_DU_MAX)
        e = e * np.exp(-du)
        if e <= bottom * np.exp(0.5 * du):
            break
        out.append(e)
    out.append(bottom)
    extra = np.asarray(required, dtype=float)
    extra = extra[(extra < reach) & (extra > bottom)]
    return np.unique(np.concatenate([nodes, out, extra]))[::-1]


def _cache_key(terms, energy, coarse, du):
    import hashlib
    from pathlib import Path

    from zerog.energy import inelastic

    digest = hashlib.sha256()
    for module in (__file__, inelastic.__file__):
        digest.update(Path(module).read_bytes())
    for array in (energy, coarse):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(repr(du).encode())
    for term in terms:
        if term is None:
            digest.update(b"none")
            continue
        digest.update(repr((term.awr, term.kT, term.free_top, term.sab_top)).encode())
        arrays = [term.curve_energy, term.elastic]
        if term.sab is not None:
            arrays += [term.sab_energy, term.sab_xs, *term.sab.__dict__.values()]
        for array in arrays:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:32]


def build_thermal(terms, energy, kernels, du=None, cache=None, workers=None):
    """kernels, an InelasticKernels, with each term's thermal emission and rows added.

    terms holds a ThermalTerm or None per term. Where a hat holds both inelastic and thermal rows,
    such as U235's MT 5, which emits neutrons down to the library's bottom, they are mixed by each
    one's hat-integrated emission cross section, holding F flat across the hat, as
    inelastic.term_transfer mixes reactions.
    """
    from zerog.energy.inelastic import InelasticKernels

    coarse = kernels.coarse
    if cache is not None:
        from pathlib import Path

        path = Path(cache) / f"thermal-{_cache_key(terms, energy, coarse, du)}.npz"
        if path.exists():
            with np.load(path) as stored:
                rows = [
                    None if f"first{t}" not in stored.files else _stored_rows(stored, t)
                    for t in range(len(terms))
                ]
        else:
            rows = _thermal_transfers(terms, coarse, du, workers)
            arrays = {}
            for t, item in enumerate(rows):
                if item is None:
                    continue
                r, lost = item
                for field in ("starts", "first", "last", "values"):
                    arrays[f"{field}{t}"] = getattr(r, field)
                arrays[f"lost{t}"] = lost
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **arrays)
    else:
        rows = _thermal_transfers(terms, coarse, du, workers)

    n_terms, n = kernels.first.shape
    if len(terms) != n_terms:
        raise ValueError(f"{len(terms)} thermal terms for kernels of {n_terms}")
    emission = kernels.emission.copy()
    starts = np.zeros((n_terms, n), dtype=np.int64)
    first = kernels.first.copy()
    last = kernels.last.copy()
    lost = kernels.lost.copy()
    values = []
    offset = 0
    for t in range(n_terms):
        both = None
        if terms[t] is not None:
            curve = emission_curve(terms[t], energy)
            # The thermal share of each hat's emission, for hats holding both kinds of row.
            share = _hat_integrals(kernels, energy, curve)
            total = share + _hat_integrals(kernels, energy, kernels.emission[t])
            both = share / np.where(total > 0.0, total, 1.0)
        for j in range(n):
            starts[t, j] = offset
            a, b = kernels.first[t, j], kernels.last[t, j]
            piece = kernels.values[kernels.starts[t, j] : kernels.starts[t, j] + b - a]
            if rows[t] is not None:
                thermal, thermal_lost = rows[t]
                c, d = thermal.first[j], thermal.last[j]
                if d > c or b > a:
                    # Weight each kind by its share of the hat's emission, even when its row is
                    # empty because everything it emits lands off the grid.
                    w = both[j]
                    if d <= c:
                        c, d = a, a
                    if b <= a:
                        a, b = c, c
                    lo, hi = min(a, c), max(b, d)
                    mixed = np.zeros(hi - lo)
                    mixed[a - lo : b - lo] += (1.0 - w) * piece
                    mixed[c - lo : d - lo] += (
                        w * thermal.values[thermal.starts[j] : thermal.starts[j] + d - c]
                    )
                    first[t, j], last[t, j] = lo, hi
                    piece = mixed
                    lost[t, j] = (1.0 - w) * lost[t, j] + w * thermal_lost[j]
            values.append(piece)
            offset += piece.shape[0]
        if terms[t] is not None:
            emission[t] += curve
    return InelasticKernels(
        coarse=coarse,
        coarse_index=kernels.coarse_index,
        hat=kernels.hat,
        emission=emission,
        starts=starts,
        first=first,
        last=last,
        values=np.concatenate(values) if values else np.empty(0),
        lost=lost,
    )


def _hat_integrals(kernels, energy, curve):
    """Each hat's integral of a lin-lin emission curve times a flat F, as restriction splits it."""
    from zerog.energy.grid import rate_weights
    from zerog.energy.inelastic import panel_hats

    panel_m, share, _, _ = panel_hats(energy, kernels.coarse)
    w = rate_weights(energy)
    rates = curve[:-1] * (w[:, 0] + w[:, 1]) + curve[1:] * (w[:, 2] + w[:, 3])
    out = np.zeros(kernels.coarse.shape[0])
    np.add.at(out, panel_m, rates * share)
    np.add.at(out, panel_m + 1, rates * (1.0 - share))
    return out


def _stored_rows(stored, t):
    rows = SparseRows(*(stored[f"{field}{t}"] for field in ("starts", "first", "last", "values")))
    return rows, stored[f"lost{t}"]
