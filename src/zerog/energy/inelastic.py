"""Expected outgoing lab energies of inelastic and (n,xn) reactions, as weighted segments.

Each sampler of OpenMC v0.16.0 is transcribed with its random numbers replaced by quadrature:
discrete choices (law by applicability, table i or i + 1) become exact weights, and each cumulative
distribution is stratified, energies in equal lethargy and cosines in equal probability, each with
its exact weight. A stratum is an interval in lab energy, since lab energy is linear in mu, and its
weight is spread over it. Earlier these were point masses, which turned the source spiky once hats
were narrower than the point spacing. The weights sum to one, and the energies follow OpenMC's
sampling, including what that adds to the data:

- ContinuousTabular::sample, KalbachMann::sample_params and CorrelatedAngleEnergy::sample_dist pick
  table i + 1 with probability r, linear in E, then scale E' between the bounding tables' first and
  last outgoing energies (unit-base). ContinuousTabular clamps E outside its tables and skips both
  for a single histogram region; the other two use get_energy_index.
- A correlated law takes the angle table of the nearer outgoing point in cumulative probability, or
  point k for a histogram table.
- inelastic_scatter converts centre-of-mass E' and mu to the lab with the target at rest, with mu
  sampled at the incoming energy.
- ReactionProduct::sample_dist picks a law by cumulative applicability, falling back to the last.

Tables here run in the library's ascending order, which is the reverse of zerog.data.scatter's.
Points go to hat functions on a coarse lethargy grid, so a kernel is a matrix from incoming nodes to
outgoing ones.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from zerog.data.interp import evaluate_tabulated
from zerog.data.scatter import (
    CorrelatedLaw,
    EvaporationSpectrum,
    KalbachMannLaw,
    LevelEnergy,
    TabularSpectrum,
    UncorrelatedLaw,
)

RULE_HISTOGRAM = 1
RULE_LIN_LIN = 2
ENERGY_RESOLUTION = 0.002  # largest lethargy between stratified outgoing energies
MAX_BIN_POINTS = 4096
COSINE_POINTS = 32  # equal-probability cosine strata per outgoing energy, in the centre of mass
PIECES_PER_HAT = 1  # outgoing-energy pieces per narrowest hat; segments carry their own width


def _ascending(spectrum):
    """A TabularSpectrum's arrays in the library's order: incoming and outgoing energies ascend."""
    lengths = np.diff(spectrum.offsets)[::-1]
    return (
        spectrum.energy[::-1].copy(),
        np.concatenate([[0], np.cumsum(lengths)]),
        spectrum.interpolation[::-1].copy(),
        spectrum.energy_out[::-1].copy(),
        spectrum.pdf[::-1].copy(),
        spectrum.cdf[::-1].copy(),
    )


@njit(cache=True)
def _bracket(energy, e, clamp):
    """Table index i and weight r of table i + 1, for ascending incoming energies.

    clamp follows ContinuousTabular, which holds the end tables; otherwise get_energy_index, which
    holds table 0 below the data. Energies above the data raise, since get_energy_index then reads
    past its tables.
    """
    n = energy.shape[0]
    if e < energy[0]:
        return 0, 0.0
    if e > energy[n - 1]:
        if clamp:
            return n - 2, 1.0
        raise ValueError("incoming energy above the law's tables")
    i = np.searchsorted(energy, e, side="right") - 1
    if i >= n - 1:
        i = n - 2
    return i, (e - energy[i]) / (energy[i + 1] - energy[i])


@njit(cache=True)
def _sample_out(e_out, p, c, rule, k, r1):
    """Outgoing energy at cumulative probability r1 inside bin k, as OpenMC inverts it."""
    e_k = e_out[k]
    p_k = p[k]
    if rule == RULE_HISTOGRAM:
        return e_k + (r1 - c[k]) / p_k if p_k > 0.0 else e_k
    e_k1 = e_out[k + 1]
    if e_k1 == e_k:
        return e_k
    frac = (p[k + 1] - p_k) / (e_k1 - e_k)
    if frac == 0.0:
        return e_k + (r1 - c[k]) / p_k
    return e_k + (math.sqrt(max(0.0, p_k * p_k + 2.0 * frac * (r1 - c[k]))) - p_k) / frac


@njit(cache=True)
def _bin_cdf(e_out, p, c, rule, k, e):
    """Cumulative probability at e inside bin k, the inverse of _sample_out."""
    x = e - e_out[k]
    if rule == RULE_HISTOGRAM:
        return c[k] + p[k] * x
    frac = (p[k + 1] - p[k]) / (e_out[k + 1] - e_out[k])
    return c[k] + p[k] * x + 0.5 * frac * x * x


@njit(cache=True)
def tabular_points(energy, offsets, rules, e_out, p, c, e, clamp, unit_base, du):
    """Pieces of a tabulated outgoing energy at incoming e.

    Returns (median, weight, table, bin, r1, lower, upper): each piece's median energy, its lower
    and upper energies, and what produced it.

    Each bin is split in equal lethargy at most du apart, each piece weighted by its exact
    cumulative probability and placed at its median. Weights sum to one; a cdf ending a little off
    one is renormalized.
    """
    i, r = _bracket(energy, e, clamp)
    tables = (i, i + 1)
    shares = (1.0 - r, r) if unit_base else (1.0, 0.0)
    first = (e_out[offsets[i]], e_out[offsets[i + 1]])
    last = (e_out[offsets[i + 1] - 1], e_out[offsets[i + 2] - 1])
    e_1 = first[0] + r * (first[1] - first[0])
    e_k = last[0] + r * (last[1] - last[0])
    counts = np.zeros(e_out.shape[0], dtype=np.int64)
    size = 0
    for side in range(2):
        if shares[side] <= 0.0:
            continue
        table = tables[side]
        for k in range(offsets[table], offsets[table + 1] - 1):
            if c[k + 1] - c[k] <= 0.0:
                continue  # never sampled: OpenMC's search passes over bins of no probability
            high = e_out[k + 1]
            low = max(e_out[k], 1e-3 * high)
            counts[k] = min(MAX_BIN_POINTS, 2 + int(math.log(high / low) / du))
            size += counts[k]
    out_e = np.empty(size)
    out_w = np.empty(size)
    out_l = np.empty(size, dtype=np.int64)
    out_k = np.empty(size, dtype=np.int64)
    out_r = np.empty(size)
    out_lo = np.empty(size)
    out_hi = np.empty(size)
    n = 0
    for side in range(2):
        share = shares[side]
        if share <= 0.0:
            continue
        table = tables[side]
        start = offsets[table]
        stop = offsets[table + 1]
        total = c[stop - 1] - c[start]
        sub_e = e_out[start:stop]
        sub_p = p[start:stop]
        sub_c = c[start:stop]
        rule = rules[table]
        for k in range(stop - start - 1):
            count = counts[start + k]
            if count == 0:
                continue
            high = sub_e[k + 1]
            low = sub_e[k]
            floor = max(low, 1e-3 * high)
            previous = sub_c[k]
            previous_edge = high
            for s in range(count):
                # Equal lethargy down from the top, with the bottom piece reaching the bin's start.
                edge = high * (floor / high) ** ((s + 1.0) / count) if s < count - 1 else low
                current = (
                    _bin_cdf(sub_e, sub_p, sub_c, rule, k, edge) if s < count - 1 else sub_c[k]
                )
                upper_c = previous if s > 0 else sub_c[k + 1]
                # Keep pieces nested even where the formula and the stored cdf part by round-off,
                # so the weights telescope to the bin's probability.
                lower_c = min(max(current, sub_c[k]), upper_c)
                previous = lower_c
                piece_high = previous_edge
                previous_edge = edge
                weight = upper_c - lower_c
                if weight <= 0.0:
                    continue
                r1 = 0.5 * (upper_c + lower_c)
                value = _sample_out(sub_e, sub_p, sub_c, rule, k, r1)
                piece_low = edge
                if unit_base:
                    scale = (e_k - e_1) / (last[side] - first[side])
                    value = e_1 + (value - first[side]) * scale
                    piece_low = e_1 + (piece_low - first[side]) * scale
                    piece_high = e_1 + (piece_high - first[side]) * scale
                out_lo[n] = piece_low
                out_hi[n] = piece_high
                out_e[n] = value
                out_w[n] = share * weight / total
                out_l[n] = table
                out_k[n] = start + k
                out_r[n] = r1
                n += 1
    return out_e[:n], out_w[:n], out_l[:n], out_k[:n], out_r[:n], out_lo[:n], out_hi[:n]


@njit(cache=True)
def _tabular_cosine(mu, p, c, rule, r1):
    """OpenMC's Tabular::sample_unbiased of a cosine table at cumulative probability r1."""
    n = mu.shape[0]
    i = 0
    while i < n - 2 and c[i + 1] <= r1:
        i += 1
    if rule == RULE_HISTOGRAM:
        return mu[i] + (r1 - c[i]) / p[i] if p[i] > 0.0 else mu[i]
    m = (p[i + 1] - p[i]) / (mu[i + 1] - mu[i])
    if m == 0.0:
        return mu[i] + (r1 - c[i]) / p[i]
    return mu[i] + (math.sqrt(max(0.0, p[i] * p[i] + 2.0 * m * (r1 - c[i]))) - p[i]) / m


@njit(cache=True)
def table_cosines(mu, p, c, rule, n_points):
    """Bounds of equal-probability cosine strata of one table: n_points + 1 values, ascending."""
    out = np.empty(n_points + 1)
    out[0] = mu[0]
    out[n_points] = mu[mu.shape[0] - 1]
    for m in range(1, n_points):
        value = _tabular_cosine(mu, p, c, rule, m / n_points)
        out[m] = min(1.0, max(-1.0, value))
    return out


@njit(cache=True)
def kalbach_cosines(slope, precompound, n_points):
    """Cosine strata of a Kalbach-Mann distribution: (lower bounds, upper bounds, weights).

    Each of its two terms is split into n_points strata of equal probability.
    """
    low = np.empty(2 * n_points)
    high = np.empty(2 * n_points)
    weights = np.empty(2 * n_points)
    for m in range(n_points):
        for side in range(2):
            xi = (m + side) / n_points
            t = (2.0 * xi - 1.0) * math.sinh(slope)
            value = math.log(t + math.sqrt(t * t + 1.0)) / slope
            if side == 0:
                low[m] = value
            else:
                high[m] = value
            value = math.log(xi * math.exp(slope) + (1.0 - xi) * math.exp(-slope)) / slope
            if side == 0:
                low[n_points + m] = value
            else:
                high[n_points + m] = value
        weights[m] = (1.0 - precompound) / n_points
        weights[n_points + m] = precompound / n_points
    return np.maximum(low, -1.0), np.minimum(high, 1.0), weights


@njit(cache=True)
def _lab(e_in, e_cm, mu, awr):
    return e_cm + (e_in + 2.0 * mu * (awr + 1.0) * math.sqrt(e_in * e_cm)) / ((awr + 1.0) ** 2)


@njit(cache=True)
def _kalbach_segments(
    e, e_cm, w, k, r1, point_rule, precompound, slope, e_asc, p_asc, c_asc, n_cosine, awr
):
    """Lab segments of a Kalbach-Mann law's centre-of-mass pieces, as law_points makes them."""
    size = e_cm.shape[0] * 2 * n_cosine
    lo = np.empty(size)
    hi = np.empty(size)
    out_w = np.empty(size)
    n = 0
    for i in range(e_cm.shape[0]):
        j = k[i]
        if point_rule[j] == RULE_HISTOGRAM:
            km_r = precompound[j]
            km_a = slope[j]
        else:
            # OpenMC interpolates r and a with the outgoing energy before unit-base scaling.
            raw = _sample_out(e_asc, p_asc, c_asc, RULE_LIN_LIN, j, r1[i])
            t = (raw - e_asc[j]) / (e_asc[j + 1] - e_asc[j])
            km_r = precompound[j] + t * (precompound[j + 1] - precompound[j])
            km_a = slope[j] + t * (slope[j + 1] - slope[j])
        low, high, cosine_w = kalbach_cosines(km_a, km_r, n_cosine)
        for m in range(low.shape[0]):
            lo[n] = _lab(e, e_cm[i], low[m], awr)
            hi[n] = _lab(e, e_cm[i], high[m], awr)
            out_w[n] = w[i] * cosine_w[m]
            n += 1
    return lo[:n], hi[:n], out_w[:n]


@njit(cache=True)
def _correlated_segments(
    e,
    e_cm,
    w,
    k,
    r1,
    point_rule,
    c_asc,
    mu_start,
    mu_stop,
    mu_rule,
    mu,
    mu_pdf,
    mu_cdf,
    n_cosine,
    awr,
):
    """Lab segments of a correlated law's centre-of-mass pieces, as law_points makes them."""
    size = e_cm.shape[0] * n_cosine
    lo = np.empty(size)
    hi = np.empty(size)
    out_w = np.empty(size)
    n = 0
    for i in range(e_cm.shape[0]):
        j = k[i]
        near = point_rule[j] == RULE_HISTOGRAM or r1[i] - c_asc[j] < c_asc[j + 1] - r1[i]
        point = j if near else j + 1
        a = mu_start[point]
        b = mu_stop[point]
        bounds = table_cosines(mu[a:b], mu_pdf[a:b], mu_cdf[a:b], mu_rule[point], n_cosine)
        for m in range(n_cosine):
            lo[n] = _lab(e, e_cm[i], bounds[m], awr)
            hi[n] = _lab(e, e_cm[i], bounds[m + 1], awr)
            out_w[n] = w[i] / n_cosine
            n += 1
    return lo[:n], hi[:n], out_w[:n]


def _point_rules(tables):
    """Each flat outgoing point's table rule, for ascending tables from _ascending."""
    _, offsets, rules, e_out, _, _ = tables
    return np.repeat(rules, np.diff(offsets)).astype(np.int64)[: e_out.shape[0]]


def to_lab(e_in, e_cm, mu, awr):
    """Lab energy of a centre-of-mass emission, as in inelastic_scatter. Takes arrays."""
    return e_cm + (e_in + 2.0 * mu * (awr + 1.0) * np.sqrt(e_in * e_cm)) / ((awr + 1.0) ** 2)


def _angle_strata(angle, e, n_points):
    """Cosine strata from AngleTables at e, as AngleDistribution::sample mixes them.

    Returns (lower bounds, upper bounds, weights).
    """
    energy = angle.energy[::-1]
    i, r = _bracket(energy, e, False) if e <= energy[-1] else (energy.shape[0] - 1, 0.0)
    lows, highs, weights = [], [], []
    for table, share in ((i, 1.0 - r), (i + 1, r)):
        if share <= 0.0:
            continue
        stored = angle.energy.shape[0] - 1 - table
        span = slice(angle.offsets[stored], angle.offsets[stored + 1])
        bounds = table_cosines(
            angle.mu[span], angle.pdf[span], angle.cdf[span], angle.interpolation[stored], n_points
        )
        lows.append(bounds[:-1])
        highs.append(bounds[1:])
        weights.append(np.full(n_points, share / n_points))
    return np.concatenate(lows), np.concatenate(highs), np.concatenate(weights)


def _uniform_strata(n_points):
    bounds = np.linspace(-1.0, 1.0, n_points + 1)
    return bounds[:-1], bounds[1:], np.full(n_points, 1.0 / n_points)


def _evaporation_points(law, e, du):
    """E' exp(-E' / theta) on 0 <= E' <= e - u, in equal-lethargy pieces with exact weights.

    Returns (lower, upper, weight) per piece.
    """
    theta = evaluate_tabulated(
        law.energy[::-1].copy(),
        law.theta[::-1].copy(),
        law.breakpoints_ascending,
        law.interpolation_ascending,
        np.array([e]),
    )[0]
    y = (e - law.u) / theta
    if y <= 0.0:
        return np.empty(0), np.empty(0), np.empty(0)
    # Below 1e-6 of the top the spectrum holds about 1e-12 of its neutrons, so one piece covers it.
    count = 2 + int(math.log(1e6) / du)
    edges = np.concatenate([y * np.exp(-np.linspace(0.0, math.log(1e6), count)), [0.0]])
    cdf = 1.0 - (1.0 + edges) * np.exp(-edges)
    weights = (cdf[:-1] - cdf[1:]) / cdf[0]
    keep = weights > 0.0
    return (edges[1:] * theta)[keep], (edges[:-1] * theta)[keep], weights[keep]


def _lab_segments(e, e_cm, w, low, high, cosine_w, awr):
    """Lab segments of centre-of-mass emissions at energies e_cm, over cosine strata.

    Lab energy is linear in mu, so a stratum maps to an exact lab interval.
    """
    lo = to_lab(e, e_cm[:, None], low[None, :], awr)
    hi = to_lab(e, e_cm[:, None], high[None, :], awr)
    return lo.ravel(), hi.ravel(), (w[:, None] * cosine_w[None, :]).ravel()


def law_points(law, e, awr, center_of_mass, du=ENERGY_RESOLUTION, n_cosine=COSINE_POINTS):
    """Lab outgoing energy segments of one law at incoming e: (lower, upper, weight).

    Each segment carries its probability, spread over its interval. Weights sum to one.
    """
    if isinstance(law, UncorrelatedLaw):
        energy = law.energy
        if isinstance(energy, LevelEnergy):
            if e <= energy.threshold:
                return np.empty(0), np.empty(0), np.empty(0)
            e_cm = np.array([energy.mass_ratio * (e - energy.threshold)])
            lo_cm = hi_cm = e_cm
            w = np.ones(1)
        elif isinstance(energy, TabularSpectrum):
            tables = _ascending(energy)
            histogram = (
                energy.interpolation_ascending.shape[0] == 1
                and energy.interpolation_ascending[0] == RULE_HISTOGRAM
            )
            e_cm, w, _, _, _, lo_cm, hi_cm = tabular_points(*tables, e, True, not histogram, du)
        elif isinstance(energy, EvaporationSpectrum):
            lo_cm, hi_cm, w = _evaporation_points(energy, e, du)
            e_cm = 0.5 * (lo_cm + hi_cm)
        else:
            raise NotImplementedError(f"scattering energy law {type(energy).__name__}")
        if not center_of_mass:
            return lo_cm, hi_cm, w
        if law.angle is None:
            low, high, cosine_w = _uniform_strata(n_cosine)
        else:
            low, high, cosine_w = _angle_strata(law.angle, e, n_cosine)
        return _lab_segments(e, e_cm, w, low, high, cosine_w, awr)

    if isinstance(law, KalbachMannLaw | CorrelatedLaw):
        tables = _ascending(law.spectrum)
        e_cm, w, _, k, r1, lo_cm, hi_cm = tabular_points(*tables, e, False, True, du)
        if not center_of_mass:
            return lo_cm, hi_cm, w
        rules = _point_rules(tables)
        if isinstance(law, KalbachMannLaw):
            return _kalbach_segments(
                e,
                e_cm,
                w,
                k,
                r1,
                rules,
                law.precompound[::-1].copy(),
                law.slope[::-1].copy(),
                tables[3],
                tables[4],
                tables[5],
                n_cosine,
                awr,
            )
        return _correlated_segments(
            e,
            e_cm,
            w,
            k,
            r1,
            rules,
            tables[5],
            law.mu_start[::-1].copy(),
            law.mu_stop[::-1].copy(),
            law.mu_rule[::-1].copy(),
            law.mu,
            law.mu_pdf,
            law.mu_cdf,
            n_cosine,
            awr,
        )

    raise NotImplementedError(f"scattering law {type(law).__name__}")


def applicability_shares(reaction, e):
    """Probability of each law of a reaction at e, as ReactionProduct::sample_dist picks them."""
    n = len(reaction.laws)
    if n == 1:
        return np.ones(1)
    shares = np.zeros(n)
    remaining = 1.0
    for i, applicability in enumerate(reaction.applicability):
        p = evaluate_tabulated(
            applicability.energy[::-1].copy(),
            applicability.probability[::-1].copy(),
            applicability.breakpoints_ascending,
            applicability.interpolation_ascending,
            np.array([e]),
        )[0]
        take = min(max(p, 0.0), remaining)
        shares[i] = take
        remaining -= take
    shares[-1] += remaining
    return shares


def reaction_points(reaction, e, awr, du=ENERGY_RESOLUTION, n_cosine=COSINE_POINTS):
    """Lab outgoing segments of a ReactionLaw at e: (lower, upper, weight), summing to one."""
    parts = []
    for share, law in zip(applicability_shares(reaction, e), reaction.laws, strict=True):
        if share <= 0.0:
            continue
        lo, hi, w = law_points(law, e, awr, reaction.center_of_mass, du, n_cosine)
        parts.append((lo, hi, share * w))
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(3))


# Transfer on a coarse lethargy grid. A term's emission from incoming hat j, the integral of
# hat_j sigma_em F du, goes to outgoing hats by row j of its matrix, and outgoing hat values are
# interpolated onto the fine grid. The coarse grid is a subset of the fine one and spans it, so hats
# sum to one everywhere and the transfer conserves neutrons exactly.


def coarse_grid(energy, du, required=()):
    """A coarse lethargy grid spanning a fine descending grid, of width du, with required energies.

    Pass its nodes to the fine grid as required energies, so the coarse grid stays a subset.
    """
    top, bottom = energy[0], energy[-1]
    n = max(1, int(np.ceil(np.log(top / bottom) / du)))
    # Interior nodes only: the computed ends miss top and bottom by round-off, which would leave a
    # second node a hair away from each and an interval of zero width.
    base = top * np.exp(-np.log(top / bottom) * np.arange(1, n) / n)
    extra = np.asarray(required, dtype=float)
    extra = extra[(extra < top) & (extra > bottom)]
    return np.unique(np.concatenate([base, extra, [top, bottom]]))[::-1]


def hat_weights(coarse):
    """The integral of each hat over lethargy."""
    widths = np.log(coarse[:-1] / coarse[1:])
    out = np.zeros(coarse.shape[0])
    out[:-1] += 0.5 * widths
    out[1:] += 0.5 * widths
    return out


def deposit(lower, upper, weights, coarse):
    """Hat values of lab segments on a coarse grid, and the weight landing outside it.

    Each segment's weight is spread uniformly in lethargy over its interval, and each hat takes
    its exact integral over the part it covers. A segment of zero width is a point.
    """
    return _deposit_segments(
        np.log(coarse[0] / np.maximum(upper, 1e-300)),
        np.log(coarse[0] / np.maximum(lower, 1e-300)),
        weights,
        np.log(coarse[0] / coarse),
    )


def bin_segments(lower, upper, weights, edges):
    """Segment weight in each bin of ascending energy edges, spread uniformly in lethargy."""
    u = np.log(edges[-1] / edges[::-1])
    out = np.zeros(edges.shape[0] - 1)
    a = np.log(edges[-1] / np.maximum(upper, 1e-300))
    b = np.log(edges[-1] / np.maximum(lower, 1e-300))
    point = b - a <= 1e-14
    for i in range(u.shape[0] - 1):
        overlap = np.clip(np.minimum(b, u[i + 1]) - np.maximum(a, u[i]), 0.0, None)
        spread = np.where(point, 0.0, overlap / np.where(point, 1.0, b - a))
        inside = point & (a >= u[i]) & (a < u[i + 1])
        out[i] = np.sum(weights * np.where(inside, 1.0, spread))
    return out[::-1]


@njit(cache=True)
def _partial(row, density, grid_u, m, low, high):
    """Add a constant density over [low, high] inside interval m to its two hats."""
    width = grid_u[m + 1] - grid_u[m]
    t0 = (low - grid_u[m]) / width
    t1 = (high - grid_u[m]) / width
    upper_part = 0.5 * (t1 * t1 - t0 * t0)
    row[m] += density * width * ((t1 - t0) - upper_part)
    row[m + 1] += density * width * upper_part


@njit(cache=True)
def _deposit_segments(start, stop, weights, grid_u):
    """Hat values of segments uniform in lethargy over [start, stop], and the weight off the grid.

    The two end intervals of a segment are added directly. The whole intervals between take its
    density through a difference array, spread once at the end, so a segment costs O(log n)
    however many hats it spans.
    """
    n = grid_u.shape[0]
    row = np.zeros(n)
    covered = np.zeros(n)  # density over whole intervals, as differences
    lost = 0.0
    for s in range(start.shape[0]):
        a = min(start[s], stop[s])
        b = max(start[s], stop[s])
        w = weights[s]
        if b - a <= 1e-14 * max(1.0, abs(b)):
            # A point: split between the two hats it lies between.
            if a < grid_u[0] or a > grid_u[n - 1]:
                lost += w
                continue
            m = min(max(np.searchsorted(grid_u, a, side="right") - 1, 0), n - 2)
            theta = (a - grid_u[m]) / (grid_u[m + 1] - grid_u[m])
            row[m] += w * (1.0 - theta)
            row[m + 1] += w * theta
            continue
        density = w / (b - a)
        inside_a = max(a, grid_u[0])
        inside_b = min(b, grid_u[n - 1])
        if inside_b <= inside_a:
            lost += w
            continue
        lost += density * ((b - a) - (inside_b - inside_a))
        m_a = min(max(np.searchsorted(grid_u, inside_a, side="right") - 1, 0), n - 2)
        m_b = min(max(np.searchsorted(grid_u, inside_b, side="left") - 1, 0), n - 2)
        if m_a == m_b:
            _partial(row, density, grid_u, m_a, inside_a, inside_b)
            continue
        _partial(row, density, grid_u, m_a, inside_a, grid_u[m_a + 1])
        _partial(row, density, grid_u, m_b, grid_u[m_b], inside_b)
        covered[m_a + 1] += density
        covered[m_b] -= density
    level = 0.0
    for m in range(n - 1):
        level += covered[m]
        if level != 0.0:
            half = 0.5 * level * (grid_u[m + 1] - grid_u[m])
            row[m] += half
            row[m + 1] += half
    return row, lost


@dataclass(frozen=True)
class InelasticKernels:
    """Every term's inelastic transfer, flat. Rows are compressed by their first and last column.

    Term t's row j holds values[starts[t, j]:starts[t, j] + (last[t, j] - first[t, j])] for columns
    first[t, j]:last[t, j], per neutron emitted from hat j; lost[t, j] is the part landing off the
    grid. emission (terms, fine energies) is the sum over reactions of yield * cross section.
    coarse_index gives each coarse node's fine node.
    """

    coarse: np.ndarray
    coarse_index: np.ndarray
    hat: np.ndarray
    emission: np.ndarray
    starts: np.ndarray
    first: np.ndarray
    last: np.ndarray
    values: np.ndarray
    lost: np.ndarray


def _hat_moments(coarse, energy, curve):
    """Each hat's integral of a lin-lin curve over lethargy, and its centroid energy.

    The integrand is sampled at the curve's own nodes inside the hat's support and at the hat's
    nodes, then integrated by trapezoids, which follows resonances in the curve.
    """
    u = np.log(coarse[0] / energy)
    grid_u = np.log(coarse[0] / coarse)
    integral = np.zeros(coarse.shape[0])
    centroid = coarse.copy()
    for j in range(coarse.shape[0]):
        low = grid_u[j - 1] if j > 0 else grid_u[0]
        high = grid_u[j + 1] if j + 1 < coarse.shape[0] else grid_u[-1]
        inside = u[(u > low) & (u < high)]
        points = np.unique(np.concatenate([inside, [low, grid_u[j], high]]))
        values = np.interp(points, u, curve, left=0.0, right=0.0)
        shape = np.where(
            points <= grid_u[j],
            (points - low) / (grid_u[j] - low) if j > 0 else 1.0,
            (high - points) / (high - grid_u[j]) if j + 1 < coarse.shape[0] else 1.0,
        )
        weight = shape * values
        integral[j] = np.trapezoid(weight, points)
        if integral[j] > 0.0:
            centroid[j] = coarse[0] * np.exp(-np.trapezoid(weight * points, points) / integral[j])
    return integral, centroid


def point_spacing(coarse):
    """Lethargy between outgoing-energy pieces for a coarse grid: a fraction of its median hat.

    Pieces are spread over their intervals, so this sets how well a piece's uniform spread follows
    the spectrum inside it, not whether hats are hit at all. A sliver hat between two close
    thresholds (1e-6 wide at 6.5 MeV for U238 and O16) takes its exact share of each piece, so it
    must not set the spacing: the narrowest hat made heavy-fast's kernels 40 times slower.
    """
    return min(ENERGY_RESOLUTION, np.median(np.log(coarse[:-1] / coarse[1:])) / PIECES_PER_HAT)


def term_transfer(name, temperature, awr, coarse, du=None, n_cosine=COSINE_POINTS):
    """One term's rows on a coarse grid, and its emission curve as (energies, values) for lin-lin.

    Row j mixes reactions by their hat-integrated yield * cross section, each evaluated at its
    centroid in the hat. So every hat that emits has a full row, even one whose node lies below a
    threshold, and reaction shares are held fixed across a hat: an O(du) approximation.
    """
    from zerog.data.neutron import read_reactions, read_scattering_yields
    from zerog.data.scatter import read_scattering_laws

    du = point_spacing(coarse) if du is None else du
    curves = read_reactions(name, temperature)
    laws = [r for r in read_scattering_laws(name) if r.mt != 2]
    rows = np.zeros((coarse.shape[0], coarse.shape[0]))
    lost = np.zeros(coarse.shape[0])
    mts = np.array([law.mt for law in laws], dtype=np.int64)
    if mts.shape[0] == 0:
        return rows, lost, curves.energy, np.zeros(curves.energy.shape[0])
    index = np.array([list(curves.mt).index(mt) for mt in mts])
    produced = curves.xs[index] * read_scattering_yields(name, mts, curves.energy)
    emission = produced.sum(axis=0)
    moments = [_hat_moments(coarse, curves.energy, curve) for curve in produced]
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
                # A hat straddling a threshold can have its centroid below it, so take the reaction
                # at the hat's top instead of dropping its share.
                lo, hi, w = reaction_points(law, coarse[j - 1], awr, du, n_cosine)
            if w.shape[0] == 0:
                lost[j] += share
                continue
            row, off = deposit(lo, hi, w * share, coarse)
            rows[j] += row
            lost[j] += off
    return rows, lost, curves.energy, emission


def _cache_key(terms, energy, coarse, du, n_cosine):
    """A key that changes with anything the kernels depend on, data readers' code included."""
    import hashlib
    from pathlib import Path

    from zerog.data import interp, neutron, scatter
    from zerog.data.library import nuclide_path

    digest = hashlib.sha256()
    for module in (__file__, interp.__file__, neutron.__file__, scatter.__file__):
        digest.update(Path(module).read_bytes())
    digest.update(np.ascontiguousarray(energy).tobytes())
    digest.update(np.ascontiguousarray(coarse).tobytes())
    digest.update(repr((du, n_cosine)).encode())
    for term in terms:
        digest.update(repr(term).encode())
        if term is not None:
            stat = Path(nuclide_path(term[0])).stat()
            digest.update(repr((stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()[:32]


def build_inelastic(terms, energy, coarse, du=None, n_cosine=COSINE_POINTS, cache=None):
    """InelasticKernels for terms given as (name, temperature label, awr), or None for no reactions.

    coarse must be a subset of energy and span it. With cache, a directory, kernels are stored
    there and reused by any later call with the same grid, terms, settings, data files and code.
    """
    if cache is not None:
        from pathlib import Path

        path = Path(cache) / f"inelastic-{_cache_key(terms, energy, coarse, du, n_cosine)}.npz"
        if path.exists():
            with np.load(path) as stored:
                return InelasticKernels(**{name: stored[name] for name in stored.files})
        kernels = build_inelastic(terms, energy, coarse, du, n_cosine)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **kernels.__dict__)
        return kernels
    index = np.searchsorted(-energy, -coarse)
    if not np.array_equal(energy[np.minimum(index, energy.shape[0] - 1)], coarse):
        raise ValueError("the coarse grid must be a subset of the fine grid")
    if coarse[0] != energy[0] or coarse[-1] != energy[-1]:
        raise ValueError("the coarse grid must span the fine grid")
    n_terms = len(terms)
    n = coarse.shape[0]
    emission = np.zeros((n_terms, energy.shape[0]))
    starts = np.zeros((n_terms, n), dtype=np.int64)
    first = np.zeros((n_terms, n), dtype=np.int64)
    last = np.zeros((n_terms, n), dtype=np.int64)
    lost = np.zeros((n_terms, n))
    values = []
    offset = 0
    for t, term in enumerate(terms):
        if term is None:
            rows = np.zeros((n, n))
        else:
            name, label, awr = term
            rows, lost[t], table_energy, curve = term_transfer(
                name, label, awr, coarse, du, n_cosine
            )
            emission[t] = np.interp(energy[::-1], table_energy[::-1], curve[::-1])[::-1]
        for j in range(n):
            nonzero = np.nonzero(rows[j])[0]
            starts[t, j] = offset
            if nonzero.shape[0] == 0:
                continue
            first[t, j] = nonzero[0]
            last[t, j] = nonzero[-1] + 1
            values.append(rows[j, first[t, j] : last[t, j]])
            offset += last[t, j] - first[t, j]
    return InelasticKernels(
        coarse=coarse,
        coarse_index=index,
        hat=hat_weights(coarse),
        emission=emission,
        starts=starts,
        first=first,
        last=last,
        values=np.concatenate(values) if values else np.empty(0),
        lost=lost,
    )


def no_inelastic(energy, n_terms):
    """Empty kernels, for terms without inelastic reactions."""
    coarse = np.array([energy[0], energy[-1]])
    return InelasticKernels(
        coarse=coarse,
        coarse_index=np.array([0, energy.shape[0] - 1]),
        hat=hat_weights(coarse),
        emission=np.zeros((n_terms, energy.shape[0])),
        starts=np.zeros((n_terms, 2), dtype=np.int64),
        first=np.zeros((n_terms, 2), dtype=np.int64),
        last=np.zeros((n_terms, 2), dtype=np.int64),
        values=np.empty(0),
        lost=np.zeros((n_terms, 2)),
    )


def panel_hats(energy, coarse):
    """Each fine panel's coarse interval, and the mean of that interval's upper hat over the panel.

    Also each fine node's interval and its position in it, for interpolating hat values.
    """
    u = np.log(energy[0] / energy)
    grid_u = np.log(energy[0] / coarse)
    node_m = np.clip(np.searchsorted(grid_u, u, side="right") - 1, 0, coarse.shape[0] - 2)
    node_theta = (u - grid_u[node_m]) / (grid_u[node_m + 1] - grid_u[node_m])
    panel_m = node_m[:-1]
    start = (u[:-1] - grid_u[panel_m]) / (grid_u[panel_m + 1] - grid_u[panel_m])
    end = (u[1:] - grid_u[panel_m]) / (grid_u[panel_m + 1] - grid_u[panel_m])
    return panel_m, 1.0 - 0.5 * (start + end), node_m, node_theta


def restrict(kernels, energy, flux, rate_weights):
    """Emission from each incoming hat, (terms, coarse nodes), as the march accumulates it.

    Each panel's exact emission is split between the two hats by their mean over it, which keeps the
    sum exact.
    """
    panel_m, share, _, _ = panel_hats(energy, kernels.coarse)
    out = np.zeros(kernels.emission.shape[:1] + kernels.coarse.shape)
    for t in range(kernels.emission.shape[0]):
        sigma = kernels.emission[t]
        rates = sigma[:-1] * (
            rate_weights[:, 0] * flux[:-1] + rate_weights[:, 1] * flux[1:]
        ) + sigma[1:] * (rate_weights[:, 2] * flux[:-1] + rate_weights[:, 3] * flux[1:])
        np.add.at(out[t], panel_m, rates * share)
        np.add.at(out[t], panel_m + 1, rates * (1.0 - share))
    return out


def point_emission_coarse(name, temperature, awr, e, coarse, du=None, n_cosine=COSINE_POINTS):
    """Hat values of inelastic emission from one collision at e, per collision, and the emission
    cross section there, yield * barns summed over reactions.

    Pass the hat values through interpolate_hats for a fine-grid source per unit lethargy.
    """
    from zerog.data.neutron import read_reactions, read_scattering_yields
    from zerog.data.scatter import read_scattering_laws

    du = point_spacing(coarse) if du is None else du
    curves = read_reactions(name, temperature)
    laws = [r for r in read_scattering_laws(name) if r.mt != 2]
    row = np.zeros(coarse.shape[0])
    if not laws:
        return row, 0.0
    mts = np.array([law.mt for law in laws], dtype=np.int64)
    yields = read_scattering_yields(name, mts, np.array([e]))[:, 0]
    total = 0.0
    for law, y in zip(laws, yields, strict=True):
        r = list(curves.mt).index(law.mt)
        xs = float(np.interp(e, curves.energy[::-1], curves.xs[r, ::-1]))
        if xs * y <= 0.0:
            continue
        lo, hi, w = reaction_points(law, e, awr, du, n_cosine)
        if w.shape[0] == 0:
            continue
        row += deposit(lo, hi, w * xs * y, coarse)[0]
        total += xs * y
    return (row / total if total > 0.0 else row), total


def interpolate_hats(ops, values):
    """A fine-grid curve per unit lethargy from hat values, as the march spreads coarse sources."""
    hat = ops.inelastic.hat
    m = ops.node_m
    theta = ops.node_theta
    return (1.0 - theta) * values[m] / hat[m] + theta * values[m + 1] / hat[m + 1]
