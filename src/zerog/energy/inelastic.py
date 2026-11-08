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
def _sample_out(e_k, e_k1, p_k, p_k1, c_k, rule, r1):
    """Outgoing energy at cumulative probability r1 inside a bin from e_k to e_k1, with pdf p_k
    and p_k1 at its ends and cdf c_k at its start, as OpenMC inverts it.

    Scalars, not arrays, so a call per piece costs no reference counting.
    """
    if rule == RULE_HISTOGRAM:
        return e_k + (r1 - c_k) / p_k if p_k > 0.0 else e_k
    if e_k1 == e_k:
        return e_k
    frac = (p_k1 - p_k) / (e_k1 - e_k)
    if frac == 0.0:
        return e_k + (r1 - c_k) / p_k
    return e_k + (math.sqrt(max(0.0, p_k * p_k + 2.0 * frac * (r1 - c_k))) - p_k) / frac


@njit(cache=True)
def _bin_cdf(e_k, e_k1, p_k, p_k1, c_k, rule, e):
    """Cumulative probability at e inside a bin, the inverse of _sample_out."""
    x = e - e_k
    if rule == RULE_HISTOGRAM:
        return c_k + p_k * x
    frac = (p_k1 - p_k) / (e_k1 - e_k)
    return c_k + p_k * x + 0.5 * frac * x * x


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
            bin_p = sub_p[k]
            bin_p1 = sub_p[k + 1]
            bin_c = sub_c[k]
            floor = max(low, 1e-3 * high)
            previous = sub_c[k]
            previous_edge = high
            for s in range(count):
                # Equal lethargy down from the top, with the bottom piece reaching the bin's start.
                edge = high * (floor / high) ** ((s + 1.0) / count) if s < count - 1 else low
                current = (
                    _bin_cdf(low, high, bin_p, bin_p1, bin_c, rule, edge)
                    if s < count - 1
                    else bin_c
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
                value = _sample_out(low, high, bin_p, bin_p1, bin_c, rule, r1)
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
            raw = _sample_out(
                e_asc[j], e_asc[j + 1], p_asc[j], p_asc[j + 1], c_asc[j], RULE_LIN_LIN, r1[i]
            )
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
def _point_cosines(mu_start, mu_stop, mu_rule, mu, mu_pdf, mu_cdf, n_cosine):
    """table_cosines of every outgoing point's mu table of a correlated law, as (points, n + 1).

    They depend only on the point, so a law computes them once instead of once per piece.
    """
    out = np.empty((mu_start.shape[0], n_cosine + 1))
    for point in range(mu_start.shape[0]):
        a = mu_start[point]
        b = mu_stop[point]
        out[point] = table_cosines(mu[a:b], mu_pdf[a:b], mu_cdf[a:b], mu_rule[point], n_cosine)
    return out


@njit(cache=True)
def _correlated_segments(e, e_cm, w, k, r1, point_rule, c_asc, cosines, n_cosine, awr):
    """Lab segments of a correlated law's centre-of-mass pieces, as law_points makes them.

    cosines holds each ascending point's strata bounds, from _point_cosines.
    """
    size = e_cm.shape[0] * n_cosine
    lo = np.empty(size)
    hi = np.empty(size)
    out_w = np.empty(size)
    n = 0
    for i in range(e_cm.shape[0]):
        j = k[i]
        near = point_rule[j] == RULE_HISTOGRAM or r1[i] - c_asc[j] < c_asc[j + 1] - r1[i]
        point = j if near else j + 1
        for m in range(n_cosine):
            lo[n] = _lab(e, e_cm[i], cosines[point, m], awr)
            hi[n] = _lab(e, e_cm[i], cosines[point, m + 1], awr)
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


def _table_strata(angle, n_points):
    """table_cosines of each of an AngleTables' tables, in its stored order, as (tables, n + 1)."""
    out = np.empty((angle.energy.shape[0], n_points + 1))
    for stored in range(angle.energy.shape[0]):
        span = slice(angle.offsets[stored], angle.offsets[stored + 1])
        out[stored] = table_cosines(
            angle.mu[span], angle.pdf[span], angle.cdf[span], angle.interpolation[stored], n_points
        )
    return out


def _angle_strata(angle, e, n_points, strata=None):
    """Cosine strata from AngleTables at e, as AngleDistribution::sample mixes them.

    Returns (lower bounds, upper bounds, weights). strata, from _table_strata, saves recomputing
    each table's bounds.
    """
    if strata is None:
        strata = _table_strata(angle, n_points)
    energy = angle.energy[::-1]
    i, r = _bracket(energy, e, False) if e <= energy[-1] else (energy.shape[0] - 1, 0.0)
    lows, highs, weights = [], [], []
    for table, share in ((i, 1.0 - r), (i + 1, r)):
        if share <= 0.0:
            continue
        bounds = strata[angle.energy.shape[0] - 1 - table]
        lows.append(bounds[:-1])
        highs.append(bounds[1:])
        weights.append(np.full(n_points, share / n_points))
    return np.concatenate(lows), np.concatenate(highs), np.concatenate(weights)


def _uniform_strata(n_points):
    bounds = np.linspace(-1.0, 1.0, n_points + 1)
    return bounds[:-1], bounds[1:], np.full(n_points, 1.0 / n_points)


def _evaporation_points(law, e, du, ascending=None):
    """E' exp(-E' / theta) on 0 <= E' <= e - u, in equal-lethargy pieces with exact weights.

    Returns (lower, upper, weight) per piece. ascending, from _ascending_tab1, saves the flip.
    """
    if ascending is None:
        ascending = _ascending_tab1(law.energy, law.theta, law)
    theta = evaluate_tabulated(*ascending, np.array([e]))[0]
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


def _ascending_tab1(energy, values, regions):
    """evaluate_tabulated's first four arguments for a descending TAB1 with ascending regions."""
    return (
        energy[::-1].copy(),
        values[::-1].copy(),
        regions.breakpoints_ascending,
        regions.interpolation_ascending,
    )


def _lab_segments(e, e_cm, w, low, high, cosine_w, awr):
    """Lab segments of centre-of-mass emissions at energies e_cm, over cosine strata.

    Lab energy is linear in mu, so a stratum maps to an exact lab interval.
    """
    lo = to_lab(e, e_cm[:, None], low[None, :], awr)
    hi = to_lab(e, e_cm[:, None], high[None, :], awr)
    return lo.ravel(), hi.ravel(), (w[:, None] * cosine_w[None, :]).ravel()


@dataclass(frozen=True)
class _PreparedLaw:
    """A law's tables in the order its samplers read them, flipped once instead of per call.

    kind is "level", "tabular", "evaporation", "kalbach", "correlated", or "unsupported", which
    raises error only when the law is sampled, as before preparation.
    """

    kind: str
    law: object
    n_cosine: int
    error: str = ""
    tables: tuple = ()
    histogram: bool = False
    rules: np.ndarray | None = None
    precompound: np.ndarray | None = None
    slope: np.ndarray | None = None
    cosines: np.ndarray | None = None  # correlated: strata bounds per ascending point
    theta: tuple = ()  # evaporation: evaluate_tabulated's arguments
    strata: np.ndarray | None = None  # uncorrelated: strata bounds per stored angle table


def _prepare_law(law, n_cosine):
    """Flip and precompute what law_points needs of a law, for any number of incoming energies."""
    if isinstance(law, UncorrelatedLaw):
        energy = law.energy
        strata = None if law.angle is None else _table_strata(law.angle, n_cosine)
        if isinstance(energy, LevelEnergy):
            return _PreparedLaw("level", law, n_cosine, strata=strata)
        if isinstance(energy, TabularSpectrum):
            histogram = (
                energy.interpolation_ascending.shape[0] == 1
                and energy.interpolation_ascending[0] == RULE_HISTOGRAM
            )
            return _PreparedLaw(
                "tabular",
                law,
                n_cosine,
                tables=_ascending(energy),
                histogram=histogram,
                strata=strata,
            )
        if isinstance(energy, EvaporationSpectrum):
            return _PreparedLaw(
                "evaporation",
                law,
                n_cosine,
                theta=_ascending_tab1(energy.energy, energy.theta, energy),
                strata=strata,
            )
        return _PreparedLaw(
            "unsupported", law, n_cosine, error=f"scattering energy law {type(energy).__name__}"
        )
    if isinstance(law, KalbachMannLaw):
        tables = _ascending(law.spectrum)
        return _PreparedLaw(
            "kalbach",
            law,
            n_cosine,
            tables=tables,
            rules=_point_rules(tables),
            precompound=law.precompound[::-1].copy(),
            slope=law.slope[::-1].copy(),
        )
    if isinstance(law, CorrelatedLaw):
        tables = _ascending(law.spectrum)
        cosines = _point_cosines(
            law.mu_start[::-1].copy(),
            law.mu_stop[::-1].copy(),
            law.mu_rule[::-1].copy(),
            law.mu,
            law.mu_pdf,
            law.mu_cdf,
            n_cosine,
        )
        return _PreparedLaw(
            "correlated", law, n_cosine, tables=tables, rules=_point_rules(tables), cosines=cosines
        )
    return _PreparedLaw("unsupported", law, n_cosine, error=f"scattering law {type(law).__name__}")


def _prepared_points(prepared, e, awr, center_of_mass, du):
    """law_points for a _PreparedLaw."""
    kind = prepared.kind
    law = prepared.law
    n_cosine = prepared.n_cosine
    if kind == "unsupported":
        raise NotImplementedError(prepared.error)
    if kind in ("level", "tabular", "evaporation"):
        if kind == "level":
            energy = law.energy
            if e <= energy.threshold:
                return np.empty(0), np.empty(0), np.empty(0)
            e_cm = np.array([energy.mass_ratio * (e - energy.threshold)])
            lo_cm = hi_cm = e_cm
            w = np.ones(1)
        elif kind == "tabular":
            e_cm, w, _, _, _, lo_cm, hi_cm = tabular_points(
                *prepared.tables, e, True, not prepared.histogram, du
            )
        else:
            lo_cm, hi_cm, w = _evaporation_points(law.energy, e, du, prepared.theta)
            e_cm = 0.5 * (lo_cm + hi_cm)
        if not center_of_mass:
            return lo_cm, hi_cm, w
        if law.angle is None:
            low, high, cosine_w = _uniform_strata(n_cosine)
        else:
            low, high, cosine_w = _angle_strata(law.angle, e, n_cosine, prepared.strata)
        return _lab_segments(e, e_cm, w, low, high, cosine_w, awr)

    tables = prepared.tables
    e_cm, w, _, k, r1, lo_cm, hi_cm = tabular_points(*tables, e, False, True, du)
    if not center_of_mass:
        return lo_cm, hi_cm, w
    if kind == "kalbach":
        return _kalbach_segments(
            e,
            e_cm,
            w,
            k,
            r1,
            prepared.rules,
            prepared.precompound,
            prepared.slope,
            tables[3],
            tables[4],
            tables[5],
            n_cosine,
            awr,
        )
    return _correlated_segments(
        e, e_cm, w, k, r1, prepared.rules, tables[5], prepared.cosines, n_cosine, awr
    )


def law_points(law, e, awr, center_of_mass, du=ENERGY_RESOLUTION, n_cosine=COSINE_POINTS):
    """Lab outgoing energy segments of one law at incoming e: (lower, upper, weight).

    Each segment carries its probability, spread over its interval. Weights sum to one.
    """
    return _prepared_points(_prepare_law(law, n_cosine), e, awr, center_of_mass, du)


@dataclass(frozen=True)
class _PreparedReaction:
    """A ReactionLaw with its laws prepared and its applicability flipped, once."""

    reaction: object
    laws: tuple
    applicability: tuple


def _prepare_reaction(reaction, n_cosine):
    applicability = ()
    if len(reaction.laws) > 1:
        applicability = tuple(
            _ascending_tab1(a.energy, a.probability, a) for a in reaction.applicability
        )
    return _PreparedReaction(
        reaction, tuple(_prepare_law(law, n_cosine) for law in reaction.laws), applicability
    )


def _shares(n, applicability, e):
    if n == 1:
        return np.ones(1)
    shares = np.zeros(n)
    remaining = 1.0
    for i, arguments in enumerate(applicability):
        p = evaluate_tabulated(*arguments, np.array([e]))[0]
        take = min(max(p, 0.0), remaining)
        shares[i] = take
        remaining -= take
    shares[-1] += remaining
    return shares


def applicability_shares(reaction, e):
    """Probability of each law of a reaction at e, as ReactionProduct::sample_dist picks them."""
    n = len(reaction.laws)
    if n == 1:
        return np.ones(1)
    return _shares(
        n, [_ascending_tab1(a.energy, a.probability, a) for a in reaction.applicability], e
    )


def _reaction_parts(prepared, e, awr, du):
    """Each applicable law's (lower, upper, weight) at e, weighted by its share, unconcatenated."""
    reaction = prepared.reaction
    parts = []
    shares = _shares(len(prepared.laws), prepared.applicability, e)
    for share, law in zip(shares, prepared.laws, strict=True):
        if share <= 0.0:
            continue
        lo, hi, w = _prepared_points(law, e, awr, reaction.center_of_mass, du)
        parts.append((lo, hi, share * w))
    return parts


def reaction_points(reaction, e, awr, du=ENERGY_RESOLUTION, n_cosine=COSINE_POINTS):
    """Lab outgoing segments of a ReactionLaw at e: (lower, upper, weight), summing to one."""
    parts = _reaction_parts(_prepare_reaction(reaction, n_cosine), e, awr, du)
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(3))


def _prepared_deposit(
    prepared, e, awr, center_of_mass, du, share, scale, row, covered, touched, grid_u, top
):
    """Deposit _prepared_points' segments, weighted by share then scale, without making them.

    Returns how many segments there were and the weight landing off the grid. Values match
    depositing the segments' arrays to round-off in the order of summation.
    """
    kind = prepared.kind
    law = prepared.law
    n_cosine = prepared.n_cosine
    if kind == "unsupported":
        raise NotImplementedError(prepared.error)
    if kind in ("kalbach", "correlated"):
        tables = prepared.tables
        e_cm, w, _, k, r1, lo_cm, hi_cm = tabular_points(*tables, e, False, True, du)
        if not center_of_mass:
            return w.shape[0], _accumulate(
                row, covered, touched, lo_cm, hi_cm, share * w, scale, grid_u, top
            )
        if kind == "kalbach":
            lost = _deposit_kalbach(
                row,
                covered,
                touched,
                grid_u,
                top,
                e,
                e_cm,
                w,
                k,
                r1,
                prepared.rules,
                prepared.precompound,
                prepared.slope,
                tables[3],
                tables[4],
                tables[5],
                n_cosine,
                share,
                scale,
                awr,
            )
            return 2 * n_cosine * e_cm.shape[0], lost
        lost = _deposit_correlated(
            row,
            covered,
            touched,
            grid_u,
            top,
            e,
            e_cm,
            w,
            k,
            r1,
            prepared.rules,
            tables[5],
            prepared.cosines,
            n_cosine,
            share,
            scale,
            awr,
        )
        return n_cosine * e_cm.shape[0], lost
    if kind == "level":
        energy = law.energy
        if e <= energy.threshold:
            return 0, 0.0
        e_cm = np.array([energy.mass_ratio * (e - energy.threshold)])
        lo_cm = hi_cm = e_cm
        w = np.ones(1)
    elif kind == "tabular":
        e_cm, w, _, _, _, lo_cm, hi_cm = tabular_points(
            *prepared.tables, e, True, not prepared.histogram, du
        )
    else:
        lo_cm, hi_cm, w = _evaporation_points(law.energy, e, du, prepared.theta)
        e_cm = 0.5 * (lo_cm + hi_cm)
    if not center_of_mass:
        return w.shape[0], _accumulate(
            row, covered, touched, lo_cm, hi_cm, share * w, scale, grid_u, top
        )
    if law.angle is None:
        low, high, cosine_w = _uniform_strata(n_cosine)
    else:
        low, high, cosine_w = _angle_strata(law.angle, e, n_cosine, prepared.strata)
    lost = _deposit_strata(
        row, covered, touched, grid_u, top, e, e_cm, w, low, high, cosine_w, share, scale, awr
    )
    return e_cm.shape[0] * low.shape[0], lost


def _reaction_deposit(prepared, e, awr, du, scale, row, covered, touched, grid_u, top):
    """_prepared_deposit for each applicable law of a reaction: (segments, weight off the grid)."""
    count = 0
    lost = 0.0
    shares = _shares(len(prepared.laws), prepared.applicability, e)
    for share, law in zip(shares, prepared.laws, strict=True):
        if share <= 0.0:
            continue
        c, off = _prepared_deposit(
            law,
            e,
            awr,
            prepared.reaction.center_of_mass,
            du,
            share,
            scale,
            row,
            covered,
            touched,
            grid_u,
            top,
        )
        count += c
        lost += off
    return count, lost


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


@njit(cache=True)
def _add_segment(row, covered, touched, grid_u, a, count_a, b, count_b, w):
    """Add weight w, uniform in lethargy over [a, b], to a row in place, as _deposit_segments does,
    and return the weight landing off the grid.

    count_a and count_b are the numbers of grid nodes at or below a and b, so callers that walk
    sorted bounds can skip the searches. Whole intervals go to covered as differences, for
    _take_row, and touched holds the lowest and highest interval used.
    """
    n = grid_u.shape[0]
    if b - a <= 1e-14 * max(1.0, abs(b)):
        # A point: split between the two hats it lies between.
        if a < grid_u[0] or a > grid_u[n - 1]:
            return w
        m = min(max(count_a - 1, 0), n - 2)
        theta = (a - grid_u[m]) / (grid_u[m + 1] - grid_u[m])
        row[m] += w * (1.0 - theta)
        row[m + 1] += w * theta
        touched[0] = min(touched[0], m)
        touched[1] = max(touched[1], m)
        return 0.0
    density = w / (b - a)
    inside_a = max(a, grid_u[0])
    inside_b = min(b, grid_u[n - 1])
    if inside_b <= inside_a:
        return w
    lost = density * ((b - a) - (inside_b - inside_a))
    # The intervals np.searchsorted finds at the clipped ends, sides "right" and "left".
    m_a = 0 if a < grid_u[0] else min(max(count_a - 1, 0), n - 2)
    if b > grid_u[n - 1]:
        m_b = n - 2
    else:
        below = count_b - 1 if count_b > 0 and grid_u[count_b - 1] == b else count_b
        m_b = min(max(below - 1, 0), n - 2)
    touched[0] = min(touched[0], m_a)
    touched[1] = max(touched[1], m_b)
    if m_a == m_b:
        _partial(row, density, grid_u, m_a, inside_a, inside_b)
        return lost
    _partial(row, density, grid_u, m_a, inside_a, grid_u[m_a + 1])
    _partial(row, density, grid_u, m_b, grid_u[m_b], inside_b)
    covered[m_a + 1] += density
    covered[m_b] -= density
    return lost


@njit(cache=True)
def _off_grid_bounds(grid_u, top):
    """Energies below and above which a segment certainly misses the grid, so deposit would add
    its whole weight to the loss. The margin is far above the round-off of the logs."""
    return top * math.exp(-grid_u[grid_u.shape[0] - 1]) * (1.0 - 1e-9), top * (1.0 + 1e-9)


@njit(cache=True)
def _accumulate(row, covered, touched, lower, upper, weights, scale, grid_u, top):
    """Add lab segments of weights * scale to a row in place, as deposit does, and return the
    weight landing off the grid. touched and covered are as in _add_segment."""
    below, above = _off_grid_bounds(grid_u, top)
    lost = 0.0
    for s in range(lower.shape[0]):
        w = weights[s] * scale
        if (lower[s] < below and upper[s] < below) or (lower[s] > above and upper[s] > above):
            lost += w
            continue
        start = math.log(top / max(upper[s], 1e-300))
        stop = math.log(top / max(lower[s], 1e-300))
        a = min(start, stop)
        b = max(start, stop)
        lost += _add_segment(
            row,
            covered,
            touched,
            grid_u,
            a,
            np.searchsorted(grid_u, a, side="right"),
            b,
            np.searchsorted(grid_u, b, side="right"),
            w,
        )
    return lost


@njit(cache=True)
def _fan(
    row,
    covered,
    touched,
    grid_u,
    top,
    e,
    e_cm,
    low,
    high,
    base,
    cosine_w,
    divisor,
    share,
    scale,
    awr,
    guess,
):
    """Deposit the lab segments of one centre-of-mass energy's cosine strata, low[m] to high[m].

    Each weighs (share * (base * cosine_w[m])) * scale, or base / divisor in place of the product
    if divisor is positive: the arithmetic of the segment arrays law_points returns, weighted by
    applicability share and then by the hat's share, as term_transfer deposited them. Adjacent
    strata share a bound, whose lab energy and lethargy are then computed once, bounds are found
    by walking from the last, and segments wholly off the grid skip both. Returns the weight off
    the grid and a node count to start the next walk from.

    _add_segment and _partial are written out here, with a walking node search: calls passing
    arrays cost reference counting per segment, which was a quarter of the build.
    """
    n = grid_u.shape[0]
    g_first = grid_u[0]
    g_last = grid_u[n - 1]
    below = top * math.exp(-g_last) * (1.0 - 1e-9)
    above = top * (1.0 + 1e-9)
    t_low = touched[0]
    t_high = touched[1]
    lost = 0.0
    lab_high = 0.0
    u_high = 0.0
    count_high = guess
    known = False  # whether u_high and count_high hold the previous upper bound's
    for m in range(low.shape[0]):
        if m > 0 and low[m] == high[m - 1]:
            lab_low = lab_high
            reuse = known
        else:
            lab_low = _lab(e, e_cm, low[m], awr)
            reuse = False
        lab_high = _lab(e, e_cm, high[m], awr)
        weight = base / divisor if divisor > 0.0 else base * cosine_w[m]
        w = (share * weight) * scale
        if (lab_low < below and lab_high < below) or (lab_low > above and lab_high > above):
            lost += w
            known = False
            continue
        if reuse:
            u_low = u_high
            count_low = count_high
        else:
            u_low = math.log(top / max(lab_low, 1e-300))
            if guess < 0:
                count_low = np.searchsorted(grid_u, u_low, side="right")
            else:
                count_low = guess
                while count_low < n and grid_u[count_low] <= u_low:
                    count_low += 1
                while count_low > 0 and grid_u[count_low - 1] > u_low:
                    count_low -= 1
        u_high = math.log(top / max(lab_high, 1e-300))
        count_high = count_low
        while count_high < n and grid_u[count_high] <= u_high:
            count_high += 1
        while count_high > 0 and grid_u[count_high - 1] > u_high:
            count_high -= 1
        guess = count_high
        known = True
        if u_high <= u_low:
            a, count_a, b, count_b = u_high, count_high, u_low, count_low
        else:
            a, count_a, b, count_b = u_low, count_low, u_high, count_high
        # _add_segment, inline.
        if b - a <= 1e-14 * max(1.0, abs(b)):
            if a < g_first or a > g_last:
                lost += w
                continue
            k = min(max(count_a - 1, 0), n - 2)
            theta = (a - grid_u[k]) / (grid_u[k + 1] - grid_u[k])
            row[k] += w * (1.0 - theta)
            row[k + 1] += w * theta
            t_low = min(t_low, k)
            t_high = max(t_high, k)
            continue
        density = w / (b - a)
        inside_a = max(a, g_first)
        inside_b = min(b, g_last)
        if inside_b <= inside_a:
            lost += w
            continue
        lost += density * ((b - a) - (inside_b - inside_a))
        m_a = 0 if a < g_first else min(max(count_a - 1, 0), n - 2)
        if b > g_last:
            m_b = n - 2
        else:
            below_b = count_b - 1 if count_b > 0 and grid_u[count_b - 1] == b else count_b
            m_b = min(max(below_b - 1, 0), n - 2)
        t_low = min(t_low, m_a)
        t_high = max(t_high, m_b)
        # _partial, inline, for the end intervals.
        top_a = inside_b if m_a == m_b else grid_u[m_a + 1]
        width = grid_u[m_a + 1] - grid_u[m_a]
        t0 = (inside_a - grid_u[m_a]) / width
        t1 = (top_a - grid_u[m_a]) / width
        upper_part = 0.5 * (t1 * t1 - t0 * t0)
        row[m_a] += density * width * ((t1 - t0) - upper_part)
        row[m_a + 1] += density * width * upper_part
        if m_a == m_b:
            continue
        width = grid_u[m_b + 1] - grid_u[m_b]
        t0 = (grid_u[m_b] - grid_u[m_b]) / width
        t1 = (inside_b - grid_u[m_b]) / width
        upper_part = 0.5 * (t1 * t1 - t0 * t0)
        row[m_b] += density * width * ((t1 - t0) - upper_part)
        row[m_b + 1] += density * width * upper_part
        covered[m_a + 1] += density
        covered[m_b] -= density
    touched[0] = t_low
    touched[1] = t_high
    return lost, guess


@njit(cache=True)
def _deposit_strata(
    row, covered, touched, grid_u, top, e, e_cm, w, low, high, cosine_w, share, scale, awr
):
    """_fan for every centre-of-mass energy, with the same strata for each."""
    lost = 0.0
    guess = -1
    for i in range(e_cm.shape[0]):
        part, guess = _fan(
            row,
            covered,
            touched,
            grid_u,
            top,
            e,
            e_cm[i],
            low,
            high,
            w[i],
            cosine_w,
            0.0,
            share,
            scale,
            awr,
            guess,
        )
        lost += part
    return lost


@njit(cache=True)
def _deposit_kalbach(
    row,
    covered,
    touched,
    grid_u,
    top,
    e,
    e_cm,
    w,
    k,
    r1,
    point_rule,
    precompound,
    slope,
    e_asc,
    p_asc,
    c_asc,
    n_cosine,
    share,
    scale,
    awr,
):
    """_kalbach_segments, deposited as they are made."""
    lost = 0.0
    guess = -1
    for i in range(e_cm.shape[0]):
        j = k[i]
        if point_rule[j] == RULE_HISTOGRAM:
            km_r = precompound[j]
            km_a = slope[j]
        else:
            raw = _sample_out(
                e_asc[j], e_asc[j + 1], p_asc[j], p_asc[j + 1], c_asc[j], RULE_LIN_LIN, r1[i]
            )
            t = (raw - e_asc[j]) / (e_asc[j + 1] - e_asc[j])
            km_r = precompound[j] + t * (precompound[j + 1] - precompound[j])
            km_a = slope[j] + t * (slope[j + 1] - slope[j])
        low, high, cosine_w = kalbach_cosines(km_a, km_r, n_cosine)
        part, guess = _fan(
            row,
            covered,
            touched,
            grid_u,
            top,
            e,
            e_cm[i],
            low,
            high,
            w[i],
            cosine_w,
            0.0,
            share,
            scale,
            awr,
            guess,
        )
        lost += part
    return lost


@njit(cache=True)
def _deposit_correlated(
    row,
    covered,
    touched,
    grid_u,
    top,
    e,
    e_cm,
    w,
    k,
    r1,
    point_rule,
    c_asc,
    cosines,
    n_cosine,
    share,
    scale,
    awr,
):
    """_correlated_segments, deposited as they are made."""
    lost = 0.0
    guess = -1
    unused = np.empty(0)
    for i in range(e_cm.shape[0]):
        j = k[i]
        near = point_rule[j] == RULE_HISTOGRAM or r1[i] - c_asc[j] < c_asc[j + 1] - r1[i]
        point = j if near else j + 1
        bounds = cosines[point]
        part, guess = _fan(
            row,
            covered,
            touched,
            grid_u,
            top,
            e,
            e_cm[i],
            bounds[:-1],
            bounds[1:],
            w[i],
            unused,
            float(n_cosine),
            share,
            scale,
            awr,
            guess,
        )
        lost += part
    return lost


@njit(cache=True)
def _take_row(row, covered, touched, grid_u):
    """Spread covered into row over the touched intervals, as _deposit_segments does, and return
    the row's columns first:last that hold its nonzero values. Both arrays are left zero."""
    level = 0.0
    for m in range(touched[0], touched[1] + 1):
        level += covered[m]
        covered[m] = 0.0
        if level != 0.0:
            half = 0.5 * level * (grid_u[m + 1] - grid_u[m])
            row[m] += half
            row[m + 1] += half
    first = touched[0]
    last = touched[1] + 2
    while first < last and row[first] == 0.0:
        first += 1
    while last > first and row[last - 1] == 0.0:
        last -= 1
    values = row[first:last].copy()
    row[touched[0] : touched[1] + 2] = 0.0
    return first, last, values


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


@njit(cache=True)
def _block_sum(a, start, n):
    """NumPy's pairwise_sum of a[start:start + n] for n <= 128: eight running sums, then a tree."""
    if n < 8:
        total = 0.0
        for i in range(start, start + n):
            total += a[i]
        return total
    r = np.empty(8)
    for q in range(8):
        r[q] = a[start + q]
    i = 8
    while i < n - n % 8:
        for q in range(8):
            r[q] += a[start + i + q]
        i += 8
    total = (r[0] + r[1] + (r[2] + r[3])) + (r[4] + r[5] + (r[6] + r[7]))
    while i < n:
        total += a[start + i]
        i += 1
    return total


@njit(cache=True)
def _pairwise_sum(a, start, n):
    """NumPy's pairwise summation of a[start:start + n], so sums match np.add.reduce's bits.

    NumPy halves blocks above 128 recursively; this walks the same halves with a stack, since a
    cached recursive Numba function can crash on loading.
    """
    if n <= 128:
        return _block_sum(a, start, n)
    starts = np.empty(64, dtype=np.int64)
    sizes = np.empty(64, dtype=np.int64)
    partial = np.empty(64)
    state = np.zeros(64, dtype=np.int64)  # 0: not split, 1: left done
    depth = 0
    starts[0] = start
    sizes[0] = n
    result = 0.0
    while depth >= 0:
        size = sizes[depth]
        if size <= 128:
            value = _block_sum(a, starts[depth], size)
            depth -= 1
            while depth >= 0 and state[depth] == 2:
                value = partial[depth] + value
                state[depth] = 0
                depth -= 1
            if depth < 0:
                result = value
                break
            # value is the left half of depth's block: keep it and descend into the right half.
            partial[depth] = value
            state[depth] = 2
            half = sizes[depth] // 2
            half -= half % 8
            starts[depth + 1] = starts[depth] + half
            sizes[depth + 1] = sizes[depth] - half
            state[depth + 1] = 0
            depth += 1
            continue
        half = size // 2
        half -= half % 8
        starts[depth + 1] = starts[depth]
        sizes[depth + 1] = half
        state[depth + 1] = 0
        depth += 1
    return result


@njit(cache=True)
def _trapezoid(y, x):
    """np.trapezoid(y, x), summed in the same order."""
    n = x.shape[0] - 1
    if n <= 0:
        return 0.0
    terms = np.empty(n)
    for i in range(n):
        terms[i] = (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2.0
    return _pairwise_sum(terms, 0, n)


@njit(cache=True)
def _hat_moments_kernel(coarse, u, curve, grid_u):
    n = coarse.shape[0]
    integral = np.zeros(n)
    centroid = coarse.copy()
    for j in range(n):
        low = grid_u[j - 1] if j > 0 else grid_u[0]
        high = grid_u[j + 1] if j + 1 < n else grid_u[n - 1]
        # u ascends, so the curve's nodes strictly inside the support are one slice.
        i0 = np.searchsorted(u, low, side="right")
        i1 = max(np.searchsorted(u, high, side="left"), i0)
        raw = np.empty(i1 - i0 + 3)
        raw[: i1 - i0] = u[i0:i1]
        raw[i1 - i0] = low
        raw[i1 - i0 + 1] = grid_u[j]
        raw[i1 - i0 + 2] = high
        points = np.unique(raw)
        values = np.interp(points, u, curve)
        weight = np.empty(points.shape[0])
        nonzero = False
        for q in range(points.shape[0]):
            p = points[q]
            if p < u[0] or p > u[u.shape[0] - 1]:
                values[q] = 0.0
            if p <= grid_u[j]:
                shape = (p - low) / (grid_u[j] - low) if j > 0 else 1.0
            else:
                shape = (high - p) / (high - grid_u[j]) if j + 1 < n else 1.0
            weight[q] = shape * values[q]
            nonzero = nonzero or weight[q] != 0.0
        if not nonzero:
            continue
        integral[j] = _trapezoid(weight, points)
        if integral[j] > 0.0:
            centroid[j] = coarse[0] * np.exp(-_trapezoid(weight * points, points) / integral[j])
    return integral, centroid


def _hat_moments(coarse, energy, curve):
    """Each hat's integral of a lin-lin curve over lethargy, and its centroid energy.

    The integrand is sampled at the curve's own nodes inside the hat's support and at the hat's
    nodes, then integrated by trapezoids, which follows resonances in the curve.
    """
    return _hat_moments_kernel(
        coarse,
        np.log(coarse[0] / energy),
        np.ascontiguousarray(curve, dtype=float),
        np.log(coarse[0] / coarse),
    )


def point_spacing(coarse):
    """Lethargy between outgoing-energy pieces for a coarse grid: a fraction of its median hat.

    Pieces are spread over their intervals, so this sets how well a piece's uniform spread follows
    the spectrum inside it, not whether hats are hit at all. A sliver hat between two close
    thresholds (1e-6 wide at 6.5 MeV for U238 and O16) takes its exact share of each piece, so it
    must not set the spacing: the narrowest hat made heavy-fast's kernels 40 times slower.
    """
    return min(ENERGY_RESOLUTION, np.median(np.log(coarse[:-1] / coarse[1:])) / PIECES_PER_HAT)


@dataclass(frozen=True)
class SparseRows:
    """Rows compressed by their first and last nonzero column, as InelasticKernels holds them.

    Row j holds values[starts[j]:starts[j] + last[j] - first[j]] for columns first[j]:last[j].
    """

    starts: np.ndarray
    first: np.ndarray
    last: np.ndarray
    values: np.ndarray

    def dense(self):
        """The rows as a dense matrix, for tests and small grids."""
        n = self.first.shape[0]
        out = np.zeros((n, n))
        for j in range(n):
            out[j, self.first[j] : self.last[j]] = self.values[
                self.starts[j] : self.starts[j] + self.last[j] - self.first[j]
            ]
        return out

    def sums(self):
        return np.array(
            [
                self.values[self.starts[j] : self.starts[j] + self.last[j] - self.first[j]].sum()
                for j in range(self.first.shape[0])
            ]
        )


# Above this many emitting hats per worker, build_inelastic spreads hats over processes.
HATS_PER_WORKER = 256


def _hat_rows(laws, awr, coarse, du, n_cosine, hats, integrals, centroids, totals):
    """Rows of the given hats, as (first, last, values per hat, lost). One process's share."""
    reactions = [_prepare_reaction(r, n_cosine) for r in laws]
    n = coarse.shape[0]
    top = coarse[0]
    grid_u = np.log(top / coarse)
    row = np.zeros(n)
    covered = np.zeros(n)
    touched = np.empty(2, dtype=np.int64)
    first = np.zeros(len(hats), dtype=np.int64)
    last = np.zeros(len(hats), dtype=np.int64)
    lost = np.zeros(len(hats))
    values = []
    for h, j in enumerate(hats):
        total = totals[j]
        touched[0] = n
        touched[1] = -1
        for r, reaction in enumerate(reactions):
            if integrals[r, j] <= 0.0:
                continue
            share = integrals[r, j] / total
            work = (awr, du, share, row, covered, touched, grid_u, top)
            count, off = _reaction_deposit(reaction, centroids[r, j], *work)
            if count == 0 and j > 0:
                # A hat straddling a threshold can have its centroid below it, so take the reaction
                # at the hat's top instead of dropping its share. Nothing was deposited above.
                count, off = _reaction_deposit(reaction, coarse[j - 1], *work)
            if count == 0:
                lost[h] += share
                continue
            lost[h] += off
        a, b, v = _take_row(row, covered, touched, grid_u)
        if v.shape[0] > 0:
            first[h], last[h] = a, b
        values.append(v)
    return first, last, values, lost


def _start_method(workers):
    """fork where it exists, since spawn re-runs a script without a main guard in every worker.
    Elsewhere workers run only when asked for."""
    import multiprocessing

    if "fork" in multiprocessing.get_all_start_methods():
        return "fork"
    return "spawn" if workers is not None else None


def _worker_count(workers, emitting):
    import os

    if workers is not None:
        return max(1, int(workers))
    if _start_method(workers) is None:
        return 1
    return max(1, min(os.cpu_count() or 1, emitting // HATS_PER_WORKER))


def term_transfer(name, temperature, awr, coarse, du=None, n_cosine=COSINE_POINTS, workers=None):
    """One term's rows on a coarse grid as SparseRows, the part of each hat landing off the grid,
    and its emission curve as (energies, values) for lin-lin.

    Row j mixes reactions by their hat-integrated yield * cross section, each evaluated at its
    centroid in the hat. So every hat that emits has a full row, even one whose node lies below a
    threshold, and reaction shares are held fixed across a hat: an O(du) approximation.

    Hats are independent, so workers processes (by default one per HATS_PER_WORKER emitting hats,
    up to the CPU count) each take every workers-th hat. Each row is the same however they split.
    """
    from zerog.data.neutron import read_reactions, read_scattering_yields
    from zerog.data.scatter import read_scattering_laws

    du = point_spacing(coarse) if du is None else du
    n = coarse.shape[0]
    curves = read_reactions(name, temperature)
    laws = [r for r in read_scattering_laws(name) if r.mt != 2]
    lost = np.zeros(n)
    first = np.zeros(n, dtype=np.int64)
    last = np.zeros(n, dtype=np.int64)
    starts = np.zeros(n, dtype=np.int64)
    mts = np.array([law.mt for law in laws], dtype=np.int64)
    if mts.shape[0] == 0:
        return (
            SparseRows(starts, first, last, np.empty(0)),
            lost,
            curves.energy,
            np.zeros(curves.energy.shape[0]),
        )
    index = np.array([list(curves.mt).index(mt) for mt in mts])
    produced = curves.xs[index] * read_scattering_yields(name, mts, curves.energy)
    emission = produced.sum(axis=0)
    integrals = np.empty((len(laws), n))
    centroids = np.empty((len(laws), n))
    for r, curve in enumerate(produced):
        integrals[r], centroids[r] = _hat_moments(coarse, curves.energy, curve)
    totals = np.zeros(n)
    for r in range(len(laws)):
        totals += integrals[r]
    hats = np.nonzero(totals > 0.0)[0]
    count = _worker_count(workers, hats.shape[0])
    arguments = (laws, awr, coarse, du, n_cosine)
    tables = (integrals, centroids, totals)
    if count == 1:
        results = [(hats, _hat_rows(*arguments, hats, *tables))]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # Costs vary smoothly with energy, so interleaved shares balance.
        shares = [hats[w::count] for w in range(count)]
        context = multiprocessing.get_context(_start_method(workers))
        with ProcessPoolExecutor(count, mp_context=context) as pool:
            futures = [pool.submit(_hat_rows, *arguments, share, *tables) for share in shares]
            results = [(share, f.result()) for share, f in zip(shares, futures, strict=True)]
    pieces = [None] * n
    for share, (a, b, v, off) in results:
        first[share] = a
        last[share] = b
        lost[share] = off
        for j, values in zip(share, v, strict=True):
            pieces[j] = values
    offset = 0
    values = []
    for j in range(n):
        starts[j] = offset
        if pieces[j] is not None and pieces[j].shape[0] > 0:
            values.append(pieces[j])
            offset += pieces[j].shape[0]
    flat = np.concatenate(values) if values else np.empty(0)
    return SparseRows(starts, first, last, flat), lost, curves.energy, emission


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


def build_inelastic(
    terms, energy, coarse, du=None, n_cosine=COSINE_POINTS, cache=None, workers=None
):
    """InelasticKernels for terms given as (name, temperature label, awr), or None for no reactions.

    coarse must be a subset of energy and span it. With cache, a directory, kernels are stored
    there and reused by any later call with the same grid, terms, settings, data files and code.
    workers is passed to term_transfer, and doesn't change the kernels.
    """
    if cache is not None:
        from pathlib import Path

        path = Path(cache) / f"inelastic-{_cache_key(terms, energy, coarse, du, n_cosine)}.npz"
        if path.exists():
            with np.load(path) as stored:
                return InelasticKernels(**{name: stored[name] for name in stored.files})
        kernels = build_inelastic(terms, energy, coarse, du, n_cosine, workers=workers)
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
            starts[t] = offset
            continue
        name, label, awr = term
        rows, lost[t], table_energy, curve = term_transfer(
            name, label, awr, coarse, du, n_cosine, workers
        )
        emission[t] = np.interp(energy[::-1], table_energy[::-1], curve[::-1])[::-1]
        starts[t] = rows.starts + offset
        first[t] = rows.first
        last[t] = rows.last
        values.append(rows.values)
        offset += rows.values.shape[0]
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
