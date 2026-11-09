"""Unresolved-range probability tables as bands of the energy subproblem.

OpenMC v0.16.0 (Nuclide::calculate_urr_xs) samples, at every flight strictly inside a nuclide's
table range, one random number r per nuclide. The same r picks a band at the two bracketing tables,
i = lower_bound_index(energy, E) and i + 1, as the first band whose cdf exceeds r, and the band's
elastic, fission and capture values are interpolated between them in E (lin-lin; log-log where
both are positive). With multiply_smooth they are factors on the smooth elastic, fission and
absorption less fission. Negative values become zero. The total is elastic + capture + fission +
the flagged inelastic reaction's smooth cross section, so other reactions drop out inside the
range. After each collision that changes the energy the stream advances
(physics.cpp: advance_prn_seed), so the next flight draws a fresh r for every nuclide.

So a flight's cross sections are a function of r = (r_1, ..., r_K), independent between flights,
and the expected flux splits into F_c, one per point c of the unit cube, all fed by one scattering
source in proportion to their weight. The cube is integrated here by a fixed scrambled Sobol set
of points with equal weights, which handles nuclides sampled independently without the full
product of their bands.

At one energy a band's values are steps in r, so a point set converges only as 1 / points where a
stratum straddles a cdf breakpoint, and with fewer points than bands even the mean is off by
percents. So the march scales each banded reaction at each node so that the points' mean is the
exact mean over r, band_mean, which is the infinitely dilute cross section; the points then carry
only the distribution's shape. The scale tends to one as points are added.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from zerog.data.neutron import MT_ELASTIC, MT_FISSION, MT_PARTIAL_FISSION, read_reactions
from zerog.data.urr import CAPTURE, CDF, ELASTIC, FISSION, read_urr
from zerog.energy.grid import interpolate

RULE_LIN_LIN = 2
RULE_LOG_LOG = 5
SOBOL_SEED = 20261108
GAP = 1e-10  # node pairs at a range's ends, where OpenMC's strict bounds switch the tables on

# Reactions OpenMC counts as absorption besides fission (endf.cpp, is_disappearance): MT 101 to
# 117, 600 to 849, and six light-particle reactions. MT 101 is redundant, and read_reactions skips
# redundant reactions, so it adds nothing.
MT_ABSORPTION = (*range(101, 118), *range(600, 850), 155, 182, 191, 192, 193, 197)


@dataclass(frozen=True)
class UrrData:
    """Every banded term's tables and smooth curves, flat, for the march.

    terms[j] is the subproblem term of banded term j. Its tables are rows start[j]:start[j + 1] of
    energy (ascending) and of cdf, elastic, fission and capture, (rows, bands), padded with cdf 1
    past its bands[j]. smooth is (banded terms, 4, fine nodes): elastic, absorption less fission,
    fission and the flagged inelastic reaction (zero if none). points is (combinations, banded
    terms), and weights (combinations) sum to one: 1 / combinations for band_points.
    """

    terms: np.ndarray
    start: np.ndarray
    bands: np.ndarray
    energy: np.ndarray
    cdf: np.ndarray
    elastic: np.ndarray
    fission: np.ndarray
    capture: np.ndarray
    multiply: np.ndarray
    rule: np.ndarray
    smooth: np.ndarray
    points: np.ndarray
    weights: np.ndarray

    @property
    def combinations(self):
        return self.points.shape[0]


def no_urr(n_energy):
    """Empty band data, for problems without probability tables."""
    return UrrData(
        terms=np.zeros(0, dtype=np.int64),
        start=np.zeros(1, dtype=np.int64),
        bands=np.zeros(0, dtype=np.int64),
        energy=np.zeros(0),
        cdf=np.zeros((0, 1)),
        elastic=np.zeros((0, 1)),
        fission=np.zeros((0, 1)),
        capture=np.zeros((0, 1)),
        multiply=np.zeros(0, dtype=np.bool_),
        rule=np.zeros(0, dtype=np.int64),
        smooth=np.zeros((0, 4, n_energy)),
        points=np.full((1, 0), 0.5),
        weights=np.ones(1),
    )


def band_points(n_terms, n_points, seed=SOBOL_SEED):
    """A fixed scrambled Sobol set of n_points (a power of two) in the unit cube of n_terms."""
    from scipy.stats import qmc

    if n_terms == 0:
        return np.full((1, 0), 0.5)
    if n_points & (n_points - 1):
        raise ValueError(f"n_points must be a power of two, not {n_points}")
    sampler = qmc.Sobol(d=n_terms, scramble=True, seed=seed)
    return sampler.random_base2(int(np.log2(n_points)))


def smooth_curves(name, label, flag, energy):
    """(4, fine nodes): smooth elastic, absorption less fission, fission and the flagged inelastic
    reaction, as calculate_urr_xs uses them."""
    data = read_reactions(name, label)
    fission_mts = (MT_FISSION, *MT_PARTIAL_FISSION)

    def curve(mts):
        selected = data.xs[np.isin(data.mt, mts)]
        values = selected.sum(axis=0) if selected.shape[0] else np.zeros(data.energy.shape[0])
        return interpolate(energy, data.energy, values)

    out = np.zeros((4, energy.shape[0]))
    out[0] = curve((MT_ELASTIC,))
    out[1] = curve(MT_ABSORPTION)  # absorption less fission: fission isn't among these MTs
    out[2] = curve(fission_mts)
    if flag > 0:
        if flag == 4:
            # MT 4 is redundant: its value is the sum of the levels and continuum.
            out[3] = curve(tuple(range(50, 92)))
        else:
            out[3] = curve((flag,))
    return out


def build_urr(nuclides, energy, n_points, terms=None):
    """UrrData for (name, label) pairs, for those with tables; terms maps each to its subproblem
    term, by default its position. n_points is a number of Sobol combinations, or the points
    themselves, (combinations, nuclides with tables), with equal weights; see with_points for
    others."""
    terms = list(range(len(nuclides))) if terms is None else list(terms)
    tables = []
    for t, (name, label) in zip(terms, nuclides, strict=True):
        table = read_urr(name, label)
        if table is None:
            continue
        if table.interpolation != RULE_LIN_LIN:
            # band_mean, which the march scales to, is the exact mean only for lin-lin tables.
            # Every table in ENDF/B-VIII.0 is lin-lin.
            raise NotImplementedError(f"{name}: table interpolation {table.interpolation}")
        tables.append((t, name, label, table))
    if not tables:
        return no_urr(energy.shape[0])
    points = band_points(len(tables), n_points) if np.isscalar(n_points) else np.asarray(n_points)
    width = max(tb.table.shape[2] for *_, tb in tables)
    rows = sum(tb.energy.shape[0] for *_, tb in tables)
    cdf = np.ones((rows, width))
    elastic = np.zeros((rows, width))
    fission = np.zeros((rows, width))
    capture = np.zeros((rows, width))
    flat_energy = np.zeros(rows)
    start = np.zeros(len(tables) + 1, dtype=np.int64)
    smooth = np.zeros((len(tables), 4, energy.shape[0]))
    for j, (_, name, label, tb) in enumerate(tables):
        a = start[j]
        n, nb = tb.energy.shape[0], tb.table.shape[2]
        start[j + 1] = a + n
        # The reader runs high to low; OpenMC searches these ascending.
        ascending = tb.table[::-1]
        flat_energy[a : a + n] = tb.energy[::-1]
        cdf[a : a + n, :nb] = ascending[:, CDF, :]
        elastic[a : a + n, :nb] = ascending[:, ELASTIC, :]
        fission[a : a + n, :nb] = ascending[:, FISSION, :]
        capture[a : a + n, :nb] = ascending[:, CAPTURE, :]
        smooth[j] = smooth_curves(name, label, tb.inelastic, energy)
    return UrrData(
        terms=np.array([t for t, *_ in tables], dtype=np.int64),
        start=start,
        bands=np.array([tb.table.shape[2] for *_, tb in tables], dtype=np.int64),
        energy=flat_energy,
        cdf=cdf,
        elastic=elastic,
        fission=fission,
        capture=capture,
        multiply=np.array([tb.multiply_smooth for *_, tb in tables], dtype=np.bool_),
        rule=np.array([tb.interpolation for *_, tb in tables], dtype=np.int64),
        smooth=smooth,
        points=points,
        weights=np.full(points.shape[0], 1.0 / points.shape[0]),
    )


def urr_required(nuclides):
    """Fine nodes the tables need: each table energy, where band curves bend, and node pairs just
    inside each range's ends, where they switch on."""
    out = []
    for name, label in nuclides:
        table = read_urr(name, label)
        if table is None:
            continue
        low, high = table.energy[-1], table.energy[0]
        out.append(table.energy)
        out.append(np.array([low * (1.0 + GAP), high * (1.0 - GAP)]))
    return np.concatenate(out) if out else np.empty(0)


@njit(cache=True)
def _band(cdf_row, bands, r):
    """The first band whose cdf exceeds r: upper_bound_index + 1."""
    for b in range(bands):
        if cdf_row[b] > r:
            return b
    return bands - 1


@njit(cache=True)
def in_range(energy, start, j, e):
    """Whether banded term j's tables apply at e: strictly inside, as energy_in_bounds tests."""
    return energy[start[j]] < e < energy[start[j + 1] - 1]


@njit(cache=True)
def band_xs(energy, start, bands, cdf, elastic, fission, capture, multiply, rule, j, e, r, smooth):
    """Banded term j's (elastic, capture, fission) at e for random number r, as
    calculate_urr_xs computes them. smooth holds the four smooth values at e (see UrrData)."""
    a = start[j]
    n = start[j + 1] - a
    # lower_bound_index: the table below e, or e's own table's predecessor when e is on one.
    i = 0
    if e != energy[a]:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if energy[a + mid] < e:
                lo = mid + 1
            else:
                hi = mid
        i = lo - 1
    i = min(max(i, 0), n - 2)
    low = _band(cdf[a + i], bands[j], r)
    up = _band(cdf[a + i + 1], bands[j], r)
    e0, e1 = energy[a + i], energy[a + i + 1]
    out = np.zeros(3)
    columns = (elastic, capture, fission)
    for c in range(3):
        x0 = columns[c][a + i, low]
        x1 = columns[c][a + i + 1, up]
        if rule[j] == RULE_LIN_LIN:
            f = (e - e0) / (e1 - e0)
            out[c] = (1.0 - f) * x0 + f * x1
        elif x0 > 0.0 and x1 > 0.0:
            f = np.log(e / e0) / np.log(e1 / e0)
            out[c] = np.exp((1.0 - f) * np.log(x0) + f * np.log(x1))
    if multiply[j]:
        out[0] *= smooth[0]
        out[1] *= smooth[1]
        out[2] *= smooth[2]
    for c in range(3):
        out[c] = max(out[c], 0.0)
    return out


def inside(urr, j, energy):
    """Fine nodes where banded term j's tables apply."""
    low = urr.energy[urr.start[j]]
    high = urr.energy[urr.start[j + 1] - 1]
    return (energy > low) & (energy < high)


SHIELD_TOTAL, SHIELD_ELASTIC, SHIELD_CAPTURE, SHIELD_FISSION = range(4)


def shielded(urr, shield, t, which, curve, energy):
    """A term's curve with its self-shielded values where its tables apply.

    which is one of the SHIELD_ constants; curve is the term's own curve of that kind on the grid,
    returned unchanged where no tables apply or if the term isn't banded.
    """
    if shield is None or urr.terms.shape[0] == 0:
        return curve
    hits = np.nonzero(urr.terms == t)[0]
    if hits.shape[0] == 0:
        return curve
    j = hits[0]
    out = np.array(curve, dtype=float, copy=True)
    where = inside(urr, j, energy)
    out[where] = shield[j, which, where]
    return out


def with_points(urr, points, weights):
    """The same tables with other combinations and weights, such as a product of strata."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if points.shape != (weights.shape[0], urr.terms.shape[0]):
        raise ValueError(f"points {points.shape} do not fit {urr.terms.shape[0]} banded terms")
    if abs(weights.sum() - 1.0) > 1e-12:
        raise ValueError("weights must sum to one")
    return UrrData(**{**urr.__dict__, "points": points, "weights": weights})


def product_points(per_axis, n_terms):
    """Midpoints of per_axis equal strata on every axis, all combinations, with equal weights."""
    axis = (np.arange(per_axis) + 0.5) / per_axis
    grids = np.meshgrid(*([axis] * n_terms), indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    return points, np.full(points.shape[0], 1.0 / points.shape[0])


@njit(cache=True)
def band_mean(energy, start, bands, cdf, elastic, fission, capture, multiply, rule, j, e, smooth):
    """The exact mean over r of band_xs's (elastic, capture, fission) at e, before negative values
    are clipped: the bands' probability-weighted means at the two tables, interpolated."""
    a = start[j]
    n = start[j + 1] - a
    i = 0
    if e != energy[a]:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if energy[a + mid] < e:
                lo = mid + 1
            else:
                hi = mid
        i = lo - 1
    i = min(max(i, 0), n - 2)
    out = np.zeros(3)
    columns = (elastic, capture, fission)
    f = (e - energy[a + i]) / (energy[a + i + 1] - energy[a + i])
    for c in range(3):
        for side in range(2):
            row = a + i + side
            previous = 0.0
            mean = 0.0
            for b in range(bands[j]):
                mean += (cdf[row, b] - previous) * columns[c][row, b]
                previous = cdf[row, b]
            out[c] += (1.0 - f if side == 0 else f) * mean
    if multiply[j]:
        out[0] *= smooth[0]
        out[1] *= smooth[1]
        out[2] *= smooth[2]
    return out


def node_bands(urr, j, e, smooth):
    """Every combination's (elastic, capture, fission) of banded term j at e, scaled to the exact
    mean as the march scales them: (combinations, 3)."""
    args = (
        urr.energy,
        urr.start,
        urr.bands,
        urr.cdf,
        urr.elastic,
        urr.fission,
        urr.capture,
        urr.multiply,
        urr.rule,
        j,
        e,
    )
    smooth = np.ascontiguousarray(smooth, dtype=float)
    out = np.array([band_xs(*args, r, smooth) for r in urr.points[:, j]])
    exact = band_mean(*args, smooth)
    means = urr.weights @ out
    for c in range(3):
        if means[c] > 0.0 and exact[c] > 0.0:
            out[:, c] *= exact[c] / means[c]
    return out
