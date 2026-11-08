"""Target-at-rest elastic scattering, isotropic in the centre of mass, in flux per unit lethargy F.

The emission density per unit lethargy at E, from a nuclide with scattering cross section sigma, is

    Y(E) = E / (1 - alpha) * integral over E < E' < E / alpha of sigma(E') F(u') / E' du'

The march holds that integral as a running one from the top of the grid, so each term costs O(1) per
node, plus the search for the window's lower edge. The edge splits a panel, and the piece is still
lin-lin, so the window integral is exact for the interpolants. The largest panel must be narrower
than each window, so that the edge never falls in the panel holding the unknown.

elastic_row is the plain-Python oracle: a direct sum, independent of the running integral.

Anisotropy multiplies the integrand by 2 f(mu | E'), with mu = (2 E / E' - 1 - alpha) / (1 - alpha).
The march keeps the recursion above for the isotropic part and adds 2 f - 1, which has zero mean,
over the window. separable_correction does that in O(table knots) per node from three running
moments; anisotropic_correction, a direct sum over the window's panels, is its quadrature oracle.
Both stop at the energy where a term's tables become flat. Like OpenMC
v0.16.0 (AngleDistribution::sample, get_energy_index and Tabular::sample_unbiased), f is the mix of
the two bounding tables, with weights linear in E', each lin-lin in mu. OpenMC samples mu at the lab
energy, and at or above the free gas threshold the target is at rest, as it is here.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from zerog.energy.grid import emission_weights, panel_rate, rate_weights

# A table within this of 1/2 everywhere counts as flat. Fe56 and U238 hold noise near 1e-4 at low
# energy, which would otherwise keep the direct sum running over the whole range.
FLAT_TOLERANCE = 1e-3
RULE_LIN_LIN = 2
_GAUSS3_X = (0.1127016653792583, 0.5, 0.8872983346207417)
_GAUSS3_W = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)
PIECES_PER_PEAK = 8  # quadrature pieces per window, per unit of the local peak of f
WIDE_WINDOW = 40.0  # stands in for an infinite window, when alpha is zero
PEAK_RESOLUTION = 0.05  # largest panel, as a fraction of the window over the local peak of f
# A segment of the separable form whose terms sum to more than this times its isotropic integral
# would lose that factor in round-off, and is integrated by quadrature instead.
CANCELLATION_LIMIT = 64.0


@dataclass(frozen=True)
class AngleKernels:
    """Elastic mu tables of all terms, flat. Term t's tables span tables[t]:tables[t + 1].

    Table i's points span points[i]:points[i + 1] of mu and pdf. table_energy runs high to low and
    can repeat. Direct sums run only above cutoff[t], which is infinite for an isotropic term.
    """

    tables: np.ndarray
    table_energy: np.ndarray
    points: np.ndarray
    mu: np.ndarray
    pdf: np.ndarray
    cutoff: np.ndarray
    peak: np.ndarray  # largest pdf per table, at least 1


def compile_angle_tables(terms, tolerance=FLAT_TOLERANCE):
    """Flatten a list of AngleTables, or None for an isotropic term, into AngleKernels."""
    tables = [0]
    energies = []
    points = [0]
    mus = []
    pdfs = []
    peaks = []
    cutoff = np.full(len(terms), np.inf)
    for t, angle in enumerate(terms):
        if angle is None:
            tables.append(tables[-1])
            continue
        if np.any(angle.interpolation != RULE_LIN_LIN):
            raise NotImplementedError("elastic angle tables that are not lin-lin")
        energies.append(angle.energy)
        mus.append(angle.mu)
        pdfs.append(angle.pdf)
        peaks.append(np.maximum(np.maximum.reduceat(angle.pdf, angle.offsets[:-1]), 1.0))
        lengths = np.diff(angle.offsets)
        points.extend((points[-1] + np.cumsum(lengths)).tolist())
        tables.append(tables[-1] + angle.energy.shape[0])
        deviation = np.maximum.reduceat(np.abs(angle.pdf - 0.5), angle.offsets[:-1])
        steep = np.nonzero(deviation > tolerance)[0]
        if steep.shape[0] > 0:
            # The mix between tables is linear in E, so its deviation is bounded by the line
            # between theirs, which crosses the tolerance inside the last steep interval.
            last = steep[-1]
            if last == angle.energy.shape[0] - 1:
                cutoff[t] = angle.energy[last]
            else:
                upper, lower = angle.energy[last], angle.energy[last + 1]
                share = (tolerance - deviation[last + 1]) / (deviation[last] - deviation[last + 1])
                cutoff[t] = lower + (upper - lower) * max(share, 0.0)
    empty = np.empty(0)
    return AngleKernels(
        tables=np.array(tables, dtype=np.int64),
        table_energy=np.concatenate(energies) if energies else empty,
        points=np.array(points, dtype=np.int64),
        mu=np.concatenate(mus) if mus else empty,
        pdf=np.concatenate(pdfs) if pdfs else empty,
        cutoff=cutoff,
        peak=np.concatenate(peaks) if peaks else empty,
    )


@njit(cache=True)
def _table_pdf(mu, x, points, values, i):
    """Table i's lin-lin pdf at cosine x."""
    low = points[i]
    high = points[i + 1] - 1
    if x <= mu[low]:
        return values[low]
    if x >= mu[high]:
        return values[high]
    while high - low > 1:
        middle = (low + high) // 2
        if mu[middle] <= x:
            low = middle
        else:
            high = middle
    return values[low] + (values[high] - values[low]) * (x - mu[low]) / (mu[high] - mu[low])


@njit(cache=True)
def angle_pdf(tables, table_energy, points, mu, pdf, t, energy, x):
    """f(x | energy) of term t: the mix of its two bounding tables, held constant past the ends."""
    low = tables[t]
    high = tables[t + 1] - 1
    if energy >= table_energy[low]:
        return _table_pdf(mu, x, points, pdf, low)
    if energy <= table_energy[high]:
        return _table_pdf(mu, x, points, pdf, high)
    while high - low > 1:
        middle = (low + high) // 2
        if table_energy[middle] >= energy:
            low = middle
        else:
            high = middle
    r = (table_energy[low] - energy) / (table_energy[low] - table_energy[high])
    return (1.0 - r) * _table_pdf(mu, x, points, pdf, low) + r * _table_pdf(
        mu, x, points, pdf, high
    )


@njit(cache=True)
def _local_peak(tables, table_energy, peak, t, energy):
    """The larger peak of the tables bounding an energy."""
    low = tables[t]
    high = tables[t + 1] - 1
    if energy >= table_energy[low]:
        return peak[low]
    if energy <= table_energy[high]:
        return peak[high]
    while high - low > 1:
        middle = (low + high) // 2
        if table_energy[middle] >= energy:
            low = middle
        else:
            high = middle
    return max(peak[low], peak[high])


def refine_for_angle(energy, alpha, angle, resolution=PEAK_RESOLUTION):
    """Add nodes so that no panel is wide against a forward peak, for each term's AngleTables.

    F is linear on a panel, so a node couples to itself through about (peak of f) * h / window of
    its own emission. Quadrature can't repair that: the panel has to resolve the peak.
    """
    kernels = compile_angle_tables(angle, tolerance=-1.0)
    limits = np.full(energy.shape[0] - 1, np.inf)
    for t in range(len(angle)):
        if angle[t] is None:
            continue
        width = -math.log(alpha[t]) if alpha[t] > 0.0 else WIDE_WINDOW
        for i in range(limits.shape[0]):
            peak = _local_peak(kernels.tables, kernels.table_energy, kernels.peak, t, energy[i])
            limits[i] = min(limits[i], resolution * width / peak)
    widths = np.log(energy[:-1] / energy[1:])
    pieces = np.ceil(widths / limits * (1.0 + 1e-12)).astype(np.int64)
    out = [
        energy[i] * np.exp(-widths[i] * np.arange(pieces[i]) / pieces[i])
        for i in range(energy.shape[0] - 1)
    ]
    out.append(energy[-1:])
    return np.concatenate(out)


@njit(cache=True)
def anisotropic_correction(
    energy,
    k,
    sigma,
    flux,
    alpha,
    width,
    start,
    edge,
    cutoff,
    t,
    tables,
    table_energy,
    points,
    mu,
    pdf,
    peak,
):
    """Integral of sigma F (2 f - 1) / E' du' over node k's window, split as (known, own, mean).

    own is the coefficient of F[k]. mean is quadrature error: the same integral without sigma F,
    with f frozen at this node's energy, where it is zero by change of variable. With f varying in
    E' the true integral is not zero, so this must not use the f above. The caller removes mean,
    because any net gain per collision compounds over lethargy, by more as forward peaks shrink the
    mean lethargy gain. Quadrature pieces follow the local peak of f, so the integral holds on any
    panel, though F needs refine_for_angle to follow it.
    """
    known = 0.0
    own = 0.0
    mean = 0.0
    target = energy[k]
    for i in range(start, k):
        upper = energy[i]
        if upper <= cutoff:
            break
        lower = energy[i + 1]
        h = math.log(upper / lower)
        first = math.log(upper / edge) if upper > edge else 0.0
        local = _local_peak(tables, table_energy, peak, t, upper)
        pieces = 1 + int((h - first) * PIECES_PER_PEAK * local / width)
        step = (h - first) / pieces
        for p in range(pieces):
            for q in range(3):
                s = first + (p + _GAUSS3_X[q]) * step
                e = upper * math.exp(-s)
                g = (upper - e) / (upper - lower)
                x = (2.0 * target / e - 1.0 - alpha) / (1.0 - alpha)
                f = angle_pdf(tables, table_energy, points, mu, pdf, t, e, x)
                frozen = angle_pdf(tables, table_energy, points, mu, pdf, t, target, x)
                mean += _GAUSS3_W[q] * step * (2.0 * frozen - 1.0) / e
                kernel = _GAUSS3_W[q] * step * (2.0 * f - 1.0) / e
                value = kernel * ((1.0 - g) * sigma[i] + g * sigma[i + 1])
                r = s / h
                known += value * (1.0 - r) * flux[i]
                if i + 1 == k:
                    own += value * r
                else:
                    known += value * r * flux[i + 1]
    return known, own, mean


def point_emission(energy, source_energy, alpha, angle=None):
    """Emission per unit lethargy at nodes, from one collision at source_energy, per neutron.

    angle is a term's AngleTables, or None for isotropic. Put node_pairs at alpha * source_energy,
    where this jumps. It is normalized to one neutron on this grid.
    """
    inside = (energy <= source_energy) & (energy >= alpha * source_energy)
    out = np.zeros_like(energy)
    density = np.ones_like(energy)
    if angle is not None:
        kernels = compile_angle_tables([angle], tolerance=-1.0)
        x = (2.0 * energy / source_energy - 1.0 - alpha) / (1.0 - alpha)
        density = np.array(
            [
                2.0
                * angle_pdf(
                    kernels.tables,
                    kernels.table_energy,
                    kernels.points,
                    kernels.mu,
                    kernels.pdf,
                    0,
                    source_energy,
                    v,
                )
                for v in x
            ]
        )
    out[inside] = (energy[inside] / source_energy * density[inside]) / (1.0 - alpha)
    # Balance integrates this same interpolant, so it can't see a strength error in a sharp peak.
    return out / panel_rate(np.ones_like(energy), out, rate_weights(energy))


def table_moments(angle, i, orders):
    """Exact Legendre moments of table i, the integral of P_l(mu) f(mu) dmu, for l < orders."""
    mu = angle.mu[angle.offsets[i] : angle.offsets[i + 1]]
    pdf = angle.pdf[angle.offsets[i] : angle.offsets[i + 1]]
    nodes, weights = np.polynomial.legendre.leggauss(orders // 2 + 2)
    half = 0.5 * np.diff(mu)[:, None]
    points = 0.5 * (mu[1:] + mu[:-1])[:, None] + half * nodes
    values = pdf[:-1, None] + (pdf[1:] - pdf[:-1])[:, None] * (points - mu[:-1, None]) / (
        2.0 * half
    )
    basis = np.polynomial.legendre.legvander(points, orders - 1)
    return np.einsum("ij,ijl,j,i->l", values, basis, weights, half[:, 0])


def lab_cosine(energy_ratio, awr):
    """Lab scattering cosine at rest-target energy ratio E / E', for elastic scattering."""
    root = np.sqrt(energy_ratio)
    return 0.5 * ((awr + 1.0) * root - (awr - 1.0) / root)


def window_widths(alpha):
    """Width in lethargy of each elastic window, per term. Infinite where alpha is zero."""
    with np.errstate(divide="ignore"):
        return -np.log(np.asarray(alpha, dtype=float))


def elastic_row(energy, sigma, alpha, n):
    """Coefficients c of Y(energy[n]) = sum c[m] F[m], for one term, by direct sum."""
    weights = emission_weights(energy)
    row = np.zeros(energy.shape[0])
    edge = energy[n] / alpha if alpha > 0.0 else math.inf
    factor = energy[n] / (1.0 - alpha)
    for i in range(n):
        upper = energy[i]
        lower = energy[i + 1]
        if upper <= edge:
            w = weights[i] / upper
            row[i] += factor * (sigma[i] * w[0] + sigma[i + 1] * w[2])
            row[i + 1] += factor * (sigma[i] * w[1] + sigma[i + 1] * w[3])
        elif lower < edge:
            # Only the part of the panel below the edge is in the window.
            s = math.log(edge / lower)
            h = math.log(upper / lower)
            f_cut = s / h  # weight of F[i] at the edge
            g_cut = (edge - lower) / (upper - lower)  # weight of sigma[i] at the edge
            sigma_edge = g_cut * sigma[i] + (1.0 - g_cut) * sigma[i + 1]
            w = emission_weights(np.array([edge, lower]))[0] / edge
            row[i] += factor * f_cut * (sigma_edge * w[0] + sigma[i + 1] * w[2])
            row[i + 1] += factor * (
                (1.0 - f_cut) * (sigma_edge * w[0] + sigma[i + 1] * w[2])
                + sigma_edge * w[1]
                + sigma[i + 1] * w[3]
            )
    return row


# The separable form of the anisotropic window. Between two table energies the mix weights are
# linear in E', and on a segment of both tables' mu points each pdf is linear in mu, which is linear
# in E / E'. So there (2 f - 1) / E' is a combination of 1, 1 / E' and 1 / E'^2, with coefficients
# depending on E, and the window integral is a sum over segments of the running moments
# M_n(E') = integral of sigma F / E'^n du from the block's start. The segments' edges are the
# table energies and the knots of both tables mapped to E', so a node costs O(knots in its window)
# instead of O(panels in it), and the integral is exact for the interpolants, as the oracle's
# quadrature is only in the limit.


@njit(cache=True)
def _last_at_or_above(energy, i, e):
    """The last node at or above e, searching on from node i, which must be at or above it."""
    n = energy.shape[0]
    low = i
    high = i + 1
    step = 1
    while high < n and energy[high] >= e:
        low = high
        high = low + step
        step *= 2
    high = min(high, n)
    while high - low > 1:
        middle = (low + high) // 2
        if energy[middle] >= e:
            low = middle
        else:
            high = middle
    return low


@njit(cache=True)
def _moments_at(energy, sigma, flux, moments, previous, block_start, k, i, e, weights):
    """M_0, M_1, M_2 at e in panel i (i < k), as node i's stored value, the known part of the
    piece from node i down to e, and that piece's coefficient of F[k], each for the three.

    The stored value is returned apart so that differences between energies over the same node
    cancel it exactly: it carries round-off from its whole block, which steep segments amplify.
    moments holds the three per node, relative to each node's block; nodes before block_start
    belong to the previous block, which ended at previous. weights are the panel weights for
    powers 0, 1 and 2 of E_i / E', as (3, panels, 4).
    """
    k0 = k1 = k2 = 0.0
    o0 = o1 = o2 = 0.0
    upper = energy[i]
    lower = energy[i + 1]
    if i < block_start:
        k0 = moments[0, i] - previous[0]
        k1 = moments[1, i] - previous[1]
        k2 = moments[2, i] - previous[2]
    else:
        k0 = moments[0, i]
        k1 = moments[1, i]
        k2 = moments[2, i]
    if e >= upper:
        return k0, k1, k2, 0.0, 0.0, 0.0, o0, o1, o2
    a = 0.0  # coefficients of F[i] and F[i + 1] for each power
    b = 0.0
    a1 = b1 = a2 = b2 = 0.0
    if e <= lower:
        s0 = sigma[i]
        s1 = sigma[i + 1]
        w = weights[0, i]
        a = s0 * w[0] + s1 * w[2]
        b = s0 * w[1] + s1 * w[3]
        w = weights[1, i]
        a1 = (s0 * w[0] + s1 * w[2]) / upper
        b1 = (s0 * w[1] + s1 * w[3]) / upper
        w = weights[2, i]
        a2 = (s0 * w[0] + s1 * w[2]) / (upper * upper)
        b2 = (s0 * w[1] + s1 * w[3]) / (upper * upper)
    else:
        h = math.log(upper / lower)
        span = math.log(upper / e)
        pieces = 1 + int(span / 0.05)
        step = span / pieces
        for p in range(pieces):
            for q in range(3):
                s = (p + _GAUSS3_X[q]) * step
                x = upper * math.exp(-s)
                g = (upper - x) / (upper - lower)
                r = s / h
                v = _GAUSS3_W[q] * step * ((1.0 - g) * sigma[i] + g * sigma[i + 1])
                inverse = 1.0 / x
                a += v * (1.0 - r)
                b += v * r
                a1 += v * (1.0 - r) * inverse
                b1 += v * r * inverse
                a2 += v * (1.0 - r) * inverse * inverse
                b2 += v * r * inverse * inverse
    p0 = a * flux[i]
    p1 = a1 * flux[i]
    p2 = a2 * flux[i]
    if i + 1 == k:
        o0, o1, o2 = b, b1, b2
    else:
        p0 += b * flux[i + 1]
        p1 += b1 * flux[i + 1]
        p2 += b2 * flux[i + 1]
    return k0, k1, k2, p0, p1, p2, o0, o1, o2


@njit(cache=True)
def _segment(mu, pdf, points, table, g):
    """Table pdf on segment g as p0 + slope (x - m0), held constant past the ends."""
    low = points[table]
    high = points[table + 1] - 1
    if g < low:
        return pdf[low], 0.0, 0.0
    if g >= high:
        return pdf[high], 0.0, 0.0
    slope = (pdf[g + 1] - pdf[g]) / (mu[g + 1] - mu[g])
    return pdf[g], slope, mu[g]


@njit(cache=True)
def _first_segment(mu, points, table, x):
    """Last point of a table at or below x, or one before the table if none is."""
    g = points[table] - 1
    while g + 1 < points[table + 1] and mu[g + 1] <= x:
        g += 1
    return g


@njit(cache=True)
def _piece_quadrature(energy, sigma, flux, k, i, top, bottom, target, c, d, upper_mix, lower_mix):
    """Integral of sigma F (2 f - 1) / E' du' from top down to bottom, starting in panel i, as
    (known, coefficient of F[k]). f is the mix of two linear pdfs, each given as
    (A, B, p0, slope, m0): weight A + B E' times p0 + slope (x - m0). It is smooth, so three Gauss
    points per piece of a panel reach round-off.
    """
    a_a, b_a, p_a, m_a, mu_a = upper_mix
    a_b, b_b, p_b, m_b, mu_b = lower_mix
    known = 0.0
    own = 0.0
    while i < k and energy[i] > bottom:
        upper = energy[i]
        lower = energy[i + 1]
        h = math.log(upper / lower)
        first = math.log(upper / top) if top < upper else 0.0
        end = math.log(upper / bottom) if bottom > lower else h
        pieces = 1 + int((end - first) / 0.05)
        step = (end - first) / pieces
        for p in range(pieces):
            for q in range(3):
                s = first + (p + _GAUSS3_X[q]) * step
                e = upper * math.exp(-s)
                x = c * target / e + d
                f = (a_a + b_a * e) * (p_a + m_a * (x - mu_a)) + (a_b + b_b * e) * (
                    p_b + m_b * (x - mu_b)
                )
                g = (upper - e) / (upper - lower)
                kernel = _GAUSS3_W[q] * step * (2.0 * f - 1.0) / e
                value = kernel * ((1.0 - g) * sigma[i] + g * sigma[i + 1])
                r = s / h
                known += value * (1.0 - r) * flux[i]
                if i + 1 == k:
                    own += value * r
                else:
                    known += value * r * flux[i + 1]
        i += 1
    return known, own


@njit(cache=True)
def separable_correction(
    energy,
    k,
    sigma,
    flux,
    alpha,
    start,
    bottom,
    moments,
    previous,
    block_start,
    weights,
    t,
    tables,
    table_energy,
    points,
    mu,
    pdf,
):
    """anisotropic_correction's (known, own) by the separable form, exactly for the interpolants.

    The window runs from node k's edge, or the grid's top, down to energy[k] or bottom, whichever
    is higher. start is a panel at or above the edge. moments must be complete for nodes before k.
    On steep segments, near a forward peak, and between close table energies the moments' terms
    cancel. Where they would amplify round-off past CANCELLATION_LIMIT, relative to the segment's
    isotropic integral, the segment is integrated by quadrature over its panels instead.
    """
    target = energy[k]
    high_e = min(target / alpha, energy[0]) if alpha > 0.0 else energy[0]
    low_e = max(target, bottom)
    known = 0.0
    own = 0.0
    if high_e <= low_e:
        return known, own
    c = 2.0 / (1.0 - alpha)
    d = -(1.0 + alpha) / (1.0 - alpha)
    i = min(_last_at_or_above(energy, start, high_e), k - 1)
    v0, v1, v2, q0, q1, q2, w0, w1, w2 = _moments_at(
        energy, sigma, flux, moments, previous, block_start, k, i, high_e, weights
    )
    # F[k] is unknown, so its neighbour stands in to size the cancellation.
    guess = abs(flux[k - 1])
    # Table intervals from the top of the window down: j is the table above, j + 1 below.
    first = tables[t]
    last = tables[t + 1] - 1
    j = first - 1
    while j + 1 <= last and table_energy[j + 1] >= high_e:
        j += 1
    upper_e = high_e
    while upper_e > low_e:
        if j < first:
            lower_e = max(low_e, table_energy[first])
            s_a, s_b = first, first
            a_a, b_a, a_b, b_b = 1.0, 0.0, 0.0, 0.0
        elif j >= last:
            lower_e = low_e
            s_a, s_b = last, last
            a_a, b_a, a_b, b_b = 1.0, 0.0, 0.0, 0.0
        else:
            lower_e = max(low_e, table_energy[j + 1])
            s_a, s_b = j, j + 1
            span = table_energy[j] - table_energy[j + 1]
            # Weights A + B E' of the upper and lower tables, linear in E'.
            a_a, b_a = -table_energy[j + 1] / span, 1.0 / span
            a_b, b_b = table_energy[j] / span, -1.0 / span
        if lower_e < upper_e:
            x = c * target / upper_e + d
            x_end = c * target / lower_e + d
            g_a = _first_segment(mu, points, s_a, x)
            g_b = _first_segment(mu, points, s_b, x)
            piece_top = upper_e
            while True:
                next_a = mu[g_a + 1] if g_a + 1 < points[s_a + 1] else np.inf
                next_b = mu[g_b + 1] if g_b + 1 < points[s_b + 1] else np.inf
                x_next = min(next_a, next_b, x_end)
                if x_next > x:
                    piece_bottom = lower_e if x_next >= x_end else c * target / (x_next - d)
                    piece_bottom = min(max(piece_bottom, lower_e), piece_top)
                    p_a, m_a, mu_a = _segment(mu, pdf, points, s_a, g_a)
                    p_b, m_b, mu_b = _segment(mu, pdf, points, s_b, g_b)
                    # Each pdf is P + Q / E' on the piece.
                    big_p_a = p_a + m_a * (d - mu_a)
                    big_q_a = m_a * c * target
                    big_p_b = p_b + m_b * (d - mu_b)
                    big_q_b = m_b * c * target
                    k0 = 2.0 * (b_a * big_p_a + b_b * big_p_b)
                    k1 = 2.0 * (a_a * big_p_a + b_a * big_q_a + a_b * big_p_b + b_b * big_q_b) - 1.0
                    k2 = 2.0 * (a_a * big_q_a + a_b * big_q_b)
                    i_top = i
                    i = min(_last_at_or_above(energy, i, piece_bottom), k - 1)
                    u0, u1, u2, r0, r1, r2, z0, z1, z2 = _moments_at(
                        energy,
                        sigma,
                        flux,
                        moments,
                        previous,
                        block_start,
                        k,
                        i,
                        piece_bottom,
                        weights,
                    )
                    t0 = k0 * ((u0 - v0) + (r0 - q0))
                    t1 = k1 * ((u1 - v1) + (r1 - q1))
                    t2 = k2 * ((u2 - v2) + (r2 - q2))
                    o0 = k0 * (z0 - w0)
                    o1 = k1 * (z1 - w1)
                    o2 = k2 * (z2 - w2)
                    size = abs(t0) + abs(t1) + abs(t2) + (abs(o0) + abs(o1) + abs(o2)) * guess
                    isotropic = abs((u1 - v1) + (r1 - q1)) + abs(z1 - w1) * guess
                    if size > CANCELLATION_LIMIT * isotropic:
                        part, part_own = _piece_quadrature(
                            energy,
                            sigma,
                            flux,
                            k,
                            i_top,
                            piece_top,
                            piece_bottom,
                            target,
                            c,
                            d,
                            (a_a, b_a, p_a, m_a, mu_a),
                            (a_b, b_b, p_b, m_b, mu_b),
                        )
                        known += part
                        own += part_own
                    else:
                        known += t0 + t1 + t2
                        own += o0 + o1 + o2
                    v0, v1, v2, q0, q1, q2, w0, w1, w2 = u0, u1, u2, r0, r1, r2, z0, z1, z2
                    piece_top = piece_bottom
                    x = x_next
                if x_next >= x_end:
                    break
                if next_a <= x_next:
                    g_a += 1
                if next_b <= x_next:
                    g_b += 1
        upper_e = lower_e
        j += 1
    return known, own
