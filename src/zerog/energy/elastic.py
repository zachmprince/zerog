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
over the window. separable_correction does that in O(table knots x log panels) per node from
moments held in a tree over panels; anisotropic_correction, a direct sum over the window's panels,
is its quadrature oracle.
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
# in E / E'. So there (2 f - 1) / E' is a quadratic in z = E_r / E', for any reference energy E_r.
# The segments' edges are the table energies and the knots of both tables mapped to E', so a node
# costs O(knots in its window), and the integral is exact for the interpolants.
#
# Expanded about z = 1, at a reference E_r inside the segment, the quadratic's terms are each
# bounded by the kernel's own size over the range they act on, so nothing cancels however steep
# the pdf or close the tables. Its coefficients multiply moments of sigma F (z - 1)^n, n = 0, 1, 2,
# which must be centred at the same E_r and taken over panels no wider than the segment. A binary
# tree over panels holds them: each node's moments are centred at its top energy, and a segment's
# whole panels are O(log panels) nodes inside it. Partial panels at its ends, and the panel holding
# the unknown, take Gauss quadrature of f itself. Earlier running moments in the basis 1, 1 / E',
# 1 / E'^2 cancelled on steep segments and between close tables, and above about 100 keV most
# segments fell back to quadrature over all their panels, O(window) per node.


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
def moment_tree_size(n_nodes):
    """Leaves in a moment tree over a grid's panels: the next power of two."""
    size = 1
    while size < max(n_nodes - 1, 1):
        size *= 2
    return size


@njit(cache=True)
def moment_tree_add(tree, size, energy, sigma, flux, weights, p):
    """Store panel p's centred moments as a leaf of tree (2 size, 4), once F is known on it, and
    complete each node that it ends. Panels must be added in order.

    Node v at height h holds panels (v << h) - size onwards, 2^h of them, with moments of
    sigma F (z - 1)^n du about its top energy in columns 0 to 2, and that energy's inverse in 3.
    weights are centred_weights, as (3, panels, 4).
    """
    a = flux[p]
    b = flux[p + 1]
    s0 = sigma[p]
    s1 = sigma[p + 1]
    v = size + p
    for q in range(3):
        w = weights[q, p]
        tree[v, q] = s0 * (w[0] * a + w[1] * b) + s1 * (w[2] * a + w[3] * b)
    tree[v, 3] = 1.0 / energy[p]
    h = 0
    while v > 1 and (v & 1) == 1:
        # v ends its parent, whose right half it is. Its moments move from its top to the parent's:
        # z_parent - 1 = rho (z - 1) + delta, with rho the ratio of their top energies.
        parent = v >> 1
        h += 1
        first = (parent << h) - size
        upper = energy[first]
        lower = energy[first + (1 << (h - 1))]
        rho = upper / lower
        delta = (upper - lower) / lower
        r0 = tree[v, 0]
        r1 = tree[v, 1]
        r2 = tree[v, 2]
        tree[parent, 0] = tree[v - 1, 0] + r0
        tree[parent, 1] = tree[v - 1, 1] + rho * r1 + delta * r0
        tree[parent, 2] = (
            tree[v - 1, 2] + rho * rho * r2 + 2.0 * rho * delta * r1 + delta * delta * r0
        )
        tree[parent, 3] = tree[v - 1, 3]
        v = parent


@njit(cache=True)
def _node_value(m0, m1, m2, inverse, e_r, ct, d, mix_a, mix_b):
    """Integral of sigma F (2 f - 1) / E' over a tree node with centred moments m0, m1, m2 about
    its top energy e_r, of inverse 1 / e_r, for one segment's f.

    Each table contributes its weight A + B E' times its pdf P + Q / E', with Q = slope c target.
    About e_r, z (A + B E') = w_r + A (z - 1) and P + Q / E' = pi_r + (Q / e_r) (z - 1), where w_r
    and pi_r are the weight and pdf at e_r, taken directly.
    """
    a_a, b_a, p_a, m_a, mu_a = mix_a
    a_b, b_b, p_b, m_b, mu_b = mix_b
    x_r = ct * inverse + d
    w_a = a_a + b_a * e_r
    w_b = a_b + b_b * e_r
    pi_a = p_a + m_a * (x_r - mu_a)
    pi_b = p_b + m_b * (x_r - mu_b)
    q_a = m_a * ct * inverse
    q_b = m_b * ct * inverse
    k0 = 2.0 * (w_a * pi_a + w_b * pi_b) - 1.0
    k1 = 2.0 * (a_a * pi_a + w_a * q_a + a_b * pi_b + w_b * q_b) - 1.0
    k2 = 2.0 * (a_a * q_a + a_b * q_b)
    return (k0 * m0 + k1 * m1 + k2 * m2) * inverse


@njit(cache=True)
def _exp_minus(s):
    """exp(-s), by its series for the small offsets inside panels, exact there to round-off."""
    if 0.0 <= s < 0.01:
        return 1.0 - s * (
            1.0
            - s
            * (
                0.5
                - s
                * (
                    1.0 / 6.0
                    - s * (1.0 / 24.0 - s * (1.0 / 120.0 - s * (1.0 / 720.0 - s / 5040.0)))
                )
            )
        )
    return math.exp(-s)


@njit(cache=True)
def _piece_quadrature(
    upper, lower, h, sigma0, sigma1, flux0, flux1, unknown, first, end, ct, d, mix_a, mix_b
):
    """Integral of sigma F (2 f - 1) / E' du' over a panel of width h in lethargy, from first to
    end below its top, as (known, coefficient of the lower node's F if unknown).

    f is the mix of two linear pdfs, each given as (A, B, p0, slope, m0): weight A + B E' times
    p0 + slope (x - m0). It is smooth, so three Gauss points per 0.05 of lethargy reach round-off.
    Taking scalars, not arrays, spares the reference counting of a call per piece.
    """
    a_a, b_a, p_a, m_a, mu_a = mix_a
    a_b, b_b, p_b, m_b, mu_b = mix_b
    known = 0.0
    own = 0.0
    span = 1.0 / (upper - lower)
    width = 1.0 / h
    pieces = 1 + int((end - first) / 0.05)
    step = (end - first) / pieces
    for p in range(pieces):
        for q in range(3):
            s = first + (p + _GAUSS3_X[q]) * step
            e = upper * _exp_minus(s)
            inverse = 1.0 / e
            x = ct * inverse + d
            f = (a_a + b_a * e) * (p_a + m_a * (x - mu_a)) + (a_b + b_b * e) * (
                p_b + m_b * (x - mu_b)
            )
            g = (upper - e) * span
            kernel = _GAUSS3_W[q] * step * (2.0 * f - 1.0) * inverse
            value = kernel * ((1.0 - g) * sigma0 + g * sigma1)
            r = s * width
            known += value * (1.0 - r) * flux0
            if unknown:
                own += value * r
            else:
                known += value * r * flux1
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
    widths,
    tree,
    size,
    t,
    tables,
    table_energy,
    points,
    mu,
    pdf,
):
    """anisotropic_correction's (known, own) by the separable form, exactly for the interpolants.

    The window runs from node k's edge, or the grid's top, down to energy[k] or bottom, whichever
    is higher. start is a panel at or above the edge. widths are the panels' widths in lethargy.
    tree holds the centred moments of every panel above k - 1, from moment_tree_add, with size
    leaves.

    Each segment's whole panels above node k - 1 come from the tree. Partial panels at either end,
    and panel k - 1, whose lower node is the unknown, take quadrature. Searches and table lookups
    are written out here, since a call passing arrays costs reference counting per segment.
    """
    target = energy[k]
    n = energy.shape[0]
    high_e = min(target / alpha, energy[0]) if alpha > 0.0 else energy[0]
    low_e = max(target, bottom)
    known = 0.0
    own = 0.0
    if high_e <= low_e:
        return known, own
    c = 2.0 / (1.0 - alpha)
    d = -(1.0 + alpha) / (1.0 - alpha)
    ct = c * target
    # The segment's top lies in panel i, first_u below its top.
    i = min(_last_at_or_above(energy, start, high_e), k - 1)
    first_u = math.log(energy[i] / high_e) if high_e < energy[i] else 0.0
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
            x = ct / upper_e + d
            x_end = ct / lower_e + d
            # Each table's last point at or below x, or one before the table if none is.
            low_a = points[s_a] - 1
            high_a = points[s_a + 1]
            while high_a - low_a > 1:
                middle = (low_a + high_a) // 2
                if mu[middle] <= x:
                    low_a = middle
                else:
                    high_a = middle
            g_a = low_a
            low_b = points[s_b] - 1
            high_b = points[s_b + 1]
            while high_b - low_b > 1:
                middle = (low_b + high_b) // 2
                if mu[middle] <= x:
                    low_b = middle
                else:
                    high_b = middle
            g_b = low_b
            end_a = points[s_a + 1] - 1
            end_b = points[s_b + 1] - 1
            piece_top = upper_e
            while True:
                next_a = mu[g_a + 1] if g_a + 1 < points[s_a + 1] else np.inf
                next_b = mu[g_b + 1] if g_b + 1 < points[s_b + 1] else np.inf
                x_next = min(next_a, next_b, x_end)
                if x_next > x:
                    piece_bottom = lower_e if x_next >= x_end else ct / (x_next - d)
                    piece_bottom = min(max(piece_bottom, lower_e), piece_top)
                    if piece_bottom < piece_top:
                        # Each table's pdf on the segment as p0 + slope (x - m0), held constant
                        # past its ends.
                        if g_a < points[s_a]:
                            mix_a = (a_a, b_a, pdf[points[s_a]], 0.0, 0.0)
                        elif g_a >= end_a:
                            mix_a = (a_a, b_a, pdf[end_a], 0.0, 0.0)
                        else:
                            slope = (pdf[g_a + 1] - pdf[g_a]) / (mu[g_a + 1] - mu[g_a])
                            mix_a = (a_a, b_a, pdf[g_a], slope, mu[g_a])
                        if g_b < points[s_b]:
                            mix_b = (a_b, b_b, pdf[points[s_b]], 0.0, 0.0)
                        elif g_b >= end_b:
                            mix_b = (a_b, b_b, pdf[end_b], 0.0, 0.0)
                        else:
                            slope = (pdf[g_b + 1] - pdf[g_b]) / (mu[g_b + 1] - mu[g_b])
                            mix_b = (a_b, b_b, pdf[g_b], slope, mu[g_b])
                        # The panel holding the segment's bottom: the last node at or above it.
                        lower = i
                        step = 1
                        while lower + step < n and energy[lower + step] >= piece_bottom:
                            lower += step
                            step *= 2
                        while step > 1:
                            step //= 2
                            if lower + step < n and energy[lower + step] >= piece_bottom:
                                lower += step
                        lower = min(lower, k - 1)
                        last_u = (
                            math.log(energy[lower] / piece_bottom)
                            if piece_bottom < energy[lower]
                            else 0.0
                        )
                        if i == lower:
                            part, part_own = _piece_quadrature(
                                energy[i],
                                energy[i + 1],
                                widths[i],
                                sigma[i],
                                sigma[i + 1],
                                flux[i],
                                flux[i + 1],
                                i + 1 == k,
                                first_u,
                                last_u,
                                ct,
                                d,
                                mix_a,
                                mix_b,
                            )
                            known += part
                            own += part_own
                        else:
                            whole = i
                            if first_u > 0.0:
                                part, _ = _piece_quadrature(
                                    energy[i],
                                    energy[i + 1],
                                    widths[i],
                                    sigma[i],
                                    sigma[i + 1],
                                    flux[i],
                                    flux[i + 1],
                                    False,
                                    first_u,
                                    widths[i],
                                    ct,
                                    d,
                                    mix_a,
                                    mix_b,
                                )
                                known += part
                                whole = i + 1
                            if last_u > 0.0:
                                part, part_own = _piece_quadrature(
                                    energy[lower],
                                    energy[lower + 1],
                                    widths[lower],
                                    sigma[lower],
                                    sigma[lower + 1],
                                    flux[lower],
                                    flux[lower + 1],
                                    lower + 1 == k,
                                    0.0,
                                    last_u,
                                    ct,
                                    d,
                                    mix_a,
                                    mix_b,
                                )
                                known += part
                                own += part_own
                            # Whole panels whole:lower, over the fewest tree nodes covering them.
                            v_low = whole + size
                            v_high = lower + size
                            h = 0
                            while v_low < v_high:
                                if v_low & 1:
                                    known += _node_value(
                                        tree[v_low, 0],
                                        tree[v_low, 1],
                                        tree[v_low, 2],
                                        tree[v_low, 3],
                                        energy[(v_low << h) - size],
                                        ct,
                                        d,
                                        mix_a,
                                        mix_b,
                                    )
                                    v_low += 1
                                if v_high & 1:
                                    v_high -= 1
                                    known += _node_value(
                                        tree[v_high, 0],
                                        tree[v_high, 1],
                                        tree[v_high, 2],
                                        tree[v_high, 3],
                                        energy[(v_high << h) - size],
                                        ct,
                                        d,
                                        mix_a,
                                        mix_b,
                                    )
                                v_low >>= 1
                                v_high >>= 1
                                h += 1
                        i = lower
                        first_u = last_u
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


def thin_angle(angle, tolerance):
    """AngleTables with each table's cosines thinned so lin-lin interpolation stays within
    tolerance of the pdf, an absolute bound (a pdf integrates to one over a width of two, so its
    area moves by at most twice that), and the pdf renormalized to one (cdf recomputed). A bound
    relative to the peak instead lets a forward peak of 67 (U238 at 30 MeV) shift the area, and so
    the whole table, by 5%.

    The march's anisotropic correction costs about one segment per cosine piece in each node's
    window, so this is its main cost lever; None or a non-positive tolerance returns angle.
    """
    if angle is None or tolerance is None or tolerance <= 0.0:
        return angle
    from zerog.data.grid import thin_indices
    from zerog.data.scatter import AngleTables

    offsets = [0]
    mus, pdfs, cdfs = [], [], []
    for i in range(angle.energy.shape[0]):
        mu = angle.mu[angle.offsets[i] : angle.offsets[i + 1]]
        pdf = angle.pdf[angle.offsets[i] : angle.offsets[i + 1]]
        if angle.interpolation[i] == RULE_LIN_LIN and mu.shape[0] > 2:
            keep = thin_indices(
                mu,
                pdf[None, :].copy(),
                np.zeros(mu.shape[0], dtype=np.bool_),
                0.0,
                tolerance,
            )
            mu, pdf = mu[keep], pdf[keep]
        if angle.interpolation[i] == RULE_LIN_LIN:
            area = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(mu))])
            pdf = pdf / area[-1]
            cdf = area / area[-1]
        else:
            cdf = angle.cdf[angle.offsets[i] : angle.offsets[i + 1]]
        mus.append(mu)
        pdfs.append(pdf)
        cdfs.append(cdf)
        offsets.append(offsets[-1] + mu.shape[0])
    return AngleTables(
        energy=angle.energy,
        offsets=np.array(offsets, dtype=np.int64),
        interpolation=angle.interpolation,
        mu=np.concatenate(mus),
        pdf=np.concatenate(pdfs),
        cdf=np.concatenate(cdfs),
    )
