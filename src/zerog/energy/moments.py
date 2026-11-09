"""Legendre moments of target-at-rest elastic scattering in the lab, per incoming energy and as
transfer rows between coarse hats.

A neutron scattering elastically at E off a target of mass ratio A at rest, at centre-of-mass
cosine mu, leaves at E' = E (A^2 + 1 + 2 A mu) / (A + 1)^2 with lab cosine
mu_lab = (1 + A mu) / sqrt(A^2 + 1 + 2 A mu). mu follows the elastic tables as OpenMC samples
them (zerog.energy.elastic.angle_pdf: the two bounding tables mixed linearly in E). Order l's
kernel is the outgoing density times P_l(mu_lab), so its rows integrate to the lab moment
<P_l(mu_lab)>, and order 0 is the ordinary kernel.

Orders above zero enter the energy subproblem as rows on inelastic transfer's coarse hats
(zerog.energy.subproblem.build_operators, order_rows), while order 0 keeps the exact window
recursion of zerog.energy.elastic. Inelastic and thermal scattering are order 0 only.
"""

import numpy as np
from numba import njit

from zerog.energy.elastic import angle_pdf, compile_angle_tables
from zerog.energy.inelastic import SparseRows, _deposit_segments, point_spacing

GAUSS_POINTS = 8  # per cosine piece
INCOMING_POINTS = 2  # per half-hat
_GX, _GW = np.polynomial.legendre.leggauss(GAUSS_POINTS)
_GX, _GW = 0.5 * (_GX + 1.0), 0.5 * _GW
_IX, _IW = np.polynomial.legendre.leggauss(INCOMING_POINTS)
_IX, _IW = 0.5 * (_IX + 1.0), 0.5 * _IW


@njit(cache=True)
def legendre(order, x):
    """P_0(x) .. P_order(x)."""
    out = np.zeros(order + 1)
    out[0] = 1.0
    if order > 0:
        out[1] = x
    for n in range(1, order):
        out[n + 1] = ((2 * n + 1) * x * out[n] - n * out[n - 1]) / (n + 1)
    return out


@njit(cache=True)
def mu_lab(mu, awr):
    return (1.0 + awr * mu) / np.sqrt(awr * awr + 1.0 + 2.0 * awr * mu)


@njit(cache=True)
def outgoing_ratio(mu, awr):
    """E' / E for centre-of-mass cosine mu."""
    return (awr * awr + 1.0 + 2.0 * awr * mu) / ((awr + 1.0) ** 2)


@njit(cache=True)
def _knots(tables, table_energy, points, mu, t, e):
    """Cosines where the pdf at e bends: both bounding tables' points, and the ends."""
    out = [-1.0, 1.0]
    if tables[t + 1] > tables[t]:
        low = tables[t]
        high = tables[t + 1] - 1
        picks = [low]
        if e >= table_energy[low]:
            picks = [low]
        elif e <= table_energy[high]:
            picks = [high]
        else:
            while high - low > 1:
                middle = (low + high) // 2
                if table_energy[middle] >= e:
                    low = middle
                else:
                    high = middle
            picks = [low, high]
        for i in picks:
            for p in range(points[i], points[i + 1]):
                if -1.0 < mu[p] < 1.0:
                    out.append(mu[p])
    knots = np.array(out)
    knots.sort()
    return knots


@njit(cache=True)
def _pdf(tables, table_energy, points, mu, pdf, t, e, x):
    if tables[t + 1] == tables[t]:
        return 0.5
    return angle_pdf(tables, table_energy, points, mu, pdf, t, e, x)


@njit(cache=True)
def point_moments(tables, table_energy, points, mu, pdf, t, awr, e, orders):
    """<P_l(mu_lab)> for l = 0 .. orders at incoming e: Gauss quadrature between the pdf's knots."""
    out = np.zeros(orders + 1)
    table_knots = _knots(tables, table_energy, points, mu, t, e)
    # mu_lab has a square-root edge at mu = -1 when A is near one: crowd pieces towards it.
    # mu_lab bends over the whole range too: eight pieces at least, whatever the tables.
    knots = np.unique(np.concatenate((table_knots, np.linspace(-1.0, 1.0, 9))))
    if awr < 1.5:
        crowd = -1.0 + 2.0 * 10.0 ** (-0.5 * np.arange(1.0, 25.0))
        knots = np.unique(np.concatenate((knots, crowd)))
    for a in range(knots.shape[0] - 1):
        lo, hi = knots[a], knots[a + 1]
        for g in range(GAUSS_POINTS):
            x = lo + (hi - lo) * _GX[g]
            w = (hi - lo) * _GW[g] * _pdf(tables, table_energy, points, mu, pdf, t, e, x)
            out += w * legendre(orders, mu_lab(x, awr))
    return out


@njit(cache=True)
def point_segments(tables, table_energy, points, mu, pdf, t, awr, e, orders, du, bottom):
    """Lab energy segments of one elastic collision at e, each with its weight per order.

    Cosine pieces are cut at the pdf's knots and so that none spans more than du in outgoing
    lethargy, down to the grid's bottom. Returns (lower, upper, weights (orders + 1, pieces), and
    each order's weight below bottom)."""
    knots = _knots(tables, table_energy, points, mu, t, e)
    lower = []
    upper = []
    rows = []
    below = np.zeros(orders + 1)
    for a in range(knots.shape[0] - 1):
        lo, hi = knots[a], knots[a + 1]
        # Lethargy of E' at each end; mu = -1 may reach E' = 0 for A = 1.
        r_lo = outgoing_ratio(lo, awr)
        r_hi = outgoing_ratio(hi, awr)
        u_hi = -np.log(r_hi)
        u_lo = -np.log(max(r_lo, bottom / e * 1e-3))
        pieces = max(1, int(np.ceil((u_lo - u_hi) / du)))
        for p in range(pieces):
            # Equal lethargy pieces, mapped back to cosine (E' is linear in mu).
            u_top = u_hi + (u_lo - u_hi) * p / pieces
            u_bot = u_hi + (u_lo - u_hi) * (p + 1) / pieces
            mu_top = min(hi, ((awr + 1.0) ** 2 * np.exp(-u_top) - awr * awr - 1.0) / (2.0 * awr))
            mu_bot = max(lo, ((awr + 1.0) ** 2 * np.exp(-u_bot) - awr * awr - 1.0) / (2.0 * awr))
            if p == pieces - 1:
                mu_bot = lo
            if p == 0:
                mu_top = hi
            if mu_top <= mu_bot:
                continue
            w = np.zeros(orders + 1)
            for g in range(GAUSS_POINTS):
                x = mu_bot + (mu_top - mu_bot) * _GX[g]
                q = (
                    (mu_top - mu_bot)
                    * _GW[g]
                    * _pdf(tables, table_energy, points, mu, pdf, t, e, x)
                )
                w += q * legendre(orders, mu_lab(x, awr))
            e_top = e * outgoing_ratio(mu_top, awr)
            e_bot = e * outgoing_ratio(mu_bot, awr)
            if e_top <= bottom:
                below += w
                continue
            if e_bot < bottom:
                # Split the piece's weight by lethargy, as deposit spreads it.
                share = np.log(e_top / bottom) / np.log(e_top / max(e_bot, 1e-300))
                below += w * (1.0 - share)
                w = w * share
                e_bot = bottom
            lower.append(e_bot)
            upper.append(e_top)
            rows.append(w)
    n = len(rows)
    weights = np.zeros((orders + 1, n))
    for p in range(n):
        weights[:, p] = rows[p]
    return np.array(lower), np.array(upper), weights, below


def lab_moments(angle, awr, energy, orders):
    """<P_l(mu_lab)> per incoming energy, (energies, orders + 1), for AngleTables or None."""
    kernels = compile_angle_tables([angle], tolerance=-1.0)
    args = (kernels.tables, kernels.table_energy, kernels.points, kernels.mu, kernels.pdf, 0)
    return np.array([point_moments(*args, awr, float(e), orders) for e in energy])


def _hat_points(coarse, j, curve_energy, curve):
    """Gauss energies and weights over hat j, weighted by the hat and the cross section."""
    grid_u = np.log(coarse[0] / coarse)
    us, ws = [], []
    for side in (j - 1, j):
        if side < 0 or side + 1 >= coarse.shape[0]:
            continue
        a, b = grid_u[side], grid_u[side + 1]
        u = a + (b - a) * _IX
        shape = (u - a) / (b - a) if side == j - 1 else (b - u) / (b - a)
        us.append(u)
        ws.append((b - a) * _IW * shape)
    u = np.concatenate(us)
    e = coarse[0] * np.exp(-u)
    sigma = np.interp(e[::-1], curve_energy[::-1], curve[::-1])[::-1]
    return e, np.concatenate(ws) * np.clip(sigma, 0.0, None)


def elastic_order_rows(angle, awr, curve_energy, curve, coarse, orders, du=None):
    """Rows of elastic orders 1 .. orders on coarse hats, as (SparseRows, lost) per order.

    Row j is per neutron emitted from hat j, the hat-and-cross-section weighted mean of single
    collisions at Gauss points over the hat; its sum is the lab moment <P_l>, not one, less what
    lands below the grid. curve is the elastic cross section on curve_energy (descending), the
    part taken at rest where thermal scattering applies.
    """
    du = point_spacing(coarse) if du is None else du
    kernels = compile_angle_tables([angle], tolerance=-1.0)
    args = (kernels.tables, kernels.table_energy, kernels.points, kernels.mu, kernels.pdf, 0)
    grid_u = np.log(coarse[0] / coarse)
    n = coarse.shape[0]
    values = [[] for _ in range(orders)]
    first = np.zeros((orders, n), dtype=np.int64)
    last = np.zeros((orders, n), dtype=np.int64)
    lost = np.zeros((orders, n))
    for j in range(n):
        e, w = _hat_points(coarse, j, curve_energy, curve)
        keep = w > 0.0
        if not keep.any():
            for order in range(orders):
                values[order].append(np.empty(0))
            continue
        w = w[keep] / w[keep].sum()
        total = np.zeros((orders, n))
        off = np.zeros(orders)
        for e_p, w_p in zip(e[keep], w, strict=True):
            low, high, weights, below = point_segments(
                *args, awr, float(e_p), orders, du, coarse[-1]
            )
            a = np.log(coarse[0] / high)
            b = np.log(coarse[0] / low)
            for order in range(1, orders + 1):
                row, outside = _deposit_segments(a, b, np.ascontiguousarray(weights[order]), grid_u)
                total[order - 1] += w_p * row
                off[order - 1] += w_p * (outside + below[order])
        for order in range(orders):
            nonzero = np.nonzero(total[order])[0]
            lost[order, j] = off[order]
            if nonzero.shape[0] == 0:
                values[order].append(np.empty(0))
                continue
            first[order, j], last[order, j] = nonzero[0], nonzero[-1] + 1
            values[order].append(total[order, first[order, j] : last[order, j]])
    out = []
    for order in range(orders):
        starts = np.zeros(n, dtype=np.int64)
        offset = 0
        for j in range(n):
            starts[j] = offset
            offset += values[order][j].shape[0]
        flat = np.concatenate(values[order]) if offset else np.empty(0)
        out.append((SparseRows(starts, first[order], last[order], flat), lost[order]))
    return out
