"""Particle balance of an energy solution, using exact integrals of its interpolants."""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from zerog.energy.elastic import angle_pdf
from zerog.energy.grid import _GAUSS_S, _GAUSS_W, _weights, panel_rate
from zerog.energy.inelastic import restrict

BELOW_POINTS = 64  # midpoints across the cosines that land below the grid


@dataclass(frozen=True)
class Balance:
    """Neutrons per unit time, all from integrals in lethargy, not normalized to the source.

    residual is source + emitted - removed - lost_below. Collocation satisfies the equation at nodes
    only, so the residual is truncation error, which falls with panel width. It is not round-off.
    It is signed and sums over energy, so a small one is necessary but does not certify a rate.
    """

    source: float
    emitted: float
    removed: float
    lost_below: float
    residual: float


# Not cached, as it calls across files: see zerog.energy.subproblem._march.
@njit
def _lost_below(energy, sigma, alpha, flux, nodes, node_weights):
    """Scattering from one term landing below the grid's bottom, target at rest and isotropic.

    P(E -> below bottom) = (bottom / E - alpha) / (1 - alpha), for E < bottom / alpha.
    """
    bottom = energy[-1]
    edge = bottom / alpha if alpha > 0.0 else np.inf
    total = 0.0
    for i in range(energy.shape[0] - 1):
        upper = energy[i]
        lower = energy[i + 1]
        if lower >= edge:
            continue
        sigma_upper = sigma[i]
        flux_upper = flux[i]
        if upper > edge:
            # Only the piece below the edge can land below the bottom.
            g = (edge - lower) / (upper - lower)
            t = math.log(upper / edge) / math.log(upper / lower)
            sigma_upper = g * sigma[i] + (1.0 - g) * sigma[i + 1]
            flux_upper = (1.0 - t) * flux[i] + t * flux[i + 1]
            upper = edge
        h = math.log(upper / lower)
        r = _weights(h, 0, nodes, node_weights)
        e = _weights(h, 1, nodes, node_weights)
        rate = sigma_upper * (r[0] * flux_upper + r[1] * flux[i + 1]) + sigma[i + 1] * (
            r[2] * flux_upper + r[3] * flux[i + 1]
        )
        emission = sigma_upper * (e[0] * flux_upper + e[1] * flux[i + 1]) + sigma[i + 1] * (
            e[2] * flux_upper + e[3] * flux[i + 1]
        )
        total += (bottom * emission / upper - alpha * rate) / (1.0 - alpha)
    return total


# Not cached, as it calls across files: see zerog.energy.subproblem._march.
@njit
def _lost_below_anisotropic(
    energy, sigma, alpha, flux, cutoff, t, tables, table_energy, points, mu, pdf
):
    """The part of _lost_below from 2 f - 1, for incoming energies above the term's cutoff."""
    bottom = energy[-1]
    edge = bottom / alpha if alpha > 0.0 else np.inf
    x_nodes = (0.1127016653792583, 0.5, 0.8872983346207417)
    x_weights = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)
    total = 0.0
    for i in range(energy.shape[0] - 1):
        upper = energy[i]
        lower = energy[i + 1]
        if upper <= cutoff:
            break
        if lower >= edge:
            continue
        h = math.log(upper / lower)
        first = math.log(upper / edge) if upper > edge else 0.0
        for q in range(3):
            s = first + x_nodes[q] * (h - first)
            e = upper * math.exp(-s)
            g = (upper - e) / (upper - lower)
            r = s / h
            rate = ((1.0 - g) * sigma[i] + g * sigma[i + 1]) * (
                (1.0 - r) * flux[i] + r * flux[i + 1]
            )
            # The chance of landing below the bottom, less its isotropic value.
            limit = (2.0 * bottom / e - 1.0 - alpha) / (1.0 - alpha)
            below = 0.0
            for m in range(BELOW_POINTS):
                x = -1.0 + (m + 0.5) * (limit + 1.0) / BELOW_POINTS
                below += angle_pdf(tables, table_energy, points, mu, pdf, t, e, x) - 0.5
            total += x_weights[q] * (h - first) * rate * below * (limit + 1.0) / BELOW_POINTS
    return total


def energy_balance(ops, removal, scatter, streaming, source, flux, shield=None):
    """Balance of a solve_energy result; arguments as there. shield is the solution's, for a
    problem with probability tables: removal and elastic emission then take the bands' shielded
    cross sections where the tables apply."""
    removal = np.asarray(removal, dtype=float)
    scatter = np.asarray(scatter, dtype=float)
    total_xs = ops.removal
    elastic_xs = ops.scatter
    if shield is None and ops.urr.terms.shape[0]:
        raise ValueError("this problem has probability tables: pass the solution's shield")
    if shield is not None and ops.urr.terms.shape[0]:
        from zerog.energy.urr import SHIELD_ELASTIC, SHIELD_TOTAL, shielded

        total_xs = np.array(
            [
                shielded(ops.urr, shield, t, SHIELD_TOTAL, c, ops.energy)
                for t, c in enumerate(total_xs)
            ]
        )
        elastic_xs = np.array(
            [
                shielded(ops.urr, shield, t, SHIELD_ELASTIC, c, ops.energy)
                for t, c in enumerate(elastic_xs)
            ]
        )
    unit = np.ones(flux.shape[0])
    src = panel_rate(np.ones_like(flux), np.asarray(source, dtype=float), ops.rate_weights)
    removed = streaming * panel_rate(unit, flux, ops.rate_weights)
    emitted = 0.0
    lost = 0.0
    for t in range(ops.removal.shape[0]):
        removed += removal[t] * panel_rate(total_xs[t], flux, ops.rate_weights)
        emitted += scatter[t, 0] * panel_rate(elastic_xs[t], flux, ops.rate_weights)
        lost += scatter[t, 0] * _lost_below(
            ops.energy, elastic_xs[t], ops.alpha[t], flux, _GAUSS_S, _GAUSS_W
        )
        if np.isfinite(ops.angle.cutoff[t]):
            lost += scatter[t, 0] * _lost_below_anisotropic(
                ops.energy,
                elastic_xs[t],
                ops.alpha[t],
                flux,
                ops.angle.cutoff[t],
                t,
                ops.angle.tables,
                ops.angle.table_energy,
                ops.angle.points,
                ops.angle.mu,
                ops.angle.pdf,
            )
    kernels = ops.inelastic
    if kernels.values.shape[0] > 0:
        restricted = restrict(kernels, ops.energy, flux, ops.rate_weights)
        for t in range(ops.removal.shape[0]):
            emitted += scatter[t, 0] * restricted[t].sum()
            lost += scatter[t, 0] * float(restricted[t] @ kernels.lost[t])
    return Balance(src, emitted, removed, lost, src + emitted - removed - lost)
