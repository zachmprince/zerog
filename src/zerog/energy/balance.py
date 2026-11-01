"""Particle balance of an energy solution, using exact integrals of its interpolants."""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from zerog.energy.grid import _GAUSS_S, _GAUSS_W, _weights, panel_rate


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


@njit(cache=True)
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
        r = _weights(h, False, nodes, node_weights)
        e = _weights(h, True, nodes, node_weights)
        rate = sigma_upper * (r[0] * flux_upper + r[1] * flux[i + 1]) + sigma[i + 1] * (
            r[2] * flux_upper + r[3] * flux[i + 1]
        )
        emission = sigma_upper * (e[0] * flux_upper + e[1] * flux[i + 1]) + sigma[i + 1] * (
            e[2] * flux_upper + e[3] * flux[i + 1]
        )
        total += (bottom * emission / upper - alpha * rate) / (1.0 - alpha)
    return total


def energy_balance(ops, removal, scatter, streaming, source, flux):
    """Balance of a solve_energy result; arguments as there."""
    removal = np.asarray(removal, dtype=float)
    scatter = np.asarray(scatter, dtype=float)
    unit = np.ones(flux.shape[0])
    src = panel_rate(np.ones_like(flux), np.asarray(source, dtype=float), ops.rate_weights)
    removed = streaming * panel_rate(unit, flux, ops.rate_weights)
    emitted = 0.0
    lost = 0.0
    for t in range(ops.removal.shape[0]):
        removed += removal[t] * panel_rate(ops.removal[t], flux, ops.rate_weights)
        emitted += scatter[t, 0] * panel_rate(ops.scatter[t], flux, ops.rate_weights)
        lost += scatter[t, 0] * _lost_below(
            ops.energy, ops.scatter[t], ops.alpha[t], flux, _GAUSS_S, _GAUSS_W
        )
    return Balance(src, emitted, removed, lost, src + emitted - removed - lost)
