"""Energy grid and lethargy panel weights. Energies run high to low; lethargy u = ln(energy[0] / E).

On a panel between nodes i and i + 1, a cross section is linear in E and the flux per unit lethargy
F is linear in u, which is what lin-lin data and a constant F need to be exact. Weights hold the
integrals of (sigma basis) * (F basis) over the panel, in the order (sigma_i, F_i), (sigma_i,
F_i+1), (sigma_i+1, F_i), (sigma_i+1, F_i+1).
"""

import math

import numpy as np
from numba import njit

from zerog.data.grid import thin_nuclide, union_grid

# Exact for these integrands, to round-off, on panels up to MAX_PANEL wide.
_GAUSS_S, _GAUSS_W = np.polynomial.legendre.leggauss(12)
_GAUSS_S = 0.5 * (_GAUSS_S + 1.0)
_GAUSS_W = 0.5 * _GAUSS_W
MAX_PANEL = 1.0


@njit(cache=True)
def _weights(h, power, nodes, node_weights):
    """Panel weights for width h, with a factor (E_i / E) ** power in the integrand."""
    w00 = w01 = w10 = w11 = 0.0
    denominator = math.expm1(-h)
    for q in range(nodes.shape[0]):
        t = nodes[q]
        s = t * h
        g = math.expm1(-s) / denominator
        v = node_weights[q] * h
        if power != 0:
            v *= math.exp(power * s)
        w00 += v * (1.0 - g) * (1.0 - t)
        w01 += v * (1.0 - g) * t
        w10 += v * g * (1.0 - t)
        w11 += v * g * t
    return w00, w01, w10, w11


@njit(cache=True)
def _panel_weights(energy, power, nodes, node_weights):
    n = energy.shape[0] - 1
    out = np.empty((n, 4))
    for i in range(n):
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = _weights(
            math.log(energy[i] / energy[i + 1]), power, nodes, node_weights
        )
    return out


def rate_weights(energy):
    """Weights of the integral of sigma(E) F(u) du per panel, as (panels, 4)."""
    return _panel_weights(energy, 0, _GAUSS_S, _GAUSS_W)


def emission_weights(energy):
    """Weights of the integral of sigma(E) F(u) (E_i / E) du per panel, as (panels, 4).

    Divide by the panel's upper energy E_i to get the integral with 1 / E in the integrand.
    """
    return _panel_weights(energy, 1, _GAUSS_S, _GAUSS_W)


def square_weights(energy):
    """Weights of the integral of sigma(E) F(u) (E_i / E) ** 2 du per panel, as (panels, 4).

    Divide by the square of the panel's upper energy to get the integral with 1 / E ** 2.
    """
    return _panel_weights(energy, 2, _GAUSS_S, _GAUSS_W)


def panel_rate(sigma, flux, weights):
    """The integral of sigma(E) F(u) du over the grid, from rate_weights."""
    return np.sum(
        sigma[:-1] * (weights[:, 0] * flux[:-1] + weights[:, 1] * flux[1:])
        + sigma[1:] * (weights[:, 2] * flux[:-1] + weights[:, 3] * flux[1:])
    )


def kinematic_alpha(awr):
    """Largest fractional energy loss complement in target-at-rest elastic scattering."""
    return ((awr - 1.0) / (awr + 1.0)) ** 2


def limit_panels(energy, max_du):
    """Add nodes geometrically so no panel is wider than max_du in lethargy.

    Added nodes are not on a data curve until the curve is interpolated there, which stays on its
    lin-lin interpolant.
    """
    widths = np.log(energy[:-1] / energy[1:])
    pieces = np.ceil(widths / max_du * (1.0 + 1e-12)).astype(np.int64)
    out = [
        energy[i] * np.exp(-widths[i] * np.arange(pieces[i]) / pieces[i])
        for i in range(energy.shape[0] - 1)
    ]
    out.append(energy[-1:])
    return np.concatenate(out)


def assemble_grid(grids, energy_max, energy_min, max_du, required=()):
    """Merge descending grids and required energies between energy_max and energy_min.

    max_du must be below the narrowest elastic window of every term: see zerog.energy.elastic.
    """
    merged = union_grid([*grids, np.array([energy_max, energy_min]), np.asarray(required)])
    merged = merged[(merged <= energy_max) & (merged >= energy_min)]
    return limit_panels(merged, min(max_du, MAX_PANEL))


def build_grid(nuclides, rtol, atol, energy_max, energy_min, max_du, required=()):
    """Grid covering all reactions of (name, temperature label) pairs; see assemble_grid."""
    grids = [thin_nuclide(name, label, rtol, atol, energy_max) for name, label in nuclides]
    return assemble_grid(grids, energy_max, energy_min, max_du, required)


def node_pairs(energies, gap=1e-10):
    """Two nodes at each energy where a source or cross section jumps, the lower one by a gap.

    F is continuous, so a jump between nodes is smeared across a panel and changes its strength,
    which makes the flux below first order in panel width. Pass these as required energies.
    """
    energies = np.asarray(energies, dtype=float)
    return np.concatenate([energies, energies * (1.0 - gap)])


def interpolate(energy, table_energy, table_values):
    """A descending lin-lin table on descending energy. Zero below the table, an error above it."""
    if energy[0] > table_energy[0]:
        raise ValueError(f"grid reaches {energy[0]} eV, above the table's top {table_energy[0]} eV")
    return np.interp(energy[::-1], table_energy[::-1], table_values[::-1], left=0.0)[::-1]
