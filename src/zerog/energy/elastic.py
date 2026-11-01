"""Target-at-rest elastic scattering, isotropic in the centre of mass, in flux per unit lethargy F.

The emission density per unit lethargy at E, from a nuclide with scattering cross section sigma, is

    Y(E) = E / (1 - alpha) * integral over E < E' < E / alpha of sigma(E') F(u') / E' du'

The march holds that integral as a running one from the top of the grid, so each term costs O(1) per
node, plus the search for the window's lower edge. The edge splits a panel, and the piece is still
lin-lin, so the window integral is exact for the interpolants. The largest panel must be narrower
than each window, so that the edge never falls in the panel holding the unknown.

elastic_row is the plain-Python oracle: a direct sum, independent of the running integral.
"""

import math

import numpy as np

from zerog.energy.grid import emission_weights


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
