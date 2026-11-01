"""Thin nuclide energy grids and merge them into a union grid. Energies run high to low."""

import numpy as np
from numba import njit

from zerog.data.neutron import (
    MT_CAPTURE,
    MT_ELASTIC,
    MT_FISSION,
    MT_PARTIAL_FISSION,
    read_reactions,
)


@njit(cache=True)
def _span_within_tolerance(x, curves, start, end, rtol, atol):
    width = x[end] - x[start]
    for c in range(curves.shape[0]):
        y0 = curves[c, start]
        slope = (curves[c, end] - y0) / width
        for k in range(start + 1, end):
            exact = curves[c, k]
            if abs(y0 + slope * (x[k] - x[start]) - exact) > rtol * abs(exact) + atol:
                return False
    return True


@njit(cache=True)
def thin_indices(x, curves, required, rtol, atol):
    """Indices of a subset of x on which lin-lin interpolation reproduces every curve.

    x is strictly monotonic and curves is (curves, points). Every dropped point k satisfies
    |interpolated - curves[c, k]| <= rtol * |curves[c, k]| + atol for all c. Points where required
    is true, and both end points, are always kept.
    """
    n = x.shape[0]
    next_required = np.empty(n, dtype=np.int64)
    limit = n - 1
    for i in range(n - 1, -1, -1):
        if required[i]:
            limit = i
        next_required[i] = limit

    kept = np.empty(n, dtype=np.int64)
    kept[0] = 0
    count = 1
    i = 0
    while i < n - 1:
        limit = next_required[i + 1]
        # Gallop to bracket the span end, then bisect, so cost grows with the span, not its square.
        good = i + 1
        bad = -1
        width = 2
        while good < limit:
            j = min(i + width, limit)
            if _span_within_tolerance(x, curves, i, j, rtol, atol):
                good = j
                width *= 2
            else:
                bad = j
                break
        while bad - good > 1:
            middle = (good + bad) // 2
            if _span_within_tolerance(x, curves, i, middle, rtol, atol):
                good = middle
            else:
                bad = middle
        kept[count] = good
        count += 1
        i = good
    return kept[:count]


def thin_nuclide(name, temperature, rtol, atol, energy_max=2.0e7):
    """A nuclide's thinned grid at or below energy_max, valid for each of its reactions.

    Valid only at this temperature, and it does not cover nu or secondary distributions. A
    non-negative sum of the reactions, such as the total, keeps the relative bound, plus atol per
    reaction. Each threshold point is kept, so that a reaction stays zero below its threshold.
    """
    return thin_reactions(read_reactions(name, temperature), rtol, atol, energy_max)


def thin_reactions(data, rtol, atol, energy_max=2.0e7):
    """The thinned grid of a ReactionCurves, as described in thin_nuclide."""
    start = np.searchsorted(-data.energy, -energy_max)
    energy = data.energy[start:]
    required = np.zeros(energy.shape[0], dtype=np.bool_)
    thresholds = data.threshold[data.threshold <= energy_max]
    required[np.searchsorted(-energy, -thresholds)] = True
    return energy[thin_indices(energy, data.xs[:, start:], required, rtol, atol)]


def major_curves(data):
    """Total, elastic, capture and fission from ReactionCurves, as (4, energies).

    Fission includes the partial reactions, which replace a redundant MT 18 for some nuclides.
    """
    groups = ((MT_ELASTIC,), (MT_CAPTURE,), (MT_FISSION, *MT_PARTIAL_FISSION))
    curves = np.zeros((4, data.energy.shape[0]))
    curves[0] = data.xs.sum(axis=0)
    for g, mts in enumerate(groups, start=1):
        curves[g] = data.xs[np.isin(data.mt, mts)].sum(axis=0)
    return curves


def union_grid(grids):
    """Merge descending grids.

    Evaluate each nuclide on the result through its own thinned grid: added points then lie on the
    same interpolant, so its error against the library does not grow.
    """
    return np.unique(np.concatenate(grids))[::-1]
