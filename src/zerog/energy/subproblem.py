"""The PGD energy subproblem, in flux per unit lethargy F = E * epsilon(E) at the grid's nodes:

    [a + sum_t d_R[t] sigma_R[t](E)] F(u) = sum_t,l d_S[t, l] Y[t, l](E) + q(u)

Y[t, l] is the emission density per unit lethargy of term t and Legendre order l, so q is a source
per unit lethargy as well. Densities and the streaming coefficient a can be negative. The equation
is collocated at the nodes, and the implicit part of each node's own panel moves to the left.

Only order 0 of target-at-rest elastic scattering exists so far.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from zerog.energy.elastic import (
    WIDE_WINDOW,
    AngleKernels,
    anisotropic_correction,
    compile_angle_tables,
    window_widths,
)
from zerog.energy.grid import MAX_PANEL, emission_weights, rate_weights


@dataclass(frozen=True)
class EnergyOperators:
    """Data for the subproblem. Arrays are (terms, energies), or (terms) for alpha."""

    energy: np.ndarray
    removal: np.ndarray  # barns
    scatter: np.ndarray  # barns
    alpha: np.ndarray
    rate_weights: np.ndarray
    emission_weights: np.ndarray
    angle: AngleKernels


@dataclass(frozen=True)
class EnergySolution:
    """F at the nodes, and each node's solved diagonal in 1/cm.

    A diagonal at or near zero means a singular subproblem at that node. It can happen with negative
    densities, and it is reported here, not treated.
    """

    flux: np.ndarray
    diagonal: np.ndarray


def build_operators(energy, removal, scatter, alpha, angle=None):
    """Check a grid against its terms, and collect the panel weights.

    angle is a list of AngleTables per term, with None for an isotropic one. None means all are.
    """
    energy = np.asarray(energy, dtype=float)
    if np.any(np.diff(energy) >= 0.0):
        raise ValueError("energy must strictly decrease")
    alpha = np.asarray(alpha, dtype=float)
    removal = np.asarray(removal, dtype=float)
    scatter = np.asarray(scatter, dtype=float)
    expected = (alpha.shape[0], energy.shape[0])
    if alpha.ndim != 1 or removal.shape != expected or scatter.shape != expected:
        raise ValueError(
            f"shape: removal {removal.shape} and scatter {scatter.shape} must be {expected}"
        )
    widest = np.log(energy[:-1] / energy[1:]).max()
    if widest > MAX_PANEL:
        raise ValueError(f"panel of {widest} wider than the quadrature's limit, {MAX_PANEL}")
    narrowest = window_widths(alpha).min()
    if widest >= narrowest:
        raise ValueError(f"panel of {widest} wider than the narrowest elastic window, {narrowest}")
    return EnergyOperators(
        energy=energy,
        removal=removal,
        scatter=scatter,
        alpha=alpha,
        rate_weights=rate_weights(energy),
        emission_weights=emission_weights(energy),
        angle=compile_angle_tables([None] * alpha.shape[0] if angle is None else angle),
    )


@njit(cache=True, error_model="numpy")
def _edge_integral(energy, sigma, flux, running, j, edge):
    """The running integral of sigma F / E du at an energy edge inside panel j."""
    upper = energy[j]
    lower = energy[j + 1]
    g = (edge - lower) / (upper - lower)
    sigma_edge = g * sigma[j] + (1.0 - g) * sigma[j + 1]
    h = np.log(upper / lower)
    s = np.log(upper / edge)
    t = s / h
    flux_edge = (1.0 - t) * flux[j] + t * flux[j + 1]
    # Lin-lin holds on the piece. Three Gauss points per 0.05 of lethargy reach round-off.
    total = 0.0
    x = (0.1127016653792583, 0.5, 0.8872983346207417)
    w = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)
    denominator = np.expm1(-s) if s > 0.0 else -1.0
    pieces = 1 + int(s / 0.05)
    width = s / pieces
    for p in range(pieces):
        for q in range(3):
            v = (p + x[q]) * width
            r = v / s if s > 0.0 else 0.0
            gamma = np.expm1(-v) / denominator
            sigma_v = (1.0 - gamma) * sigma[j] + gamma * sigma_edge
            flux_v = (1.0 - r) * flux[j] + r * flux_edge
            total += w[q] * width * sigma_v * flux_v * np.exp(v) / upper
    return running[j] + total


# Not cached: Numba's cache does not notice changes to a callee in another file, so a cached caller
# would keep running the old kernel.
@njit(error_model="numpy")
def _march(
    energy,
    removal,
    scatter,
    alpha,
    weights,
    d_removal,
    d_scatter,
    streaming,
    source,
    tables,
    table_energy,
    points,
    mu,
    pdf,
    cutoff,
    peak,
):
    """Forward substitution over the nodes.

    Each term's running integral restarts once its block is a window wide. A window then spans at
    most the current block and the previous one, so the subtraction cancels only local magnitudes
    and F keeps its relative accuracy when it falls faster than E.
    """
    n_terms = removal.shape[0]
    n = energy.shape[0]
    flux = np.empty(n)
    diagonal = np.empty(n)
    running = np.zeros((n_terms, n))
    pointer = np.zeros(n_terms, dtype=np.int64)
    block_start = np.zeros(n_terms, dtype=np.int64)
    previous_total = np.zeros(n_terms)
    width = np.array([-np.log(a) if a > 0.0 else WIDE_WINDOW for a in alpha])
    for k in range(n):
        lhs = streaming
        rhs = source[k]
        for t in range(n_terms):
            lhs += d_removal[t] * removal[t, k]
            if k == 0:
                continue
            sigma = scatter[t]
            w = weights[k - 1]
            if k > 1 and energy[k - 1] <= alpha[t] * energy[block_start[t]]:
                previous_total[t] = running[t, k - 1]
                running[t, k - 1] = 0.0
                block_start[t] = k - 1
            i = 1.0 / energy[k - 1]
            known = running[t, k - 1] + i * flux[k - 1] * (sigma[k - 1] * w[0] + sigma[k] * w[2])
            own = i * (sigma[k - 1] * w[1] + sigma[k] * w[3])
            factor = energy[k] / (1.0 - alpha[t])
            window = known
            if alpha[t] > 0.0:
                edge = energy[k] / alpha[t]
                if edge < energy[0]:
                    j = pointer[t]
                    while energy[j + 1] >= edge:
                        j += 1
                    pointer[t] = j
                    # running[t, j] may belong to the previous block, which still holds its values
                    # below the node where it ended.
                    part = _edge_integral(energy, sigma, flux, running[t], j, edge)
                    window += -part if j >= block_start[t] else previous_total[t] - part
                else:
                    window += previous_total[t]
            if energy[pointer[t]] > cutoff[t]:
                extra, extra_own, mean = anisotropic_correction(
                    energy,
                    k,
                    sigma,
                    flux,
                    alpha[t],
                    width[t],
                    pointer[t],
                    energy[k] / alpha[t] if alpha[t] > 0.0 else np.inf,
                    cutoff[t],
                    t,
                    tables,
                    table_energy,
                    points,
                    mu,
                    pdf,
                    peak,
                )
                window += extra
                own += extra_own
                if alpha[t] > 0.0 and energy[k] / alpha[t] < energy[0] and energy[k] > cutoff[t]:
                    # The whole window is on the grid and above the cutoff, so this should vanish.
                    # A window crossing either is truly cut short; see the kernel.
                    own -= mean * sigma[k]
            lhs -= d_scatter[t] * factor * own
            rhs += d_scatter[t] * factor * window
            running[t, k] = known  # completed below, once F[k] is known
        diagonal[k] = lhs
        flux[k] = rhs / lhs
        if k > 0:
            for t in range(n_terms):
                sigma = scatter[t]
                w = weights[k - 1]
                running[t, k] += flux[k] * (sigma[k - 1] * w[1] + sigma[k] * w[3]) / energy[k - 1]
    return flux, diagonal


def solve_energy(ops, removal, scatter, streaming, source):
    """Solve for F. removal is (terms), scatter is (terms, orders), source is (energies)."""
    removal = np.asarray(removal, dtype=float)
    scatter = np.asarray(scatter, dtype=float)
    source = np.asarray(source, dtype=float)
    n_terms, n = ops.removal.shape
    if scatter.ndim == 2 and scatter.shape[1] != 1 and scatter.shape[0] == n_terms:
        raise NotImplementedError("only order 0 scattering exists so far")
    if removal.shape != (n_terms,) or scatter.shape != (n_terms, 1) or source.shape != (n,):
        raise ValueError(
            f"shape: removal {removal.shape}, scatter {scatter.shape} and source {source.shape} "
            f"do not fit {n_terms} terms and {n} energies"
        )
    flux, diagonal = _march(
        ops.energy,
        ops.removal,
        ops.scatter,
        ops.alpha,
        ops.emission_weights,
        removal,
        scatter[:, 0],
        float(streaming),
        source,
        ops.angle.tables,
        ops.angle.table_energy,
        ops.angle.points,
        ops.angle.mu,
        ops.angle.pdf,
        ops.angle.cutoff,
        ops.angle.peak,
    )
    return EnergySolution(flux, diagonal)
