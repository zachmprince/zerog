"""The PGD energy subproblem, in flux per unit lethargy F = E * epsilon(E) at the grid's nodes:

    [a + sum_t d_R[t] sigma_R[t](E)] F(u) = sum_t,l d_S[t, l] Y[t, l](E) + q(u)

Y[t, l] is the emission density per unit lethargy of term t and Legendre order l, so q is a source
per unit lethargy as well. Y[t, l] carries sigma_s times the kernel's P_l(mu_lab) moment, with no
(2l + 1) / 2: a projection's (2l + 1) / 2 belongs in d_S[t, l], as study 06 builds it.
Densities and the streaming coefficient a can be negative. The equation is collocated at the
nodes, and the implicit part of each node's own panel moves to the left.

Orders above zero enter as pseudo-terms with transfer rows (build_operators, order_rows).
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from zerog.energy.elastic import (
    WIDE_WINDOW,
    AngleKernels,
    anisotropic_correction,
    compile_angle_tables,
    moment_tree_add,
    moment_tree_size,
    separable_correction,
    window_widths,
)
from zerog.energy.grid import MAX_PANEL, centred_weights, emission_weights, rate_weights
from zerog.energy.inelastic import InelasticKernels, no_inelastic, panel_hats
from zerog.energy.urr import UrrData, band_mean, band_xs, in_range, no_urr
from zerog.linalg import solve

MAX_SWEEPS = 50
SWEEP_TOLERANCE = 1e-12  # on the hats' emission, relative to the largest
BLOCK_RESTARTS = 20  # GMRES restart cycles per block solve
BLOCK_RTOL = 1e-10  # GMRES stop relative to the block's right-hand side
UPSCATTER_SHARE = 1e-6  # share of a hat's row above its upper neighbour that calls for the block
# The anisotropic elastic window: the direct sum (the oracle), the moment tree with quadrature of f
# on every cut panel (decision 008), and the tree with each knot's cut shared between its segments.
ANISOTROPY_MODES = {"direct": 0, "tree": 1, "separable": 2}


@dataclass(frozen=True)
class EnergyOperators:
    """Data for the subproblem. Arrays are (terms, energies), or (terms) for alpha."""

    energy: np.ndarray
    removal: np.ndarray  # barns
    scatter: np.ndarray  # barns
    alpha: np.ndarray
    rate_weights: np.ndarray
    emission_weights: np.ndarray
    moment_weights: np.ndarray  # powers 0, 1 and 2 of E_i / E - 1, as (3, panels, 4)
    angle: AngleKernels
    inelastic: InelasticKernels
    panel_m: np.ndarray
    panel_share: np.ndarray
    node_m: np.ndarray
    node_theta: np.ndarray
    urr: UrrData
    # Orders above zero, as pseudo-terms of a second set of operators (build_operators).
    orders: int = 0
    extended: "EnergyOperators | None" = None


@dataclass(frozen=True)
class EnergySolution:
    """F at the nodes, each node's solved diagonal in 1/cm, and the inelastic sweeps it took.

    A diagonal at or near zero means a singular subproblem at that node. It can happen with negative
    densities, and it is reported here, not treated.
    """

    flux: np.ndarray
    diagonal: np.ndarray
    sweeps: int = 1
    block_marches: int = 0  # marches of the upscatter block, inside its Krylov solves
    # Per banded term (ops.urr.terms), its self-shielded (total, elastic, capture, fission) at every
    # node, the smooth cross sections where no tables apply; capture is absorption less fission.
    shield: np.ndarray | None = None
    # False if max_sweeps ran out first. emission is the lagged hat emission the next sweep would
    # take, (terms, coarse nodes) of the operators solved, to start a later solve from.
    converged: bool = True
    emission: np.ndarray | None = None


def build_operators(
    energy, removal, scatter, alpha, angle=None, inelastic=None, urr=None, order_rows=None
):
    """Check a grid against its terms, and collect the panel weights.

    angle is a list of AngleTables per term, with None for an isotropic one. None means all are.
    inelastic is InelasticKernels from zerog.energy.inelastic.build_inelastic, or None for none.
    Its emission adds to scattering, so removal must hold the term's whole total cross section.
    urr is UrrData from zerog.energy.urr.build_urr, or None for no probability tables. Where a
    banded term's tables apply, its inelastic emission is only the flagged reaction's, as OpenMC's
    total there holds no other.

    order_rows, for Legendre orders above zero, holds per term a list over l = 1 .. L of
    (SparseRows, lost) on the inelastic kernels' coarse grid, as zerog.energy.moments.
    elastic_order_rows gives them. Each (term, order) becomes a pseudo-term of extended operators,
    with no removal and no window kernel, emitting through its rows what the term's elastic
    cross section restricts to each hat; order 0 keeps the window recursion.
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
    kernels = no_inelastic(energy, alpha.shape[0]) if inelastic is None else inelastic
    if kernels.emission.shape != expected:
        raise ValueError(f"shape: inelastic emission {kernels.emission.shape} must be {expected}")
    # Kernels built on another grid would restrict and prolong across the wrong panels, silently.
    index = kernels.coarse_index
    if (
        index.shape != kernels.coarse.shape
        or index.max() >= energy.shape[0]
        or not np.array_equal(energy[index], kernels.coarse)
        or index[0] != 0
        or index[-1] != energy.shape[0] - 1
    ):
        raise ValueError(
            "inelastic kernels were built on a grid that this energy grid doesn't hold"
        )
    bands = no_urr(energy.shape[0]) if urr is None else urr
    if bands.smooth.shape[2] != energy.shape[0]:
        raise ValueError("probability tables were built on another grid")
    if bands.terms.shape[0] and (bands.terms.max() >= alpha.shape[0] or bands.terms.min() < 0):
        raise ValueError(f"banded terms {bands.terms} outside {alpha.shape[0]} terms")
    if bands.terms.shape[0]:
        emission = kernels.emission.copy()
        for j, t in enumerate(bands.terms):
            low = bands.energy[bands.start[j]]
            high = bands.energy[bands.start[j + 1] - 1]
            inside = (energy > low) & (energy < high)
            emission[t, inside] = bands.smooth[j, 3, inside]
        kernels = InelasticKernels(**{**kernels.__dict__, "emission": emission})
    extended = None
    n_orders = 0
    if order_rows is not None:
        n_orders = len(order_rows[0])
        if len(order_rows) != alpha.shape[0] or any(len(r) != n_orders for r in order_rows):
            raise ValueError("order_rows must hold the same orders for every term")
        extended = _extended_operators(
            energy, removal, scatter, alpha, angle, kernels, bands, order_rows
        )
    panel_m, panel_share, node_m, node_theta = panel_hats(energy, kernels.coarse)
    rates = rate_weights(energy)
    emissions = emission_weights(energy)
    return EnergyOperators(
        energy=energy,
        removal=removal,
        scatter=scatter,
        alpha=alpha,
        rate_weights=rates,
        emission_weights=emissions,
        moment_weights=centred_weights(energy),
        angle=compile_angle_tables([None] * alpha.shape[0] if angle is None else angle),
        inelastic=kernels,
        panel_m=panel_m,
        panel_share=panel_share,
        node_m=node_m,
        node_theta=node_theta,
        urr=bands,
        orders=n_orders,
        extended=extended,
    )


def _extended_operators(energy, removal, scatter, alpha, angle, kernels, bands, order_rows):
    """Operators over the terms and then each (order, term) pseudo-term, order by order."""
    n_terms, n = removal.shape
    n_orders = len(order_rows[0])
    n_coarse = kernels.coarse.shape[0]
    zeros = np.zeros((n_terms * n_orders, n))
    emission = np.vstack([kernels.emission, *([scatter] * n_orders)])
    # The terms' rows keep their offsets into the flat values; order rows follow them.
    starts = list(kernels.starts)
    firsts = list(kernels.first)
    lasts = list(kernels.last)
    losts = list(kernels.lost)
    values = [kernels.values]
    offset = kernels.values.shape[0]
    for order in range(n_orders):
        for t in range(n_terms):
            rows, lost = order_rows[t][order]
            if rows.starts.shape[0] != n_coarse:
                raise ValueError("order rows were built on another coarse grid")
            starts.append(rows.starts + offset)
            values.append(rows.values)
            offset += rows.values.shape[0]
            firsts.append(rows.first)
            lasts.append(rows.last)
            losts.append(lost)
    merged = InelasticKernels(
        coarse=kernels.coarse,
        coarse_index=kernels.coarse_index,
        hat=kernels.hat,
        emission=emission,
        starts=np.array(starts, dtype=np.int64),
        first=np.array(firsts, dtype=np.int64),
        last=np.array(lasts, dtype=np.int64),
        values=np.concatenate(values) if values else np.empty(0),
        lost=np.array(losts),
    )
    angles = [None] * n_terms if angle is None else list(angle)
    return build_operators(
        energy,
        np.vstack([removal, zeros]),
        np.vstack([scatter, zeros]),
        np.concatenate([alpha, np.tile(alpha, n_orders)]),
        angles + [None] * (n_terms * n_orders),
        merged,
        bands,
    )


@njit(cache=True, error_model="numpy")
def _edge_integral(energy, sigma, flux, running, j, edge):
    """The running integral of sigma F / E du at an energy edge inside panel j.

    On the piece from energy[j] down to the edge, v in lethargy below its top and s its width,
    sigma = sigma_j + (sigma_edge - sigma_j) (1 - e^-v) / (1 - e^-s), since it is linear in E, and
    F = F_j + (F_edge - F_j) v / s. So the piece's integral is exact in closed form:
    [sigma_j (F_j I + dF P) + (sigma_edge - sigma_j) (F_j J0 + dF J1) / (1 - e^-s)] / energy[j],
    with I = expm1(s), P = int v e^v / s, J0 = int (e^v - 1) and J1 = int v (e^v - 1) / s over
    0 < v < s. Below s = 0.1 these take their series, which cancel nothing.
    """
    upper = energy[j]
    lower = energy[j + 1]
    g = (edge - lower) / (upper - lower)
    sigma_edge = g * sigma[j] + (1.0 - g) * sigma[j + 1]
    h = np.log(upper / lower)
    s = np.log(upper / edge)
    flux_edge = flux[j] + (flux[j + 1] - flux[j]) * (s / h)
    df = flux_edge - flux[j]
    i0 = np.expm1(s)
    if s < 0.1:
        term = 1.0  # s^(m - 1) / m!
        j0 = 0.0
        j1 = 0.0
        for m in range(2, 12):
            term *= s / m
            j0 += term * s
            if m >= 3:
                j1 += (m - 1) * term
        p = j1 + 0.5 * s
    else:
        p = (s * (i0 + 1.0) - i0) / s
        j0 = i0 - s
        j1 = p - 0.5 * s
    shape = (sigma_edge - sigma[j]) * (flux[j] * j0 + df * j1) / -np.expm1(-s) if s > 0.0 else 0.0
    return running[j] + (sigma[j] * (flux[j] * i0 + df * p) + shape) / upper


@njit(cache=True, error_model="numpy")
def edge_integral_quadrature(energy, sigma, flux, running, j, edge):
    """_edge_integral by Gauss quadrature, its oracle."""
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
    rates,
    emission,
    hat,
    coarse_index,
    panel_m,
    panel_share,
    node_m,
    node_theta,
    starts,
    first,
    last,
    values,
    lagged,
    coarse_source,
    mode,
    moment_weights,
    start,
    u_terms,
    u_start,
    u_bands,
    u_energy,
    u_cdf,
    u_elastic,
    u_fission,
    u_capture,
    u_multiply,
    u_rule,
    u_smooth,
    u_points,
    u_weights,
):
    """Forward substitution over the nodes from start, with F zero above it.

    Inelastic emission is restricted to coarse hats as each panel completes. A hat's row goes to
    coarse_source once the march leaves the hat, replacing its lagged value; what lands in hats not
    yet passed stays lagged until the caller's next sweep.

    Each term's running integral restarts once its block is a window wide. A window then spans at
    most the current block and the previous one, so the subtraction cancels only local magnitudes
    and F keeps its relative accuracy when it falls faster than E.

    mode picks the anisotropic window: 0 is the direct sum, anisotropic_correction; above 0,
    anisotropic terms keep a tree of centred panel moments for separable_correction, which shares
    each knot's cut panel between its two segments if mode is 2 (ANISOTROPY_MODES).

    Every running sum is linear in the F above it, so with F zero above start, marching from start
    with empty sums is the whole march. The block solve uses that for sources that only reach
    below start.

    Where a banded term's probability tables apply (zerog.energy.urr), the node's F splits into one
    F_c per band combination c of weight p_c, each with its own removal, and all fed by one
    scattering source: D_c F_c = p_c (R + sum_c' g_c' F_c'), where g_c is the own panel's
    scattering, so F_c = p_c R / ((1 - A) D_c) with A = sum_c p_c g_c / D_c. The panel above
    takes each band's own products, so running sums are exact. Window edges and moment trees take
    each banded term's self-shielded cross section, sum_c sigma_c F_c / F, which shield returns
    per banded term: (total, elastic, capture, fission) at every node. capture is absorption less
    fission.
    """
    n_terms = removal.shape[0]
    n = energy.shape[0]
    separable = mode > 0
    shared_cuts = mode == 2
    flux = np.zeros(n)
    diagonal = np.zeros(n)
    running = np.zeros((n_terms, n))
    pointer = np.zeros(n_terms, dtype=np.int64)
    block_start = np.zeros(n_terms, dtype=np.int64)
    previous_total = np.zeros(n_terms)
    width = np.array([-np.log(a) if a > 0.0 else WIDE_WINDOW for a in alpha])
    # Terms with no elastic scattering, such as the pseudo-terms of orders above zero, add nothing
    # to the window recursion, so they skip it (and their running sums stay zero).
    elastic = np.zeros(n_terms, dtype=np.bool_)
    for t in range(n_terms):
        for m in range(n):
            if scatter[t, m] != 0.0:
                elastic[t] = True
                break
    for j in range(u_terms.shape[0]):
        elastic[u_terms[j]] = True  # band elastic can be nonzero where the smooth one is not
    restricted = np.zeros(lagged.shape)
    # Anisotropic terms' slots in moments, and the node below which their direct sums stop.
    slot = np.full(n_terms, -1)
    bottom = np.zeros(n_terms)
    n_slots = 0
    # Panels the moment trees must hold: windows never reach below a term's bottom node.
    tree_panels = 0
    for t in range(n_terms):
        reach = n - 1
        for m in range(n):
            if energy[m] <= cutoff[t]:
                bottom[t] = energy[m]
                reach = m
                break
        if separable and cutoff[t] < np.inf:
            slot[t] = n_slots
            n_slots += 1
            tree_panels = max(tree_panels, reach)
    size = moment_tree_size(tree_panels + 1)
    widths = np.log(energy[:-1] / energy[1:])
    trees = np.zeros((n_slots, 2 * size if n_slots > 0 else 0, 4))
    # Bands: each banded term's index, its band cross sections at this node and the one above, and
    # every combination's F there.
    n_banded = u_terms.shape[0]
    n_comb = u_points.shape[0]
    band_of = np.full(n_terms, -1)
    for j in range(n_banded):
        band_of[u_terms[j]] = j
    scatter = scatter.copy()  # banded nodes take their self-shielded elastic
    shield = np.zeros((n_banded, 4, n if n_banded > 0 else 0))
    for j in range(n_banded):
        t = u_terms[j]
        shield[j, 0] = removal[t]
        shield[j, 1] = scatter[t]
        shield[j, 2] = u_smooth[j, 1]
        shield[j, 3] = u_smooth[j, 2]
    in_now = np.zeros(n_banded, dtype=np.bool_)
    in_before = np.zeros(n_banded, dtype=np.bool_)
    el_now = np.zeros((n_banded, n_comb))
    el_before = np.zeros((n_banded, n_comb))
    tot_now = np.zeros((n_banded, n_comb))
    cap_now = np.zeros((n_banded, n_comb))
    fis_now = np.zeros((n_banded, n_comb))
    f_now = np.zeros(n_comb)
    f_before = np.zeros(n_comb)
    d_band = np.zeros(n_comb)
    g_band = np.zeros(n_comb)
    smooth_k = np.zeros(4)
    # The node above start only sets the band state, which the panel down to start needs.
    for k in range(max(start - 1, 0), n):
        lhs = streaming
        rhs = source[k]
        # Band state at node k: which tables apply, and every combination's cross sections.
        any_now = False
        for j in range(n_banded):
            in_before[j] = in_now[j]
            for c in range(n_comb):
                el_before[j, c] = el_now[j, c]
            in_now[j] = in_range(u_energy, u_start, j, energy[k])
            t = u_terms[j]
            if in_now[j]:
                any_now = True
                for q in range(4):
                    smooth_k[q] = u_smooth[j, q, k]
                for c in range(n_comb):
                    xs = band_xs(
                        u_energy,
                        u_start,
                        u_bands,
                        u_cdf,
                        u_elastic,
                        u_fission,
                        u_capture,
                        u_multiply,
                        u_rule,
                        j,
                        energy[k],
                        u_points[c, j],
                        smooth_k,
                    )
                    el_now[j, c] = xs[0]
                    cap_now[j, c] = xs[1]
                    fis_now[j, c] = xs[2]
                # Scale each reaction so the points' mean is the exact mean over r.
                exact = band_mean(
                    u_energy,
                    u_start,
                    u_bands,
                    u_cdf,
                    u_elastic,
                    u_fission,
                    u_capture,
                    u_multiply,
                    u_rule,
                    j,
                    energy[k],
                    smooth_k,
                )
                means = np.zeros(3)
                for c in range(n_comb):
                    means[0] += u_weights[c] * el_now[j, c]
                    means[1] += u_weights[c] * cap_now[j, c]
                    means[2] += u_weights[c] * fis_now[j, c]
                for c in range(n_comb):
                    if means[0] > 0.0 and exact[0] > 0.0:
                        el_now[j, c] *= exact[0] / means[0]
                    if means[1] > 0.0 and exact[1] > 0.0:
                        cap_now[j, c] *= exact[1] / means[1]
                    if means[2] > 0.0 and exact[2] > 0.0:
                        fis_now[j, c] *= exact[2] / means[2]
                    tot_now[j, c] = el_now[j, c] + cap_now[j, c] + fis_now[j, c] + smooth_k[3]
            else:
                for c in range(n_comb):
                    el_now[j, c] = scatter[t, k]
                    tot_now[j, c] = removal[t, k]
                    cap_now[j, c] = u_smooth[j, 1, k]
                    fis_now[j, c] = u_smooth[j, 2, k]
        if k < start:
            continue
        banded = any_now
        for j in range(n_banded):
            if in_before[j]:
                banded = True
        if banded:
            for c in range(n_comb):
                d_band[c] = 0.0
                g_band[c] = 0.0
        common_own = 0.0
        for t in range(n_terms):
            jt = band_of[t]
            # A banded term takes the band path where its tables apply at this node or the last.
            split = banded and jt >= 0 and (in_now[jt] or in_before[jt])
            if split:
                for c in range(n_comb):
                    d_band[c] += d_removal[t] * tot_now[jt, c]
            else:
                lhs += d_removal[t] * removal[t, k]
            if k == 0 or not elastic[t]:
                continue
            sigma = scatter[t]
            w = weights[k - 1]
            if k > 1 and energy[k - 1] <= alpha[t] * energy[block_start[t]]:
                previous_total[t] = running[t, k - 1]
                running[t, k - 1] = 0.0
                block_start[t] = k - 1
            i = 1.0 / energy[k - 1]
            if split:
                panel = 0.0
                for c in range(n_comb):
                    panel += f_before[c] * (el_before[jt, c] * w[0] + el_now[jt, c] * w[2])
                known = running[t, k - 1] + i * panel
            else:
                panel = flux[k - 1] * (sigma[k - 1] * w[0] + sigma[k] * w[2])
                known = running[t, k - 1] + i * panel
            own = i * (sigma[k - 1] * w[1] + sigma[k] * w[3]) if not split else 0.0
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
            if slot[t] >= 0 and energy[pointer[t]] > cutoff[t]:
                extra, extra_own = separable_correction(
                    energy,
                    k,
                    sigma,
                    flux,
                    alpha[t],
                    pointer[t],
                    bottom[t],
                    widths,
                    trees[slot[t]],
                    size,
                    t,
                    tables,
                    table_energy,
                    points,
                    mu,
                    pdf,
                    shared_cuts,
                )
                window += extra
                own += extra_own
            elif energy[pointer[t]] > cutoff[t]:
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
            if split:
                # own holds the anisotropic part only, taken as common to every band.
                for c in range(n_comb):
                    band_own = i * (el_before[jt, c] * w[1] + el_now[jt, c] * w[3]) + own
                    g_band[c] += d_scatter[t] * factor * band_own
            elif banded:
                common_own += d_scatter[t] * factor * own
            else:
                lhs -= d_scatter[t] * factor * own
            rhs += d_scatter[t] * factor * window
            running[t, k] = known  # completed below, once F[k] is known
        m = node_m[k]
        theta = node_theta[k]
        rhs += (1.0 - theta) * coarse_source[m] / hat[m] + theta * coarse_source[m + 1] / hat[m + 1]
        if banded:
            coupling = 0.0
            for c in range(n_comb):
                d_band[c] += lhs
                g_band[c] += common_own
                coupling += u_weights[c] * g_band[c] / d_band[c]
            shared = rhs / (1.0 - coupling)
            total = 0.0
            for c in range(n_comb):
                f_now[c] = u_weights[c] * shared / d_band[c]
                total += f_now[c]
            flux[k] = total
            # The effective diagonal, whatever the source; a band at or below zero is reported.
            inverse = 0.0
            lowest = d_band[0]
            for c in range(n_comb):
                inverse += u_weights[c] / d_band[c]
                lowest = min(lowest, d_band[c])
            diagonal[k] = (1.0 - coupling) / inverse
            if lowest <= 0.0:
                diagonal[k] = min(diagonal[k], lowest)
            for j in range(n_banded):
                if not in_now[j]:
                    continue
                t = u_terms[j]
                sums = np.zeros(4)
                for c in range(n_comb):
                    sums[0] += tot_now[j, c] * f_now[c]
                    sums[1] += el_now[j, c] * f_now[c]
                    sums[2] += cap_now[j, c] * f_now[c]
                    sums[3] += fis_now[j, c] * f_now[c]
                if total != 0.0:
                    for q in range(4):
                        shield[j, q, k] = sums[q] / total
                    scatter[t, k] = shield[j, 1, k]
        else:
            diagonal[k] = lhs
            flux[k] = rhs / lhs
            for c in range(n_comb):
                f_now[c] = u_weights[c] * flux[k]
        if k > 0 and values.shape[0] > 0:
            p = panel_m[k - 1]
            w = rates[k - 1]
            for t in range(n_terms):
                sigma = emission[t]
                rate = sigma[k - 1] * (w[0] * flux[k - 1] + w[1] * flux[k]) + sigma[k] * (
                    w[2] * flux[k - 1] + w[3] * flux[k]
                )
                restricted[t, p] += rate * panel_share[k - 1]
                restricted[t, p + 1] += rate * (1.0 - panel_share[k - 1])
            if k == coarse_index[p + 1]:
                # Hat p is complete: send its emission, less what the lagged pass already sent.
                for t in range(n_terms):
                    change = d_scatter[t] * (restricted[t, p] - lagged[t, p])
                    row_start = starts[t, p]
                    for c in range(first[t, p], last[t, p]):
                        coarse_source[c] += change * values[row_start + c - first[t, p]]
        if k > 0:
            for t in range(n_terms):
                if not elastic[t]:
                    continue
                sigma = scatter[t]
                w = weights[k - 1]
                jt = band_of[t]
                if banded and jt >= 0 and (in_now[jt] or in_before[jt]):
                    panel = 0.0
                    for c in range(n_comb):
                        panel += f_now[c] * (el_before[jt, c] * w[1] + el_now[jt, c] * w[3])
                    running[t, k] += panel / energy[k - 1]
                else:
                    running[t, k] += (
                        flux[k] * (sigma[k - 1] * w[1] + sigma[k] * w[3]) / energy[k - 1]
                    )
                if slot[t] >= 0 and k - 1 < tree_panels:
                    moment_tree_add(
                        trees[slot[t]], size, energy, sigma, flux, moment_weights, k - 1
                    )
        for c in range(n_comb):
            f_before[c] = f_now[c]
    return flux, diagonal, restricted, shield


@njit
def _lagged_source(d_scatter, lagged, starts, first, last, values, n_coarse):
    out = np.zeros(n_coarse)
    for t in range(lagged.shape[0]):
        for j in range(n_coarse):
            change = d_scatter[t] * lagged[t, j]
            for c in range(first[t, j], last[t, j]):
                out[c] += change * values[starts[t, j] + c - first[t, j]]
    return out


def upscatter_block(kernels):
    """The first coarse hat of the block solved by Krylov, and the fine node its rows reach up to.

    The block starts at the highest hat whose row puts more than UPSCATTER_SHARE of itself above
    its upper neighbour, which only thermal scattering does, and runs to the bottom. None if no
    row reaches up.
    """
    n = kernels.coarse.shape[0]
    filled = kernels.last > kernels.first
    upward = filled & (kernels.first < np.arange(n)[None, :] - 1)
    # Inelastic rows can leave round-off above their neighbour (1e-11 of a row, in Pu239's), which
    # lagged sweeps handle; only a real share of a row going up needs the block.
    for t, j in zip(*np.nonzero(upward), strict=True):
        a = kernels.starts[t, j]
        row = kernels.values[a : a + kernels.last[t, j] - kernels.first[t, j]]
        above = np.abs(row[: j - 1 - kernels.first[t, j]]).sum()
        if above <= UPSCATTER_SHARE * np.abs(row).sum():
            upward[t, j] = False
    if not upward.any():
        return None
    first_hat = int(np.nonzero(upward.any(axis=0))[0].min())
    reach = int(kernels.first[:, first_hat:][filled[:, first_hat:]].min())
    # Nodes in interval m take a share of hat m + 1, so a deposit in hat reach drives F from the
    # top of interval reach - 1.
    return first_hat, int(kernels.coarse_index[max(reach - 1, 0)])


def march(ops, removal, scatter, streaming, source, lagged, start=0, anisotropy="separable"):
    """One march from node start with lagged hat emission (terms, coarse nodes): (flux, diagonal,
    restricted emission, shield). removal is (terms) and scatter (terms, 1), as solve_energy takes
    them; F is zero above start, and source must be too."""
    kernels = ops.inelastic
    n_coarse = kernels.coarse.shape[0]
    removal = np.asarray(removal, dtype=float)
    scatter = np.asarray(scatter, dtype=float)
    source = np.asarray(source, dtype=float)
    lagged = np.asarray(lagged, dtype=float)
    coarse_source = _lagged_source(
        scatter[:, 0],
        lagged,
        kernels.starts,
        kernels.first,
        kernels.last,
        kernels.values,
        n_coarse,
    )
    return _march(
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
        ops.rate_weights,
        kernels.emission,
        kernels.hat,
        kernels.coarse_index,
        ops.panel_m,
        ops.panel_share,
        ops.node_m,
        ops.node_theta,
        kernels.starts,
        kernels.first,
        kernels.last,
        kernels.values,
        lagged,
        coarse_source,
        ANISOTROPY_MODES[anisotropy],
        ops.moment_weights,
        start,
        ops.urr.terms,
        ops.urr.start,
        ops.urr.bands,
        ops.urr.energy,
        ops.urr.cdf,
        ops.urr.elastic,
        ops.urr.fission,
        ops.urr.capture,
        ops.urr.multiply,
        ops.urr.rule,
        ops.urr.smooth,
        ops.urr.points,
        ops.urr.weights,
    )


def solve_energy(
    ops,
    removal,
    scatter,
    streaming,
    source,
    anisotropy="separable",
    initial=None,
    max_sweeps=None,
):
    """Solve for F. removal is (terms), scatter is (terms, orders), source is (energies).

    anisotropy is "separable" (elastic.separable_correction), "tree" (the same with quadrature of f
    on both sides of every cut, decision 008) or "direct", the oracle's direct sum.
    initial is a first guess of the lagged inelastic emission, (terms, coarse nodes) as
    zerog.energy.inelastic.restrict gives it, such as that of a previous flux; zero by default.

    Each sweep marches the whole grid with the emission of hats not yet passed lagged. Hats above
    the upscatter block then take the sweep's emission. Hats in the block take the exact solution
    of their own lagged problem given the rest, by GMRES on the lagged emission, with marches from
    the block's top as the operator: a march is Gauss-Seidel, so this is GMRES preconditioned by
    the downward sweep. Sweeps end when one reproduces the emission it was given.
    """
    if max_sweeps is not None and int(max_sweeps) < 1:
        raise ValueError(f"max_sweeps must be at least 1, not {max_sweeps}")
    if anisotropy not in ANISOTROPY_MODES:
        raise ValueError(f"anisotropy must be one of {list(ANISOTROPY_MODES)}, not {anisotropy!r}")
    removal = np.asarray(removal, dtype=float)
    scatter = np.asarray(scatter, dtype=float)
    source = np.asarray(source, dtype=float)
    n_terms, n = ops.removal.shape
    if scatter.ndim == 2 and scatter.shape[0] == n_terms and scatter.shape[1] > 1:
        orders = scatter.shape[1] - 1
        if ops.extended is None or orders > ops.orders:
            raise ValueError(f"scatter has {orders} orders above zero, ops holds {ops.orders}")
        if removal.shape != (n_terms,) or source.shape != (n,):
            raise ValueError(f"shape: removal {removal.shape} and source {source.shape}")
        # Pseudo-terms, order by order, after the terms: no removal, each order's density.
        padded = np.zeros((n_terms, ops.orders + 1))
        padded[:, : orders + 1] = scatter
        d_removal = np.concatenate([removal, np.zeros(n_terms * ops.orders)])
        d_scatter = padded.T.ravel()[:, None]
        start = None
        if initial is not None:
            initial = np.asarray(initial, dtype=float)
            if initial.shape[0] == d_removal.shape[0]:
                start = initial
            else:
                start = np.zeros((d_removal.shape[0], ops.extended.inelastic.coarse.shape[0]))
                start[:n_terms] = initial
        return solve_energy(
            ops.extended, d_removal, d_scatter, streaming, source, anisotropy, start, max_sweeps
        )
    if removal.shape != (n_terms,) or scatter.shape != (n_terms, 1) or source.shape != (n,):
        raise ValueError(
            f"shape: removal {removal.shape}, scatter {scatter.shape} and source {source.shape} "
            f"do not fit {n_terms} terms and {n} energies"
        )
    kernels = ops.inelastic
    n_coarse = kernels.coarse.shape[0]
    lagged = np.zeros((n_terms, n_coarse))
    if initial is not None:
        lagged = np.array(initial, dtype=float)
        if lagged.shape != (n_terms, n_coarse):
            raise ValueError(f"shape: initial {lagged.shape} must be {(n_terms, n_coarse)}")

    def sweep(source, lagged, start):
        return march(ops, removal, scatter, streaming, source, lagged, start, anisotropy)

    block = upscatter_block(kernels) if kernels.values.shape[0] > 0 else None
    applications = 0
    limit = MAX_SWEEPS if max_sweeps is None else min(int(max_sweeps), MAX_SWEEPS)
    for count in range(1, limit + 1):
        flux, diagonal, restricted, shield = sweep(source, lagged, 0)
        scale = np.abs(restricted).max()
        if (
            kernels.values.shape[0] == 0
            or np.abs(restricted - lagged).max() <= SWEEP_TOLERANCE * scale
        ):
            return EnergySolution(flux, diagonal, count, applications, shield, emission=restricted)
        if block is None:
            lagged = restricted
            if count == limit and max_sweeps is not None:
                return EnergySolution(flux, diagonal, count, applications, shield, False, lagged)
            continue
        first_hat, start = block
        zero = np.zeros(n)
        shape = (n_terms, n_coarse - first_hat)

        def apply(v, first_hat=first_hat, start=start, zero=zero, shape=shape):
            x = np.zeros((n_terms, n_coarse))
            x[:, first_hat:] = v.reshape(shape)
            out = sweep(zero, x, start)[2]
            return v - out[:, first_hat:].ravel()

        # The sweep gave r = G x + b; the block's own problem is (I - G_bb) dx = r_b - x_b.
        rhs = (restricted[:, first_hat:] - lagged[:, first_hat:]).ravel()
        # The outer test is SWEEP_TOLERANCE * scale on each entry. A 2-norm bound bounds every
        # entry, so a tenth of it leaves the outer test to the other hats; a bound scaled by
        # sqrt(entries) would not, and the outer sweeps would never pass.
        # A relative stop as well keeps GMRES off its round-off floor; what it leaves is a smaller
        # right-hand side for the next sweep's block solve.
        tolerance = 0.1 * SWEEP_TOLERANCE * scale
        change, used = solve(
            apply, rhs, atol=tolerance, rtol=BLOCK_RTOL, max_iterations=BLOCK_RESTARTS
        )
        applications += used
        updated = restricted.copy()
        updated[:, first_hat:] = lagged[:, first_hat:] + change.reshape(shape)
        lagged = updated
        if count == limit and max_sweeps is not None:
            return EnergySolution(flux, diagonal, count, applications, shield, False, lagged)
    raise RuntimeError(f"inelastic sweeps did not converge in {MAX_SWEEPS}")
