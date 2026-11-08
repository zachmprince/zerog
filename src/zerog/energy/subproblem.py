"""The PGD energy subproblem, in flux per unit lethargy F = E * epsilon(E) at the grid's nodes:

    [a + sum_t d_R[t] sigma_R[t](E)] F(u) = sum_t,l d_S[t, l] Y[t, l](E) + q(u)

Y[t, l] is the emission density per unit lethargy of term t and Legendre order l, so q is a source
per unit lethargy as well. Densities and the streaming coefficient a can be negative. The equation
is collocated at the nodes, and the implicit part of each node's own panel moves to the left.

Only order 0 scattering exists so far.
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
from zerog.linalg import solve

MAX_SWEEPS = 50
SWEEP_TOLERANCE = 1e-12  # on the hats' emission, relative to the largest
BLOCK_RESTARTS = 20  # GMRES restart cycles per block solve
BLOCK_RTOL = 1e-10  # GMRES stop relative to the block's right-hand side


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


def build_operators(energy, removal, scatter, alpha, angle=None, inelastic=None):
    """Check a grid against its terms, and collect the panel weights.

    angle is a list of AngleTables per term, with None for an isotropic one. None means all are.
    inelastic is InelasticKernels from zerog.energy.inelastic.build_inelastic, or None for none.
    Its emission adds to scattering, so removal must hold the term's whole total cross section.
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
    separable,
    moment_weights,
    start,
):
    """Forward substitution over the nodes from start, with F zero above it.

    Inelastic emission is restricted to coarse hats as each panel completes. A hat's row goes to
    coarse_source once the march leaves the hat, replacing its lagged value; what lands in hats not
    yet passed stays lagged until the caller's next sweep.

    Each term's running integral restarts once its block is a window wide. A window then spans at
    most the current block and the previous one, so the subtraction cancels only local magnitudes
    and F keeps its relative accuracy when it falls faster than E.

    If separable, anisotropic terms keep a tree of centred panel moments, for
    separable_correction; otherwise they take the direct sum, anisotropic_correction.

    Every running sum is linear in the F above it, so with F zero above start, marching from start
    with empty sums is the whole march. The block solve uses that for sources that only reach
    below start.
    """
    n_terms = removal.shape[0]
    n = energy.shape[0]
    flux = np.zeros(n)
    diagonal = np.zeros(n)
    running = np.zeros((n_terms, n))
    pointer = np.zeros(n_terms, dtype=np.int64)
    block_start = np.zeros(n_terms, dtype=np.int64)
    previous_total = np.zeros(n_terms)
    width = np.array([-np.log(a) if a > 0.0 else WIDE_WINDOW for a in alpha])
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
    for k in range(start, n):
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
            lhs -= d_scatter[t] * factor * own
            rhs += d_scatter[t] * factor * window
            running[t, k] = known  # completed below, once F[k] is known
        m = node_m[k]
        theta = node_theta[k]
        rhs += (1.0 - theta) * coarse_source[m] / hat[m] + theta * coarse_source[m + 1] / hat[m + 1]
        diagonal[k] = lhs
        flux[k] = rhs / lhs
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
                sigma = scatter[t]
                w = weights[k - 1]
                running[t, k] += flux[k] * (sigma[k - 1] * w[1] + sigma[k] * w[3]) / energy[k - 1]
                if slot[t] >= 0 and k - 1 < tree_panels:
                    moment_tree_add(
                        trees[slot[t]], size, energy, sigma, flux, moment_weights, k - 1
                    )
    return flux, diagonal, restricted


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

    The block starts at the highest hat whose row reaches above its upper neighbour, which only
    thermal scattering does, and runs to the bottom. None if no row reaches up.
    """
    n = kernels.coarse.shape[0]
    filled = kernels.last > kernels.first
    upward = filled & (kernels.first < np.arange(n)[None, :] - 1)
    if not upward.any():
        return None
    first_hat = int(np.nonzero(upward.any(axis=0))[0].min())
    reach = int(kernels.first[:, first_hat:][filled[:, first_hat:]].min())
    # Nodes in interval m take a share of hat m + 1, so a deposit in hat reach drives F from the
    # top of interval reach - 1.
    return first_hat, int(kernels.coarse_index[max(reach - 1, 0)])


def solve_energy(ops, removal, scatter, streaming, source, anisotropy="separable", initial=None):
    """Solve for F. removal is (terms), scatter is (terms, orders), source is (energies).

    anisotropy is "separable" (elastic.separable_correction) or "direct", the oracle's direct sum.
    initial is a first guess of the lagged inelastic emission, (terms, coarse nodes) as
    zerog.energy.inelastic.restrict gives it, such as that of a previous flux; zero by default.

    Each sweep marches the whole grid with the emission of hats not yet passed lagged. Hats above
    the upscatter block then take the sweep's emission. Hats in the block take the exact solution
    of their own lagged problem given the rest, by GMRES on the lagged emission, with marches from
    the block's top as the operator: a march is Gauss-Seidel, so this is GMRES preconditioned by
    the downward sweep. Sweeps end when one reproduces the emission it was given.
    """
    if anisotropy not in ("separable", "direct"):
        raise ValueError(f"anisotropy must be 'separable' or 'direct', not {anisotropy!r}")
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
    kernels = ops.inelastic
    n_coarse = kernels.coarse.shape[0]
    lagged = np.zeros((n_terms, n_coarse))
    if initial is not None:
        lagged = np.array(initial, dtype=float)
        if lagged.shape != (n_terms, n_coarse):
            raise ValueError(f"shape: initial {lagged.shape} must be {(n_terms, n_coarse)}")

    def sweep(source, lagged, start):
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
            anisotropy == "separable",
            ops.moment_weights,
            start,
        )

    block = upscatter_block(kernels) if kernels.values.shape[0] > 0 else None
    applications = 0
    for count in range(1, MAX_SWEEPS + 1):
        flux, diagonal, restricted = sweep(source, lagged, 0)
        scale = np.abs(restricted).max()
        if (
            kernels.values.shape[0] == 0
            or np.abs(restricted - lagged).max() <= SWEEP_TOLERANCE * scale
        ):
            return EnergySolution(flux, diagonal, count, applications)
        if block is None:
            lagged = restricted
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
    raise RuntimeError(f"inelastic sweeps did not converge in {MAX_SWEEPS}")
