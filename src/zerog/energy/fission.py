"""Fission neutron emission: production rates collapsed onto incoming-energy hats, then spectra.

Each fission reaction emits prompt neutrons, weighted nu_total - sum(nu_delayed) as
zerog.data.neutron.FissionData says, and one delayed component per group. OpenMC v0.16.0's
sample_fission_neutron samples each with the product's own distribution, the same samplers that
zerog.energy.inelastic transcribes (ContinuousTabular, or MaxwellEnergy for a Maxwell spectrum),
and resamples any energy at or above data::energy_max, the lowest top energy of the loaded
nuclides. So a spectrum here is truncated at that top and renormalized, rather than losing a tail.

A component's expected spectrum depends on the incoming energy only through its table mix:

- Lin-lin between tables, it is linear in E where the bounding tables share their outgoing range,
  so hats in E on the table energies represent it exactly. Where unit-base scaling stretches the
  range, intervals are split until the range changes by at most CHI_STEP per piece.
- Histogram between tables (delayed spectra), it is constant on each interval.
- A Maxwell spectrum varies with theta(E), split the same way.

The production rate of each component is restricted onto those hats (or intervals) by exact panel
sums, and each node's spectrum is stored as hat values on the coarse transfer grid of
zerog.energy.inelastic, so fission adds a few dozen dense rows per term.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

from zerog.data.interp import evaluate_tabulated
from zerog.data.neutron import read_fission, read_neutron
from zerog.data.scatter import CorrelatedLaw, MaxwellSpectrum, TabularSpectrum
from zerog.energy.grid import interpolate
from zerog.energy.inelastic import (
    ENERGY_RESOLUTION,
    RULE_HISTOGRAM,
    RULE_LIN_LIN,
    _ascending,
    deposit,
    hat_weights,
    point_spacing,
    tabular_points,
)

CHI_STEP = 1e-3  # largest relative change in a table's outgoing range, or in theta, per chi node
CAP_STEP = 0.02  # largest relative change in e - u per chi node where it truncates a Maxwell
CAP_START = 1.0e3  # eV above u of a truncating Maxwell's first emitting node
MERGE = 1e-12  # relative gap below which chi nodes of different components are one node
MAXWELL_SPAN = math.log(1e9)  # lethargy below the top that a Maxwell's pieces cover; ~1e-13 left


def _incoming_rule(spectrum):
    """RULE_HISTOGRAM or RULE_LIN_LIN between incoming tables; other forms are refused."""
    rules = np.unique(spectrum.interpolation_ascending)
    # OpenMC's ContinuousTabular treats a table as histogram only with a single region.
    single = spectrum.interpolation_ascending.shape[0] == 1
    if rules.shape[0] != 1 or rules[0] not in (RULE_HISTOGRAM, RULE_LIN_LIN):
        raise NotImplementedError(f"fission spectrum with incoming interpolation {rules.tolist()}")
    if rules[0] == RULE_HISTOGRAM and not single:
        raise NotImplementedError("histogram fission spectrum with several regions")
    return int(rules[0])


def _maxwell_points(spectrum, e, du):
    """sqrt(E') exp(-E' / theta) on 0 <= E' <= e - u, in equal-lethargy pieces with exact weights.

    Returns (lower, upper, weight). MaxwellEnergy::sample rejects E' above e - u.
    """
    theta = _theta(spectrum, e)
    y = (e - spectrum.u) / theta
    if y <= 0.0:
        return np.empty(0), np.empty(0), np.empty(0)
    count = 2 + int(MAXWELL_SPAN / du)
    edges = np.concatenate([y * np.exp(-np.linspace(0.0, MAXWELL_SPAN, count)), [0.0]])
    root = np.sqrt(edges)
    cdf = np.array([math.erf(r) for r in root]) - 2.0 / math.sqrt(math.pi) * root * np.exp(-edges)
    weights = (cdf[:-1] - cdf[1:]) / cdf[0]
    keep = weights > 0.0
    return (edges[1:] * theta)[keep], (edges[:-1] * theta)[keep], weights[keep]


def _theta(spectrum, e):
    return evaluate_tabulated(
        spectrum.energy[::-1].copy(),
        spectrum.theta[::-1].copy(),
        spectrum.breakpoints_ascending,
        spectrum.interpolation_ascending,
        np.array([e]),
    )[0]


def spectrum_segments(spectrum, e, du=ENERGY_RESOLUTION):
    """Lab outgoing-energy segments (lower, upper, weight) of a fission spectrum at incoming e.

    A CorrelatedLaw contributes its energy tables, sampled as CorrelatedAngleEnergy samples them.
    Weights sum to one.
    """
    if isinstance(spectrum, MaxwellSpectrum):
        return _maxwell_points(spectrum, e, du)
    if isinstance(spectrum, CorrelatedLaw):
        tables = _ascending(spectrum.spectrum)
        out = tabular_points(*tables, e, False, True, du)
        return out[5], out[6], out[1]
    if isinstance(spectrum, TabularSpectrum):
        histogram = _incoming_rule(spectrum) == RULE_HISTOGRAM
        out = tabular_points(*_ascending(spectrum), e, True, not histogram, du)
        return out[5], out[6], out[1]
    raise NotImplementedError(f"fission spectrum of type {type(spectrum).__name__}")


def truncate(lower, upper, weights, top):
    """Segments cut at top and renormalized, as OpenMC resamples energies at or above it.

    A segment's weight is spread uniformly in lethargy, so a cut keeps its share below top.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = lower < top
    lower, upper, weights = lower[keep], upper[keep], weights[keep]
    cut = upper > top
    if np.any(cut):
        share = np.log(top / lower[cut]) / np.log(upper[cut] / lower[cut])
        weights = weights.copy()
        upper = upper.copy()
        weights[cut] *= share
        upper[cut] = top
    total = weights.sum()
    return lower, upper, weights / total if total > 0.0 else weights


def _spread(spectrum, a, b):
    """Relative change of a tabular spectrum's outgoing range between incoming energies a > b."""
    tabular = spectrum.spectrum if isinstance(spectrum, CorrelatedLaw) else spectrum
    energy, offsets, _, e_out, _, _ = _ascending(tabular)
    middle = 0.5 * (a + b)
    i = np.searchsorted(energy, middle) - 1
    if i < 0 or i + 1 >= energy.shape[0]:
        return 0.0  # clamped to an end table, so constant
    first = [e_out[offsets[j]] for j in (i, i + 1)]
    width = [e_out[offsets[j + 1] - 1] - e_out[offsets[j]] for j in (i, i + 1)]
    return max(abs(width[1] / width[0] - 1.0), abs(first[1] - first[0]) / width[0])


def _maxwell_nodes(spectrum, nodes, cap):
    """Chi nodes of a Maxwell between base nodes: by theta, and where e - u < cap truncates it,
    geometric in e - u so that a threshold at u needs finitely many."""
    u = spectrum.u
    out = [nodes[0]]
    for a, b in itertools.pairwise(nodes):
        if b <= u:
            out.append(b)  # nothing is emitted at or below u
            continue
        n = math.ceil(abs(_theta(spectrum, a) / _theta(spectrum, b) - 1.0) / CHI_STEP)
        if a - u < cap:
            n = max(n, math.ceil(math.log(min(a - u, cap) / (b - u)) / math.log1p(CAP_STEP)))
            pieces = u + np.geomspace(a - u, b - u, max(1, n) + 1)[1:]
        else:
            pieces = np.linspace(a, b, max(1, n) + 1)[1:]
        pieces[-1] = b
        out.extend(pieces.tolist())
    return np.array(out)


def chi_nodes(spectrum, top, bottom, cap=None):
    """Incoming energies, descending from top to bottom, at which a spectrum is stored.

    cap is OpenMC's data::energy_max, top by default. Returns (energies, histogram): with
    histogram, the spectrum is constant on each interval.
    """
    cap = top if cap is None else cap
    if isinstance(spectrum, MaxwellSpectrum):
        u = spectrum.u
        extra = [x for x in (u, u + CAP_START) if bottom < x < top]
        inside = spectrum.energy[(spectrum.energy < top) & (spectrum.energy > bottom)]
        nodes = np.unique(np.concatenate([[top, bottom], inside, extra]))[::-1]
        return _maxwell_nodes(spectrum, nodes, cap), False
    tabular = spectrum.spectrum if isinstance(spectrum, CorrelatedLaw) else spectrum
    histogram = _incoming_rule(tabular) == RULE_HISTOGRAM
    inside = tabular.energy[(tabular.energy < top) & (tabular.energy > bottom)]
    nodes = np.unique(np.concatenate([[top, bottom], inside]))[::-1]
    if histogram:
        return nodes, True
    out = [nodes[0]]
    for a, b in itertools.pairwise(nodes):
        pieces = max(1, math.ceil(_spread(spectrum, a, b) / CHI_STEP))
        out.extend(np.linspace(a, b, pieces + 1)[1:].tolist())
    return np.array(out), False


def _merge(energies):
    """Descending energies with any within MERGE (relative) of a kept neighbour dropped."""
    energies = np.unique(energies)[::-1]
    keep = [energies[0]]
    for e in energies[1:]:
        if keep[-1] - e > MERGE * keep[-1]:
            keep.append(e)
    return np.array(keep)


def _snap(nodes, energy):
    """Chi nodes moved onto the fine grid's nodes, which _merge may have shifted by round-off."""
    index = np.clip(np.searchsorted(-energy, -nodes), 1, energy.shape[0] - 1)
    nearer = np.where(
        np.abs(energy[index - 1] - nodes) < np.abs(energy[index] - nodes), index - 1, index
    )
    snapped = energy[nearer]
    if np.any(np.abs(snapped - nodes) > 10.0 * MERGE * nodes):
        raise ValueError("every chi node must be a node of the fine grid: see fission_nodes")
    return snapped


def fission_components(name, temperature):
    """A nuclide's fission emission as (energy, production, spectrum) per component.

    production is yield * barns on the nuclide's descending energy grid: prompt neutrons, then each
    delayed group, for every fission reaction.
    """
    data = read_neutron(name, temperature)
    out = []
    for reaction in read_fission(data):
        delayed = reaction.nu_delayed.sum(axis=0) if reaction.nu_delayed.shape[0] else 0.0
        out.append((data.energy, reaction.xs * (reaction.nu_total - delayed), reaction.chi_prompt))
        for nu, chi in zip(reaction.nu_delayed, reaction.chi_delayed, strict=True):
            out.append((data.energy, reaction.xs * nu, chi))
    return out


def fission_nodes(nuclides, top, bottom, cap=None):
    """Every chi node of (name, temperature) pairs: required energies for the fine grid.

    Nodes of different components closer than MERGE are one node, so no panel is a sliver.
    """
    out = []
    for name, temperature in nuclides:
        for _, _, spectrum in fission_components(name, temperature):
            out.extend(chi_nodes(spectrum, top, bottom, cap)[0].tolist())
    return _merge(np.array(out))


def data_top(nuclides):
    """OpenMC's data::energy_max for these nuclides: the lowest top of their energy grids."""
    return min(read_neutron(name, temperature).energy[0] for name, temperature in nuclides)


@dataclass(frozen=True)
class FissionKernels:
    """Fission emission of every component, flattened.

    Component k belongs to term[k]. Its chi nodes are node_energy[node_start[k]:node_start[k + 1]];
    its rows, rows[row_start[k]:row_start[k + 1]], are hat values on coarse of one neutron born at
    each node, or for a histogram component in each interval between nodes, with lost the share
    landing outside the grid. production[k] is its yield * barns at the fine nodes.
    """

    coarse: np.ndarray
    hat: np.ndarray
    term: np.ndarray
    histogram: np.ndarray
    node_start: np.ndarray
    node_energy: np.ndarray
    row_start: np.ndarray
    rows: np.ndarray
    lost: np.ndarray
    production: np.ndarray


def build_fission(terms, energy, coarse, top, du=None):
    """FissionKernels for terms given as (name, temperature label), or None for no fission.

    top is OpenMC's data::energy_max (data_top over every nuclide in the problem). Every chi node
    (fission_nodes, with the same top as cap) must be a node of energy, and coarse must be the
    inelastic transfer grid.
    """
    du = point_spacing(coarse) if du is None else du
    term, histogram, node_start, node_energy, row_start, rows, lost, production = (
        [],
        [],
        [0],
        [],
        [0],
        [],
        [],
        [],
    )
    for t, entry in enumerate(terms):
        if entry is None:
            continue
        for table_energy, curve, spectrum in fission_components(*entry):
            nodes, flat = chi_nodes(spectrum, energy[0], energy[-1], top)
            nodes = _snap(nodes, energy)
            at = 0.5 * (nodes[:-1] + nodes[1:]) if flat else nodes
            for e in at:
                lower, upper, weights = truncate(*spectrum_segments(spectrum, e, du), top)
                if weights.sum() <= 0.0:
                    # Nothing is emitted here (a Maxwell at e <= u), so count it as lost.
                    rows.append(np.zeros(coarse.shape[0]))
                    lost.append(1.0)
                    continue
                row, off = deposit(lower, upper, weights, coarse)
                rows.append(row)
                lost.append(off)
            term.append(t)
            histogram.append(flat)
            node_energy.extend(nodes.tolist())
            node_start.append(len(node_energy))
            row_start.append(len(rows))
            production.append(interpolate(energy, table_energy, curve))
    n = energy.shape[0]
    return FissionKernels(
        coarse=coarse,
        hat=hat_weights(coarse),
        term=np.array(term, dtype=np.int64),
        histogram=np.array(histogram, dtype=bool),
        node_start=np.array(node_start, dtype=np.int64),
        node_energy=np.array(node_energy),
        row_start=np.array(row_start, dtype=np.int64),
        rows=np.array(rows) if rows else np.zeros((0, coarse.shape[0])),
        lost=np.array(lost),
        production=np.array(production) if production else np.zeros((0, n)),
    )


def _panel_rates(sigma, flux, rate_weights):
    """Each panel's exact integral of sigma * F over lethargy, sigma lin-lin in E."""
    return sigma[:-1] * (rate_weights[:, 0] * flux[:-1] + rate_weights[:, 1] * flux[1:]) + sigma[
        1:
    ] * (rate_weights[:, 2] * flux[:-1] + rate_weights[:, 3] * flux[1:])


def node_weights(kernels, k, energy, flux, rate_weights):
    """Component k's production restricted to its chi nodes, or to its intervals if histogram.

    Each panel's total is exact. Its split between the two hats treats hat * production as linear
    in E over the panel, which is off by about the product of their changes across it.
    """
    nodes = kernels.node_energy[kernels.node_start[k] : kernels.node_start[k + 1]]
    q = kernels.production[k]
    rates = _panel_rates(q, flux, rate_weights)
    # A panel lies in the interval below the highest node at or above its upper end.
    m = np.clip(np.searchsorted(-nodes, -energy[:-1], side="right") - 1, 0, nodes.shape[0] - 2)
    if kernels.histogram[k]:
        return np.bincount(m, rates, minlength=nodes.shape[0] - 1)
    span = nodes[m] - nodes[m + 1]
    upper = (energy[:-1] - nodes[m + 1]) / span
    lower = (energy[1:] - nodes[m + 1]) / span
    up = q[:-1] * upper * (rate_weights[:, 0] * flux[:-1] + rate_weights[:, 1] * flux[1:]) + q[
        1:
    ] * lower * (rate_weights[:, 2] * flux[:-1] + rate_weights[:, 3] * flux[1:])
    return np.bincount(m, up, minlength=nodes.shape[0]) + np.bincount(
        m + 1, rates - up, minlength=nodes.shape[0]
    )


def production_rates(kernels, densities, energy, flux, rate_weights):
    """Fission neutrons produced per term: densities * yield * sigma_f * F, integrated exactly."""
    out = np.zeros(len(densities))
    for k in range(kernels.term.shape[0]):
        t = kernels.term[k]
        out[t] += densities[t] * _panel_rates(kernels.production[k], flux, rate_weights).sum()
    return out


def fission_hats(kernels, densities, energy, flux, rate_weights):
    """Hat values of fission emission on the coarse grid, and the neutrons born outside it."""
    values = np.zeros(kernels.coarse.shape[0])
    lost = 0.0
    for k in range(kernels.term.shape[0]):
        d = densities[kernels.term[k]]
        if d == 0.0:
            continue
        w = d * node_weights(kernels, k, energy, flux, rate_weights)
        rows = slice(kernels.row_start[k], kernels.row_start[k + 1])
        values += w @ kernels.rows[rows]
        lost += float(w @ kernels.lost[rows])
    return values, lost


def fission_source(ops, kernels, densities, flux):
    """Fission emission per unit lethargy at the fine nodes, for solve_energy's source."""
    if not np.array_equal(kernels.coarse, ops.inelastic.coarse):
        raise ValueError("fission kernels must share the inelastic kernels' coarse grid")
    values, _ = fission_hats(kernels, densities, ops.energy, flux, ops.rate_weights)
    m = ops.node_m
    theta = ops.node_theta
    return (1.0 - theta) * values[m] / kernels.hat[m] + theta * values[m + 1] / kernels.hat[m + 1]
