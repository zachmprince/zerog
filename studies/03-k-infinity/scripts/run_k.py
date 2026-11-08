"""ZeroG side of study 03. Usage: run_k.py <case> [rtol] [subdivide] [light]. Compares with the
twin. light sets the free gas threshold, in kT, for targets of awr <= 1, which OpenMC treats as free
gas at every energy; zerog.energy.thermal.LIGHT_FREE_GAS_THRESHOLD by default."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from cases import CASES, LIBRARY_BOTTOM

from zerog.data.neutron import MT_FISSION, MT_PARTIAL_FISSION, read_reactions
from zerog.energy.balance import energy_balance
from zerog.energy.eigen import solve_k
from zerog.energy.fission import (
    build_fission,
    data_top,
    fission_nodes,
    fission_source,
    production_rates,
)
from zerog.energy.grid import interpolate, panel_rate
from zerog.energy.inelastic import build_inelastic
from zerog.energy.subproblem import build_operators
from zerog.energy.terms import (
    apply_thermal,
    awr,
    load_term,
    thermal_grid_settings,
    thermal_required,
    thermal_spec,
    transport_grid,
)
from zerog.energy.thermal import build_thermal

TWIN = Path(__file__).parents[3] / "benchmarks" / "05-fast-k-infinity"
CACHE = Path(__file__).parent / "results" / "kernels"


def fission_xs(name, label, energy):
    data = read_reactions(name, label)
    selected = data.xs[np.isin(data.mt, (MT_FISSION, *MT_PARTIAL_FISSION))].sum(axis=0)
    return interpolate(energy, data.energy, selected)


def run(name, rtol=1e-3, subdivide=1, light=None):
    case = CASES[name]
    densities = case["densities"]
    label = case["label"]
    energy_min = case["energy_min"] or LIBRARY_BOTTOM
    nuclides = [(n, label) for n in densities]
    top = data_top(nuclides)
    summary = json.loads((TWIN / f"{name}.json").read_text())
    edges = np.array(summary["edges_low"])
    # Thermal scattering, where the grid reaches below a free gas top.
    specs = [
        thermal_spec(n, label, case["sab"].get(n), light if awr(n) <= 1.0 else None)
        for n in densities
    ]
    specs = [s if energy_min < s.free_top else None for s in specs]
    thermal = [s for s in specs if s is not None]
    required = np.concatenate(
        [
            fission_nodes(nuclides, top, energy_min),
            edges[edges > energy_min],
            *[thermal_required(s, energy_min) for s in thermal],
        ]
    )
    grid, coarse = transport_grid(
        nuclides,
        rtol,
        1e-8,
        top,
        energy_min,
        subdivide,
        required=required,
        thermal=thermal_grid_settings(thermal) if thermal else None,
    )
    terms = [load_term(n, label, grid) for n in densities]
    terms = [
        t if s is None else apply_thermal(t, s, grid) for t, s in zip(terms, specs, strict=True)
    ]
    d = np.array([densities[n] for n in densities])
    start = time.perf_counter()
    inelastic = build_inelastic(
        [(n, label, awr(n)) for n in densities],
        grid,
        coarse,
        cache=CACHE,
    )
    if thermal:
        inelastic = build_thermal(specs, grid, inelastic, cache=CACHE)
    fission = build_fission(nuclides, grid, coarse, top)
    kernels = time.perf_counter() - start
    ops = build_operators(
        grid,
        np.array([t.total for t in terms]),
        np.array([t.elastic for t in terms]),
        [t.alpha for t in terms],
        [t.angle for t in terms],
        inelastic,
    )
    start = time.perf_counter()
    solution = solve_k(ops, fission, d, d[:, None], 0.0, d)
    elapsed = time.perf_counter() - start
    flux = solution.flux
    source = fission_source(ops, fission, d, flux) / solution.k
    balance = energy_balance(ops, d, d[:, None], 0.0, source, flux)
    closure = balance.residual / balance.source

    k_twin, k_std = summary["k"], summary["k_std"]
    print(
        f"{name}: {grid.shape[0]} nodes, kernels {kernels:.0f} s, solve {elapsed:.0f} s, "
        f"{solution.iterations} iterations, {solution.sweeps} sweeps, "
        f"{solution.block_marches} block marches, residual/source {closure:.2e}"
    )
    print(
        f"  k {solution.k:.6f} vs {k_twin:.6f} +- {k_std:.6f}: "
        f"{(solution.k - k_twin) * 1e5:+.1f} pcm"
    )
    print("  k history " + " ".join(f"{k:.7f}" for k in solution.history))

    # OpenMC's tallies are per source neutron. This flux produces one fission neutron, which is
    # emitted as 1 / k, so it is k times the twin's.
    flux_twin = solution.k * flux
    curves = {
        "(n,gamma)": lambda i: terms[i].capture,
        "fission": lambda i: fission_xs(terms[i].name, label, grid),
    }
    produced = production_rates(fission, d, grid, flux_twin, ops.rate_weights)
    for row in summary["rates"]:
        if row["nuclide"] not in densities:
            continue
        i = list(densities).index(row["nuclide"])
        if row["score"] in curves:
            ours = d[i] * panel_rate(curves[row["score"]](i), flux_twin, ops.rate_weights)
        elif row["score"] == "nu-fission":
            ours = produced[i]
        else:
            continue
        if row["mean"] <= 0.0:
            continue
        diff = ours / row["mean"] - 1.0
        sigmas = (ours - row["mean"]) / row["std. dev."]
        print(
            f"  {row['score']:10s} {row['nuclide']:6s} {ours:.6e} vs {row['mean']:.6e} "
            f"({row['std. dev.'] / row['mean'] * 100:.3f}%): {diff * 100:+.3f}% ({sigmas:+.1f} sd)"
        )

    low = np.array(summary["edges_low"])
    high = np.array(summary["edges_high"])
    twin = np.array(summary["flux"])
    std = np.array(summary["flux_std"])
    ours = np.empty_like(twin)
    for b in range(low.shape[0]):
        mask = (grid <= high[b]) & (grid >= low[b])
        piece = grid[mask]
        ours[b] = np.sum(
            0.5 * (flux_twin[mask][:-1] + flux_twin[mask][1:]) * np.log(piece[:-1] / piece[1:])
        )
    # Bins at the ends can be empty in the twin.
    tallied = std > 0.0
    z = np.where(tallied, (ours - twin) / np.where(tallied, std, 1.0), 0.0)
    relative = np.where(tallied, ours / np.where(tallied, twin, 1.0) - 1.0, 0.0)
    print(
        f"  flux bins {tallied.sum()} of {low.shape[0]} tallied: "
        f"rms z {np.sqrt(np.mean(z[tallied] ** 2)):.2f}, max |z| {np.abs(z).max():.1f}, "
        f"median sd {np.median(std[tallied] / twin[tallied]) * 100:.3f}%"
    )
    for b in np.argsort(np.abs(z))[-3:]:
        print(f"    bin {low[b]:.4g}-{high[b]:.4g} eV: {relative[b] * 100:+.3f}%, z {z[b]:+.1f}")
    print(f"  total flux {ours.sum() / twin.sum() * 100 - 100:+.3f}%")
    return grid, flux_twin


if __name__ == "__main__":
    args = sys.argv[1:]
    run(
        args[0],
        float(args[1]) if len(args) > 1 else 1e-3,
        int(args[2]) if len(args) > 2 else 1,
        float(args[3]) if len(args) > 3 else None,
    )
