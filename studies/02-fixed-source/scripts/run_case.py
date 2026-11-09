"""ZeroG side of study 02. Usage: run_case.py <case> [rtol] [subdivide]. Compares with the twin."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from cases import CASES, ENERGY_MIN, TEMPERATURE

from zerog.energy.balance import energy_balance
from zerog.energy.elastic import point_emission
from zerog.energy.grid import panel_rate
from zerog.energy.inelastic import build_inelastic, interpolate_hats, point_emission_coarse
from zerog.energy.subproblem import build_operators, solve_energy
from zerog.energy.terms import awr, elastic_grid, load_term, transport_grid

TWIN = Path(__file__).parents[3] / "benchmarks" / "02-elastic-fixed-source"


def run(name, rtol=1e-3, subdivide=1):
    case = CASES[name]
    e0 = case["source_energy"]
    densities = case["densities"]
    nuclides = [(n, TEMPERATURE) for n in densities]
    fast = name.endswith("-fast")
    if fast:
        grid, coarse = transport_grid(nuclides, rtol, 1e-8, e0, ENERGY_MIN, subdivide)
    else:
        grid = elastic_grid(nuclides, rtol, 1e-8, e0, ENERGY_MIN, subdivide)
    summary = json.loads((TWIN / f"{name}.json").read_text())
    edges = np.array(summary["edges_low"])
    grid = np.unique(np.concatenate([grid, edges[edges > ENERGY_MIN]]))[::-1]
    terms = [load_term(n, TEMPERATURE, grid) for n in densities]
    d = np.array([densities[n] for n in densities])
    start = time.perf_counter()
    inelastic = (
        build_inelastic(
            [(n, TEMPERATURE, awr(n)) for n in densities],
            grid,
            coarse,
            # Rebuilt each run unless ZEROG_KERNEL_CACHE names a directory to keep them in.
            cache=os.environ.get("ZEROG_KERNEL_CACHE"),
        )
        if fast
        else None
    )
    kernels = time.perf_counter() - start
    ops = build_operators(
        grid,
        np.array([t.total for t in terms]),
        np.array([t.elastic for t in terms]),
        [t.alpha for t in terms],
        [t.angle for t in terms],
        inelastic,
    )
    # The source is the first collision: one neutron at e0 has not been slowed yet.
    total0 = float(d @ ops.removal[:, 0])
    source = np.zeros_like(grid)
    for density, term in zip(d, terms, strict=True):
        weight = density * term.elastic[0] / total0
        source += weight * point_emission(grid, e0, term.alpha, term.angle)
        if fast:
            row, emission = point_emission_coarse(
                term.name, TEMPERATURE, awr(term.name), e0, coarse
            )
            source += density * emission / total0 * interpolate_hats(ops, row)
    start = time.perf_counter()
    solution = solve_energy(ops, d, d[:, None], 0.0, source)
    elapsed = time.perf_counter() - start
    flux = solution.flux
    balance = energy_balance(ops, d, d[:, None], 0.0, source, flux)

    closure = balance.residual / balance.source
    print(
        f"{name}: {grid.shape[0]} nodes, {elapsed:.2f} s, {solution.sweeps} sweeps, "
        f"kernels {kernels:.0f} s, residual/source {closure:.2e}"
    )
    uncollided = 1.0 / total0
    for row in summary["rates"]:
        if row["score"] != "(n,gamma)":
            continue
        i = list(densities).index(row["nuclide"])
        ours = densities[row["nuclide"]] * (
            panel_rate(terms[i].capture, flux, ops.rate_weights) + terms[i].capture[0] * uncollided
        )
        diff = ours / row["mean"] - 1.0
        sigmas = (ours - row["mean"]) / row["std. dev."]
        print(
            f"  capture {row['nuclide']:6s} {ours:.6f} vs {row['mean']:.6f} "
            f"+- {row['std. dev.']:.6f}: {diff * 100:+.3f}% ({sigmas:+.1f} sd)"
        )

    low = np.array(summary["edges_low"])
    high = np.array(summary["edges_high"])
    twin = np.array(summary["flux"])
    std = np.array(summary["flux_std"])
    ours = np.empty_like(twin)
    for b in range(low.shape[0]):
        mask = (grid <= min(high[b], e0)) & (grid >= low[b])
        piece = grid[mask]
        lethargy = np.log(piece[:-1] / piece[1:])
        ours[b] = np.sum(0.5 * (flux[mask][:-1] + flux[mask][1:]) * lethargy)
    # The top bin holds the uncollided flight.
    ours[-1] += uncollided
    z = (ours - twin) / std
    relative = ours / twin - 1.0
    print(
        f"  flux bins {low.shape[0]}: rms z {np.sqrt(np.mean(z**2)):.2f}, "
        f"max |z| {np.abs(z).max():.1f}, median sd {np.median(std / twin) * 100:.3f}%, "
        f"max |rel| {np.abs(relative).max() * 100:.3f}%"
    )
    worst = np.argsort(np.abs(z))[-3:]
    for b in worst:
        print(f"    bin {low[b]:.4g}-{high[b]:.4g} eV: {relative[b] * 100:+.3f}%, z {z[b]:+.1f}")
    total_diff = ours.sum() / twin.sum() - 1.0
    print(f"  total flux {total_diff * 100:+.3f}%")
    return grid, flux


if __name__ == "__main__":
    args = sys.argv[1:]
    run(args[0], float(args[1]) if len(args) > 1 else 1e-3, int(args[2]) if len(args) > 2 else 1)
