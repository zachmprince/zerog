"""ZeroG side of study 03. Usage: run_k.py <case> [rtol] [subdivide] [light]. Compares with the
twin. light sets the free gas threshold, in kT, for targets of awr <= 1, which OpenMC treats as free
gas at every energy; zerog.energy.thermal.LIGHT_FREE_GAS_THRESHOLD by default."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from cases import CASES

from zerog.analysis.twin import compare
from zerog.energy.balance import energy_balance
from zerog.energy.eigen import solve_k
from zerog.energy.fission import fission_source
from zerog.energy.mix import build_mix

TWIN = Path(__file__).parents[3] / "benchmarks" / "05-fast-k-infinity"
CACHE = Path(__file__).parent / "results" / "kernels"


def solve_case(
    name, rtol=1e-3, subdivide=1, light=None, curves="all", twin=TWIN, bands=0, summary=None
):
    """Build a case's mix, solve k, and compare it with the twin. Returns (mix, solution,
    comparison, closure, build seconds, solve seconds).

    bands turns probability tables on with that many combinations (zerog.energy.mix.build_mix).
    summary names the twin's summary, by default the case's own, such as "sfr-ptables".
    """
    case = CASES[name]
    summary = json.loads((Path(twin) / f"{summary or name}.json").read_text())
    start = time.perf_counter()
    mix = build_mix(
        case["densities"],
        case["label"],
        energy_min=case["energy_min"],
        sab=case["sab"],
        rtol=rtol,
        subdivide=subdivide,
        curves=curves,
        # Twin bin edges are nodes, so each bin's flux is an exact sum of panels.
        required=np.array(summary["edges_low"]),
        light=light,
        cache=CACHE,
        bands=bands,
    )
    built = time.perf_counter() - start
    d = mix.densities
    start = time.perf_counter()
    solution = solve_k(mix.ops, mix.fission, d, d[:, None], 0.0, d)
    solved = time.perf_counter() - start
    fission = solution.fission if solution.fission is not None else mix.fission
    source = fission_source(mix.ops, fission, d, solution.flux) / solution.k
    balance = energy_balance(mix.ops, d, d[:, None], 0.0, source, solution.flux, solution.shield)
    comparison = compare(mix, solution.k, solution.flux, summary, solution.shield, fission)
    return mix, solution, comparison, balance.residual / balance.source, built, solved


def run(name, rtol=1e-3, subdivide=1, light=None, curves="all"):
    mix, solution, comparison, closure, built, solved = solve_case(
        name, rtol, subdivide, light, curves
    )
    print(
        f"{name}: {mix.energy.shape[0]} nodes, build {built:.0f} s, solve {solved:.0f} s, "
        f"{solution.iterations} iterations, {solution.sweeps} sweeps, "
        f"{solution.block_marches} block marches, residual/source {closure:.2e}"
    )
    print(comparison.report().split("\n")[0])
    print("  k history " + " ".join(f"{k:.7f}" for k in solution.history))
    print("\n".join(comparison.report().split("\n")[1:]))
    return mix, solution, comparison


if __name__ == "__main__":
    args = sys.argv[1:]
    run(
        args[0],
        float(args[1]) if len(args) > 1 else 1e-3,
        int(args[2]) if len(args) > 2 else 1,
        float(args[3]) if len(args) > 3 else None,
    )
