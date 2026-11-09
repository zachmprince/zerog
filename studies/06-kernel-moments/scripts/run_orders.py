"""Study 06: how many Legendre orders the SFR mix's forward peak needs.

Usage: run_orders.py <rtol> <subdivide> [kappa ...] [--orders 0,1,2,5]; L_MAX is always run.

The energy subproblem of an angular mode X(mu) = exp(kappa mu), projected onto itself, has removal
density N int X^2 and order-l scattering density N (2l + 1) / 2 X_l^2, with X_l = int P_l X. This
solves study 03's SFR mix, cut off at 50 eV, as a fixed-source problem (the fission spectrum of a
flat flux) with no streaming, truncated at each order, and compares each with L_MAX. Writes
results/orders_<rtol>_<subdivide>.json.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "03-k-infinity" / "scripts"))
from cases import CASES  # noqa: E402

from zerog.energy.fission import fission_source  # noqa: E402
from zerog.energy.grid import panel_rate  # noqa: E402
from zerog.energy.mix import build_mix  # noqa: E402
from zerog.energy.moments import lab_moments  # noqa: E402
from zerog.energy.subproblem import solve_energy  # noqa: E402
from zerog.energy.terms import awr  # noqa: E402

RESULTS = HERE / "results"
# Kernels are rebuilt each run (minutes), not cached: caches ran to tens of GB per study.
# Set ZEROG_KERNEL_CACHE to a directory to keep them.
CACHE = os.environ.get("ZEROG_KERNEL_CACHE")
L_MAX = 5


def mode_coefficients(kappa):
    """(removal, order coefficients) of X = exp(kappa mu), per unit density."""
    moments = [
        quad(lambda m: np.exp(kappa * m) * eval_legendre(order, m), -1.0, 1.0, epsabs=1e-14)[0]  # noqa: B023
        for order in range(L_MAX + 1)
    ]
    removal = quad(lambda m: np.exp(2.0 * kappa * m), -1.0, 1.0)[0]
    return removal, np.array([(2 * order + 1) / 2.0 * x * x for order, x in enumerate(moments)])


def main():
    args = sys.argv[1:]
    truncations = list(range(L_MAX + 1))
    if "--orders" in args:
        i = args.index("--orders")
        truncations = sorted({int(x) for x in args[i + 1].split(",")} | {L_MAX})
        args = args[:i] + args[i + 2 :]
    rtol, subdivide = float(args[0]), int(args[1])
    kappas = [float(k) for k in args[2:]] or [1.0, 3.0]
    case = CASES["sfr"]
    start = time.perf_counter()
    mix = build_mix(
        case["densities"],
        case["label"],
        energy_min=case["energy_min"],
        rtol=rtol,
        subdivide=subdivide,
        cache=CACHE,
        orders=L_MAX,
    )
    built = time.perf_counter() - start
    print(f"built {mix.energy.shape[0]} nodes in {built:.0f} s", flush=True)
    ops = mix.ops
    d = mix.densities
    source = fission_source(ops, mix.fission, d, np.ones(mix.energy.shape[0]))
    unit = np.ones(mix.energy.shape[0])
    nodes = int(mix.energy.shape[0])
    record = {"rtol": rtol, "subdivide": subdivide, "nodes": nodes, "build_s": built}
    # The forward peak itself: lab moments of each nuclide's elastic scattering.
    energies = [1e4, 1e5, 1e6, 5e6]
    record["lab_moments"] = {
        n: lab_moments(mix.terms[i].angle, awr(n), energies, L_MAX).tolist()
        for i, n in enumerate(mix.names)
    }
    for kappa in kappas:
        removal, coefficients = mode_coefficients(kappa)
        runs = {}
        for order in truncations:
            scatter = np.outer(d, coefficients[: order + 1])
            t0 = time.perf_counter()
            solution = solve_energy(ops, d * removal, scatter, 0.0, source)
            elapsed = time.perf_counter() - t0
            runs[order] = {
                "solve_s": elapsed,
                "flux": panel_rate(unit, solution.flux, ops.rate_weights),
                "capture": {
                    n: panel_rate(mix.terms[i].capture, solution.flux, ops.rate_weights)
                    for i, n in enumerate(mix.names)
                },
                "fission": {
                    n: panel_rate(mix.fission_xs(i), solution.flux, ops.rate_weights)
                    for i, n in enumerate(mix.names)
                    if mix.fission_xs(i).any()
                },
            }
            print(
                f"kappa {kappa} L {order}: solve {elapsed:.0f} s, {solution.sweeps} sweeps",
                flush=True,
            )
        reference = runs[L_MAX]
        for order in truncations[:-1]:
            run = runs[order]
            errors = {
                f"{kind} {n}": run[kind][n] / reference[kind][n] - 1.0
                for kind in ("capture", "fission")
                for n in reference[kind]
                if reference[kind][n] > 0.0
            }
            worst = max(errors, key=lambda key: abs(errors[key]))
            run["worst"] = [worst, errors[worst]]
            run["flux_error"] = run["flux"] / reference["flux"] - 1.0
            print(
                f"  kappa {kappa} L {order} vs {L_MAX}: flux {run['flux_error']:+.3e}, "
                f"worst {worst} {errors[worst]:+.3e}"
            )
        record[f"kappa {kappa}"] = {
            "removal": removal,
            "coefficients": (coefficients / removal).tolist(),
            "runs": runs,
        }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"orders_{args[0]}_{subdivide}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
