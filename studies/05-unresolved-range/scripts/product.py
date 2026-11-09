"""Study 05: Sobol band combinations against the full product of strata, for two nuclides.

U238 and Pu239 in sodium at 900 K, as a fixed-source problem from the fission spectrum of a flat
flux, cut off at 50 eV. Each nuclide's r is split into STRATA equal strata, and every pair of
midpoints is a combination (STRATA^2 of them); Sobol sets of 16 to 256 points are compared with it
on each capture and fission rate and the flux in the tables' range. All runs share one grid and its
kernels. Writes results/product.json, gitignored.
"""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from zerog.energy.fission import fission_source
from zerog.energy.grid import panel_rate
from zerog.energy.mix import build_mix
from zerog.energy.subproblem import solve_energy
from zerog.energy.urr import (
    SHIELD_CAPTURE,
    SHIELD_FISSION,
    band_points,
    product_points,
    shielded,
    with_points,
)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
DENSITIES = {"U238": 6.455e-3, "Pu239": 9.222e-4, "Na23": 8.907e-3}
STRATA = 64


def rates(mix, solution):
    ops = mix.ops
    out = {}
    for i, name in enumerate(mix.names[:2]):
        capture = shielded(
            ops.urr, solution.shield, i, SHIELD_CAPTURE, mix.terms[i].capture, mix.energy
        )
        fission = shielded(
            ops.urr, solution.shield, i, SHIELD_FISSION, mix.fission_xs(i), mix.energy
        )
        out[f"{name} capture"] = panel_rate(capture, solution.flux, ops.rate_weights)
        out[f"{name} fission"] = panel_rate(fission, solution.flux, ops.rate_weights)
    inside = (mix.energy >= 2.5e3) & (mix.energy <= 1.49e5)
    out["flux 2.5-149 keV"] = panel_rate(inside.astype(float), solution.flux, ops.rate_weights)
    return out


def main():
    rtol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
    mix = build_mix(DENSITIES, "900K", energy_min=50.0, rtol=rtol, bands=16)
    d = mix.densities
    source = fission_source(mix.ops, mix.fission, d, np.ones(mix.energy.shape[0]))
    results = {}
    sets = {"none": None, f"product {STRATA}^2": product_points(STRATA, 2)}
    for m in (16, 64, 256):
        sets[f"sobol {m}"] = (band_points(2, m), np.full(m, 1.0 / m))
    for label, points in sets.items():
        if points is None:
            ops = replace(mix.ops, urr=replace(mix.ops.urr, terms=mix.ops.urr.terms[:0]))
        else:
            ops = replace(mix.ops, urr=with_points(mix.ops.urr, *points))
        start = time.perf_counter()
        solution = solve_energy(ops, d, d[:, None], 0.0, source)
        results[label] = rates(replace(mix, ops=ops), solution)
        results[label]["solve_s"] = time.perf_counter() - start
        print(label, f"{results[label]['solve_s']:.0f} s", flush=True)
    reference = results[f"product {STRATA}^2"]
    for label, values in results.items():
        line = " ".join(
            f"{key} {values[key] / reference[key] - 1.0:+.4%}"
            for key in reference
            if key != "solve_s"
        )
        print(f"{label:16s} {line}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "product.json").write_text(
        json.dumps({"nodes": int(mix.energy.shape[0]), **results}, indent=1)
    )


if __name__ == "__main__":
    main()
