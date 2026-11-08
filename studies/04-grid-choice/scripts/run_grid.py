"""Study 04: k-infinity and rates of study 03's mixes on one grid, against study 03's twins.

Usage: run_grid.py <case> <curves> <rtol> <subdivide>. curves is all, major or macroscopic
(zerog.energy.terms.thinned_grids). Writes results/<case>_<curves>_<rtol>_<subdivide>.json and the
flux, as .npz, for certify.py; both are gitignored.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "03-k-infinity" / "scripts"))
from run_k import solve_case  # noqa: E402

RESULTS = HERE / "results"


def main():
    name, curves, rtol, subdivide = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    result = solve_case(name, rtol, subdivide, None, curves)
    mix, solution, comparison, closure, built, solved = result
    label = f"{name}_{curves}_{sys.argv[3]}_{subdivide}"
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / f"{label}.npz", energy=mix.energy, flux=solution.k * solution.flux)
    t = comparison.tallied
    record = {
        "case": name,
        "curves": curves,
        "rtol": rtol,
        "subdivide": subdivide,
        "nodes": int(mix.energy.shape[0]),
        "coarse": int(mix.coarse.shape[0]),
        "build_s": built,
        "solve_s": solved,
        "sweeps": solution.sweeps,
        "block_marches": solution.block_marches,
        "residual": closure,
        "k": solution.k,
        "pcm": comparison.pcm,
        "k_std_pcm": comparison.k_std * 1e5,
        "rates": [
            {
                "score": r.score,
                "nuclide": r.nuclide,
                "relative": r.relative,
                "sigmas": r.sigmas,
                "twin_relative_std": r.std / r.twin,
            }
            for r in comparison.rates
        ],
        "flux_rms_z": float(np.sqrt(np.mean(comparison.z[t] ** 2))),
        "flux_max_z": float(np.abs(comparison.z).max()),
    }
    (RESULTS / f"{label}.json").write_text(json.dumps(record, indent=1))
    print(f"{label}: {record['nodes']} nodes, {comparison.pcm:+.1f} pcm, solve {solved:.0f} s")
    print(comparison.report())


if __name__ == "__main__":
    main()
