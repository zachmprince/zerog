"""Study 05: the SFR mix with probability tables as bands, against the twin with tables on.

Usage: run_bands.py <bands> <rtol> <subdivide> [twin]. bands is the number of Sobol combinations
(a power of two), or 0 for infinite dilution; twin is the summary to compare with, sfr-ptables by
default. Writes results/sfr_<bands>_<rtol>_<subdivide>_<twin>.json, gitignored.
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
    bands, rtol, subdivide = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    twin = sys.argv[4] if len(sys.argv) > 4 else "sfr-ptables"
    mix, solution, comparison, closure, built, solved = solve_case(
        "sfr", rtol, subdivide, bands=bands, summary=twin
    )
    label = f"sfr_{bands}_{sys.argv[2]}_{subdivide}_{twin}"
    t = comparison.tallied
    record = {
        "bands": bands,
        "rtol": rtol,
        "subdivide": subdivide,
        "twin": twin,
        "nodes": int(mix.energy.shape[0]),
        "build_s": built,
        "solve_s": solved,
        "fixed_source_s": mix.timings.get("fixed_source"),
        "fixed_source_sweeps": mix.timings.get("fixed_source_sweeps"),
        "sweeps": solution.sweeps,
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
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{label}.json").write_text(json.dumps(record, indent=1))
    print(
        f"{label}: {record['nodes']} nodes, {comparison.pcm:+.1f} pcm, solve {solved:.0f} s, "
        f"residual {closure:.1e}"
    )
    print(comparison.report())


if __name__ == "__main__":
    main()
