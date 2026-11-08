"""Study 04: grid error with negative effective densities.

Usage: negative.py <curves> <rtol>, or negative.py reference.

A PGD energy subproblem takes effective densities per term, which can be negative, so a grid that
holds each cross section within rtol no longer bounds a sum of them relatively. This solves study
03's SFR mix as a fixed-source problem twice on one grid: with the physical densities, and with each
term's density times a sign pattern (SIGNS), both with a streaming term that keeps the diagonal
positive and the fission spectrum of a flat flux as the source. Each saves its capture rates and
flux integral; summarize compares them with the reference grid's (all reactions, 1e-5).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "03-k-infinity" / "scripts"))
from cases import CASES  # noqa: E402

from zerog.energy.fission import fission_source  # noqa: E402
from zerog.energy.grid import panel_rate  # noqa: E402
from zerog.energy.mix import build_mix  # noqa: E402
from zerog.energy.subproblem import solve_energy  # noqa: E402

RESULTS = HERE / "results"
CACHE = HERE / "results" / "kernels"
CASE = "sfr"
# A PGD correction mode: fuel and sodium add, oxygen and two steel isotopes subtract.
SIGNS = {"O16": -0.6, "Fe56": -0.8, "Ni58": -1.0, "Cr52": 0.5}
STREAMING_MARGIN = 1.5


def solve(curves, rtol):
    case = CASES[CASE]
    mix = build_mix(
        case["densities"],
        case["label"],
        energy_min=case["energy_min"],
        rtol=rtol,
        curves=curves,
        cache=CACHE,
    )
    ops = mix.ops
    d = mix.densities
    weights = np.array([SIGNS.get(n, 1.0) for n in mix.names])
    # Streaming that keeps a + sum d sigma_t above the negative terms' share everywhere.
    negative = np.clip(-(d * weights)[:, None] * ops.removal, 0.0, None).sum(axis=0)
    streaming = STREAMING_MARGIN * negative.max()
    source = fission_source(ops, mix.fission, d, np.ones(mix.energy.shape[0]))
    out = {
        "curves": curves,
        "rtol": rtol,
        "nodes": int(mix.energy.shape[0]),
        "streaming": streaming,
    }
    for label, effective in (("positive", d), ("negative", d * weights)):
        solution = solve_energy(ops, effective, effective[:, None], streaming, source)
        if np.any(solution.diagonal <= 0.0):
            raise RuntimeError(f"{label}: a non-positive diagonal")
        out[label] = {
            "flux": panel_rate(np.ones(mix.energy.shape[0]), solution.flux, ops.rate_weights),
            "capture": {
                n: panel_rate(mix.terms[i].capture, solution.flux, ops.rate_weights)
                for i, n in enumerate(mix.names)
            },
            "sweeps": solution.sweeps,
        }
    return out


def summarize():
    reference = json.loads((RESULTS / "negative_all_1e-05.json").read_text())
    print("curves rtol nodes | largest capture error, positive | negative | flux pos, neg")
    for path in sorted(RESULTS.glob("negative_*.json")):
        run = json.loads(path.read_text())
        line = f"{run['curves']:11s} {run['rtol']:.0e} {run['nodes']:8d} |"
        flux = []
        for label in ("positive", "negative"):
            ref = reference[label]
            errors = {n: run[label]["capture"][n] / ref["capture"][n] - 1.0 for n in ref["capture"]}
            worst = max(errors, key=lambda n: abs(errors[n]))
            line += f" {worst} {errors[worst] * 100:+.4f}% |"
            flux.append(run[label]["flux"] / ref["flux"] - 1.0)
        print(line + f" {flux[0] * 100:+.4f}%, {flux[1] * 100:+.4f}%")


def main():
    if sys.argv[1] == "summary":
        summarize()
        return
    curves, rtol = (
        ("all", 1e-5) if sys.argv[1] == "reference" else (sys.argv[1], float(sys.argv[2]))
    )
    out = solve(curves, rtol)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"negative_{curves}_{rtol:.0e}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("positive", "negative")}))


if __name__ == "__main__":
    main()
