"""Study 08: where the energy solver's time goes, and a regression baseline for speeding it up.

Usage:
    bench.py kernel <nuclide> [...]           elastic-only march per nuclide, by anisotropy mode
    bench.py case <sfr|pwr> [rtol] [subdivide] [--k] [--save NAME] [--compare NAME]

kernel marches one nuclide's elastic scattering alone on a fixed lethargy grid from 1 keV to
20 MeV, with and without its angle tables, and reports the anisotropic correction's cost per node.

case builds study 03's mix (default rtol 1e-2, subdivide 1: 244k nodes for the SFR), timing each
stage of build_mix, then times one march of each kind (isotropic elastic only, no inelastic, full)
and a fixed-source solve from the fission spectrum of a flat flux; --k also solves k. --save writes
F, k and timings to results/NAME.npz, and --compare diffs against a saved run.

Set ZEROG_KERNEL_CACHE to keep kernels between runs; delete it when the study is done.
"""

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "03-k-infinity" / "scripts"))
from cases import CASES  # noqa: E402

import zerog.energy.mix as mix_module  # noqa: E402
from zerog.energy.eigen import solve_k  # noqa: E402
from zerog.energy.elastic import thin_angle  # noqa: E402
from zerog.energy.fission import fission_source  # noqa: E402
from zerog.energy.subproblem import build_operators, march, solve_energy  # noqa: E402
from zerog.energy.terms import _elastic_angle, awr  # noqa: E402

RESULTS = HERE / "results"
CACHE = os.environ.get("ZEROG_KERNEL_CACHE")
BUILD_STAGES = (
    "transport_grid",
    "load_term",
    "apply_thermal",
    "_order_rows",
    "build_inelastic",
    "build_thermal",
    "build_fission",
    "build_urr",
    "build_operators",
)


def timed(label, function, *args, **kwargs):
    start = time.perf_counter()
    out = function(*args, **kwargs)
    return out, time.perf_counter() - start


@contextmanager
def stage_timers(timings):
    """Wrap build_mix's stages, as names in zerog.energy.mix, with timers."""
    saved = {}
    for name in BUILD_STAGES:
        original = getattr(mix_module, name, None)
        if original is None:
            continue
        saved[name] = original

        def wrapper(*args, _original=original, _name=name, **kwargs):
            start = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                timings[_name] = timings.get(_name, 0.0) + time.perf_counter() - start

        setattr(mix_module, name, wrapper)
    try:
        yield
    finally:
        for name, original in saved.items():
            setattr(mix_module, name, original)


def kernel(names, spacing=5e-4):
    energy = 2e7 * np.exp(-np.arange(0.0, np.log(2e7 / 1e3), spacing))
    n = energy.shape[0]
    source = np.zeros(n)
    source[:5] = 1.0
    for name in names:
        a = awr(name)
        alpha = np.array([((a - 1.0) / (a + 1.0)) ** 2])
        angle = thin_angle(_elastic_angle(name), None)
        ones = np.ones((1, n))
        iso = build_operators(energy, 2.0 * ones, ones, alpha)
        aniso = build_operators(energy, 2.0 * ones, ones, alpha, angle=[angle])
        lagged = np.zeros((1, aniso.inelastic.coarse.shape[0]))
        above = int((energy > aniso.angle.cutoff[0]).sum())
        row = {"nuclide": name, "nodes": n, "above_cutoff": above}
        for label, ops, mode in (
            ("iso", iso, "separable"),
            ("tree", aniso, "tree"),
            ("separable", aniso, "separable"),
        ):
            march(ops, [2.0], [[1.0]], 0.0, source, lagged, 0, mode)  # compile
            best = min(
                timed(label, march, ops, [2.0], [[1.0]], 0.0, source, lagged, 0, mode)[1]
                for _ in range(3)
            )
            row[label] = best
        for label in ("tree", "separable"):
            row[f"us_per_node_{label}"] = (row[label] - row["iso"]) / max(above, 1) * 1e6
        print(json.dumps(row), flush=True)


def case(name, rtol, subdivide, k=False, save=None, compare=None):
    spec = CASES[name]
    builds = {}
    with stage_timers(builds):
        mix, built = timed(
            "build",
            mix_module.build_mix,
            spec["densities"],
            spec["label"],
            energy_min=spec["energy_min"],
            sab=spec["sab"],
            rtol=rtol,
            subdivide=subdivide,
            cache=CACHE,
        )
    ops = mix.ops
    d = mix.densities
    n = mix.energy.shape[0]
    print(f"{name}: {n} nodes, build {built:.1f} s", flush=True)
    for stage, seconds in builds.items():
        print(f"  build {stage}: {seconds:.1f} s", flush=True)
    source = fission_source(ops, mix.fission, d, np.ones(n))
    n_coarse = ops.inelastic.coarse.shape[0]
    lagged = np.zeros((d.shape[0], n_coarse))
    iso = build_operators(ops.energy, ops.removal, ops.scatter, ops.alpha, None, ops.inelastic)
    bare = build_operators(
        ops.energy, ops.removal, ops.scatter, ops.alpha, [t.angle for t in mix.terms]
    )
    # The first march compiles _march, which is not cached.
    _, compile_s = timed("compile", march, ops, d, d[:, None], 0.0, source, lagged)
    marches = {"first_march_with_compile": compile_s}
    for label, variant in (("isotropic", iso), ("no_inelastic", bare), ("full", ops)):
        lag = np.zeros((d.shape[0], variant.inelastic.coarse.shape[0]))
        marches[label] = timed(label, march, variant, d, d[:, None], 0.0, source, lag)[1]
    for label, seconds in marches.items():
        print(f"  march {label}: {seconds:.2f} s", flush=True)
    solution, solved = timed("solve", solve_energy, ops, d, d[:, None], 0.0, source)
    print(
        f"  fixed-source solve {solved:.1f} s, {solution.sweeps} sweeps, "
        f"{solution.block_marches} block marches",
        flush=True,
    )
    record = {
        "case": name,
        "rtol": rtol,
        "subdivide": subdivide,
        "nodes": n,
        "build_s": built,
        "build_stages": builds,
        "marches_s": marches,
        "solve_s": solved,
        "sweeps": solution.sweeps,
        "block_marches": solution.block_marches,
    }
    arrays = {"energy": mix.energy, "flux": solution.flux}
    if k:
        eigen, k_s = timed("k", solve_k, ops, mix.fission, d, d[:, None], 0.0, d)
        print(
            f"  k {eigen.k:.9f} in {k_s:.1f} s, {eigen.iterations} iterations, "
            f"{eigen.sweeps} sweeps, {eigen.block_marches} block marches",
            flush=True,
        )
        record.update(
            k=eigen.k,
            k_s=k_s,
            k_iterations=eigen.iterations,
            k_sweeps=eigen.sweeps,
            k_block_marches=eigen.block_marches,
        )
        arrays["k_flux"] = eigen.flux
    if save:
        RESULTS.mkdir(exist_ok=True)
        np.savez(RESULTS / f"{save}.npz", **arrays)
        (RESULTS / f"{save}.json").write_text(json.dumps(record, indent=1))
    if compare:
        diff(compare, record, arrays)
    return record


def diff(name, record, arrays):
    old = np.load(RESULTS / f"{name}.npz")
    old_record = json.loads((RESULTS / f"{name}.json").read_text())
    if not np.array_equal(old["energy"], arrays["energy"]):
        print(f"  compare {name}: grids differ")
        return
    for key in ("flux", "k_flux"):
        if key in old.files and key in arrays:
            a, b = arrays[key], old[key]
            scale = np.abs(b).max()
            relative = np.abs(a - b) / np.maximum(np.abs(b), 1e-300 * scale)
            print(f"  compare {name} {key}: max relative {relative.max():.2e}")
    if "k" in old_record and "k" in record:
        print(f"  compare {name} k: {(record['k'] - old_record['k']) / record['k'] * 1e5:+.2e} pcm")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "kernel":
        kernel(args[1:])
        return
    flags = {}
    for flag in ("--save", "--compare"):
        if flag in args:
            i = args.index(flag)
            flags[flag[2:]] = args[i + 1]
            args = args[:i] + args[i + 2 :]
    k = "--k" in args
    args = [a for a in args if a != "--k"]
    rtol = float(args[2]) if len(args) > 2 else 1e-2
    subdivide = int(args[3]) if len(args) > 3 else 1
    case(args[1], rtol, subdivide, k, **flags)


if __name__ == "__main__":
    main()
