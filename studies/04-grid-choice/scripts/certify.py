"""Study 04: error of nu-fission production and inelastic emission on a thinned grid.

Usage: certify.py <case> <curves> <rtol> <subdivide>, after run_grid.py has saved that run's flux.

The grid is certified only for the cross sections it was thinned on. Two curves enter the solve
that it doesn't test: fission production, nu_t(E) sigma_f(E), and inelastic emission, the sum of
yield(E) sigma(E) over scattering reactions other than elastic. ZeroG takes each as a product at
the library's points, lin-lin between them, then lin-lin between grid nodes. OpenMC evaluates
the yield at E itself. The truth here does the same, on the union of the cross section's and the
yield's own points, each interval split in four.

Rates are weighted by the run's own flux. Each line gives the rate error on the solve's grid and,
for comparison, on the library's full grid.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "03-k-infinity" / "scripts"))
from cases import CASES  # noqa: E402

from zerog.data.library import nuclide_path  # noqa: E402
from zerog.data.neutron import (  # noqa: E402
    MT_FISSION,
    MT_PARTIAL_FISSION,
    _read_yield,
    read_reactions,
)
from zerog.data.scatter import read_scattering_laws  # noqa: E402

RESULTS = HERE / "results"
SPLIT = 4


def _yield_points(dataset):
    if dataset.attrs["type"].decode() == "Tabulated1D":
        return dataset[0, :]
    return np.empty(0)


def curves(name, label):
    """(name, energies, sigma, yield datasets) for fission production and inelastic emission."""
    data = read_reactions(name, label)
    out = []
    with h5py.File(nuclide_path(name), "r") as f:
        group = f[name]
        fission = [mt for mt in data.mt if mt in (MT_FISSION, *MT_PARTIAL_FISSION)]
        if fission:
            parts = []
            for mt in fission:
                rx = group["reactions"][f"reaction_{mt:03d}"]
                if "total_nu" in group:
                    nu = group["total_nu"]["yield"]
                else:
                    leading = next(
                        rx[k]
                        for k in rx
                        if k.startswith("product_")
                        and rx[k].attrs["particle"] == b"neutron"
                        and rx[k].attrs["emission_mode"] in (b"prompt", b"total")
                    )
                    nu = leading["yield"]
                parts.append((list(data.mt).index(mt), nu))
            out.append(("nu-fission", parts))
        laws = [r.mt for r in read_scattering_laws(name) if r.mt != 2]
        parts = []
        for mt in laws:
            rx = group["reactions"][f"reaction_{mt:03d}"]
            for k in rx:
                if k.startswith("product_") and rx[k].attrs["particle"] == b"neutron":
                    parts.append((list(data.mt).index(mt), rx[k]["yield"]))
        if parts:
            out.append(("emission", parts))
        results = []
        for kind, items in out:
            points = [data.energy] + [_yield_points(y) for _, y in items]
            truth_grid = np.unique(np.concatenate(points))[::-1]
            truth_grid = truth_grid[truth_grid <= data.energy[0]]
            u = np.log(truth_grid[0] / truth_grid)
            fine = np.unique(
                np.concatenate(
                    [u] + [u[:-1] + (u[1:] - u[:-1]) * s / SPLIT for s in range(1, SPLIT)]
                )
            )
            sample = truth_grid[0] * np.exp(-fine)
            truth = np.zeros(sample.shape[0])
            at_points = np.zeros(data.energy.shape[0])
            for r, y in items:
                sigma = np.interp(sample[::-1], data.energy[::-1], data.xs[r][::-1])[::-1]
                truth += sigma * _read_yield(y, sample)
                at_points += data.xs[r] * _read_yield(y, data.energy)
            results.append((kind, sample, truth, data.energy, at_points))
        return results


def lin(x, xp, fp):
    """Lin-lin on descending points."""
    return np.interp(x[::-1], xp[::-1], fp[::-1])[::-1]


def rate(sample, values, energy, flux):
    """The integral of values times F over lethargy, on the sample's points."""
    f = np.interp(np.log(energy[0] / sample), np.log(energy[0] / energy), flux, left=0, right=0)
    u = np.log(sample[0] / sample)
    g = values * f
    return np.sum(0.5 * (g[:-1] + g[1:]) * np.diff(u))


def main():
    name, kind, rtol, subdivide = sys.argv[1:5]
    run = np.load(RESULTS / f"{name}_{kind}_{rtol}_{subdivide}.npz")
    energy, flux = run["energy"], run["flux"]
    label = CASES[name]["label"]
    print(f"{name} {kind} {rtol} x{subdivide}: {energy.shape[0]} nodes")
    for nuclide in CASES[name]["densities"]:
        for what, sample, truth, data_energy, at_points in curves(nuclide, label):
            keep = (sample <= energy[0]) & (sample >= energy[-1])
            sample, truth = sample[keep], truth[keep]
            exact = rate(sample, truth, energy, flux)
            if exact <= 0.0:
                continue
            full = lin(sample, data_energy, at_points)
            on_grid = lin(sample, energy, lin(energy, data_energy, at_points))
            big = truth > 1e-3 * truth.max()
            worst = np.abs(on_grid[big] / truth[big] - 1.0).max()
            grid_error = rate(sample, on_grid, energy, flux) / exact - 1
            library_error = rate(sample, full, energy, flux) / exact - 1
            print(
                f"  {nuclide:6s} {what:10s} rate {grid_error:+.2e}"
                f" (library grid {library_error:+.2e}), worst point {worst:.1e}"
            )


if __name__ == "__main__":
    main()
