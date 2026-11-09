"""OpenMC twin of study 03: k-infinity of a homogenized reactor mix.

Usage: build.py <case> <particles per batch> <active batches> [nocut] [ptables]. An infinite
medium (a reflective box) at one temperature, with probability tables off and neutrons killed below
the case's energy_min, if it has one, as ZeroG's grid ends there. With nocut the cutoff is left at
OpenMC's default, to measure what it costs; with ptables, unresolved-range probability tables are
on, for study 05. Results go to results/<label>/, gitignored apart from the small summary in
<label>.json, where label is the case with -nocut and -ptables appended as they apply.
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import openmc

HERE = Path(__file__).parent
STUDY = HERE.parents[1] / "studies" / "03-k-infinity" / "scripts"
sys.path.insert(0, str(STUDY))
from cases import CASES, FLUX_BINS_PER_LETHARGY, LIBRARY_BOTTOM  # noqa: E402

SEED = 20261107
INACTIVE = 20
ENERGY_MAX = 2.0e7  # eV, the lowest top of the library's grids for these nuclides


def build(name, particles, active, cut=True, ptables=False):
    case = CASES[name]
    material = openmc.Material()
    for nuclide, density in case["densities"].items():
        material.add_nuclide(nuclide, density)
    material.set_density("atom/b-cm", sum(case["densities"].values()))
    material.temperature = case["temperature"]
    for table in case["sab"].values():
        material.add_s_alpha_beta(table)

    box = openmc.model.RectangularParallelepiped(
        -50, 50, -50, 50, -50, 50, boundary_type="reflective"
    )
    cell = openmc.Cell(fill=material, region=-box)

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = INACTIVE + active
    settings.inactive = INACTIVE
    settings.seed = SEED
    settings.photon_transport = False
    settings.ptables = ptables
    settings.temperature = {"method": "nearest", "tolerance": 1.0}
    energy_min = case["energy_min"]
    if cut and energy_min is not None:
        settings.cutoff = {"energy_neutron": energy_min}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)), energy=openmc.stats.Watt()
    )

    bottom = LIBRARY_BOTTOM if energy_min is None else energy_min
    n_bins = int(np.log(ENERGY_MAX / bottom) * FLUX_BINS_PER_LETHARGY)
    edges = np.geomspace(bottom, ENERGY_MAX, n_bins + 1)
    flux = openmc.Tally(name="flux")
    flux.filters = [openmc.EnergyFilter(edges)]
    flux.scores = ["flux"]
    flux.estimator = "tracklength"
    rates = openmc.Tally(name="rates")
    rates.scores = ["(n,gamma)", "fission", "nu-fission", "scatter"]
    rates.nuclides = list(case["densities"])
    rates.estimator = "tracklength"

    return openmc.Model(
        openmc.Geometry([cell]),
        openmc.Materials([material]),
        settings,
        openmc.Tallies([flux, rates]),
    )


def main():
    name, particles, active = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cut = "nocut" not in sys.argv[4:]
    ptables = "ptables" in sys.argv[4:]
    label = name + ("" if cut else "-nocut") + ("-ptables" if ptables else "")
    work = HERE / "results" / label
    work.mkdir(parents=True, exist_ok=True)
    statepoint = build(name, particles, active, cut, ptables).run(cwd=work, output=False)
    with openmc.StatePoint(statepoint) as sp:
        flux = sp.get_tally(name="flux")
        rates = sp.get_tally(name="rates")
        edges = flux.find_filter(openmc.EnergyFilter).bins
        rate_df = rates.get_pandas_dataframe()
        summary = {
            "case": label,
            "provenance": {
                "inputs_sha256": hashlib.sha256(
                    (STUDY / "cases.py").read_bytes() + Path(__file__).read_bytes()
                ).hexdigest(),
                "openmc_version": openmc.__version__,
                "library": "ENDF/B-VIII.0 HDF5, endfb-viii.0-hdf5",
                "seed": SEED,
                "particles": particles,
                "batches": INACTIVE + active,
                "inactive": INACTIVE,
                "cutoff": CASES[name]["energy_min"] if cut else None,
                "ptables": ptables,
                "date": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            "k": sp.keff.nominal_value,
            "k_std": sp.keff.std_dev,
            # Tallies are per source neutron: OpenMC normalizes each batch to one starting neutron.
            "edges_low": edges[:, 0].tolist(),
            "edges_high": edges[:, 1].tolist(),
            "flux": flux.mean.ravel().tolist(),
            "flux_std": flux.std_dev.ravel().tolist(),
            "rates": [
                {k: row[k] for k in ("nuclide", "score", "mean", "std. dev.")}
                for row in rate_df.to_dict("records")
            ],
        }
    (HERE / f"{label}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
