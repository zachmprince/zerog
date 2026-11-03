"""OpenMC twin of study 02: slowing down from a point source below every inelastic threshold.

Usage: build.py <case> <particles per batch> <batches>. Results go to results/<case>/, gitignored
apart from the small summary in <case>.json.
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import openmc

HERE = Path(__file__).parent
STUDY = HERE.parents[1] / "studies" / "02-fixed-source" / "scripts"
sys.path.insert(0, str(STUDY))
from cases import CASES, ENERGY_MIN, FLUX_BINS_PER_LETHARGY  # noqa: E402

SEED = 20261101


def build(name, particles, batches):
    case = CASES[name]
    material = openmc.Material()
    for nuclide, density in case["densities"].items():
        material.add_nuclide(nuclide, density)
    material.set_density("atom/b-cm", sum(case["densities"].values()))
    material.temperature = 293.6

    box = openmc.model.RectangularParallelepiped(
        -50, 50, -50, 50, -50, 50, boundary_type="reflective"
    )
    cell = openmc.Cell(fill=material, region=-box)

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = particles
    settings.batches = batches
    settings.seed = SEED
    settings.photon_transport = False
    settings.create_fission_neutrons = False
    settings.ptables = False
    settings.temperature = {"method": "nearest", "tolerance": 1.0}
    settings.cutoff = {"energy_neutron": ENERGY_MIN}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([case["source_energy"]], [1.0]),
    )

    n_bins = int(np.log(case["source_energy"] / ENERGY_MIN) * FLUX_BINS_PER_LETHARGY)
    edges = np.geomspace(ENERGY_MIN, case["source_energy"] * (1.0 + 1e-9), n_bins + 1)
    flux = openmc.Tally(name="flux")
    flux.filters = [openmc.EnergyFilter(edges)]
    flux.scores = ["flux"]
    flux.estimator = "tracklength"
    rates = openmc.Tally(name="rates")
    rates.scores = ["(n,gamma)", "scatter"]
    rates.nuclides = list(case["densities"])
    rates.estimator = "tracklength"

    return openmc.Model(
        openmc.Geometry([cell]),
        openmc.Materials([material]),
        settings,
        openmc.Tallies([flux, rates]),
    )


def main():
    name, particles, batches = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    work = HERE / "results" / name
    work.mkdir(parents=True, exist_ok=True)
    statepoint = build(name, particles, batches).run(cwd=work, output=False)
    with openmc.StatePoint(statepoint) as sp:
        flux = sp.get_tally(name="flux")
        rates = sp.get_tally(name="rates")
        edges = flux.find_filter(openmc.EnergyFilter).bins
        rate_df = rates.get_pandas_dataframe()
        summary = {
            "case": name,
            "provenance": {
                "inputs_sha256": hashlib.sha256(
                    (STUDY / "cases.py").read_bytes() + Path(__file__).read_bytes()
                ).hexdigest(),
                "openmc_version": openmc.__version__,
                "library": "ENDF/B-VIII.0 HDF5, endfb-viii.0-hdf5",
                "seed": SEED,
                "particles": particles,
                "batches": batches,
                "date": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            "edges_low": edges[:, 0].tolist(),
            "edges_high": edges[:, 1].tolist(),
            "flux": flux.mean.ravel().tolist(),
            "flux_std": flux.std_dev.ravel().tolist(),
            "rates": [
                {k: row[k] for k in ("nuclide", "score", "mean", "std. dev.")}
                for row in rate_df.to_dict("records")
            ],
        }
    (HERE / f"{name}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
