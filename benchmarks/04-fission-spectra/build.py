"""OpenMC oracle for ZeroG's fission spectra: first-collision fission neutrons by outgoing energy.

A mono-energetic source in a small vacuum box of one nuclide, so that fission neutrons mostly leak
instead of multiplying without bound. An incoming-energy bin around the source energy holds only
first collisions, so the analog nu-fission tally by outgoing energy is OpenMC's own sampling of
prompt and delayed fission neutrons at that energy, resampled below the library's top energy as
OpenMC does. The analog total score in the same bin counts those collisions. Usage: build.py
[case ...].
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import openmc

HERE = Path(__file__).parent
SEED = 20261107
PARTICLES = 2_000_000
BATCHES = 10
BINS_PER_LETHARGY = 20
ENERGY_MIN = 1.0e3  # eV, the bottom of the outgoing tally; a fission spectrum holds ~1e-5 below

# Incoming energies sit between chi tables. U238's tables change their top outgoing energy between
# 4 and 5 MeV and between 7 and 8 MeV, so unit-base scaling acts there. Pu241's prompt spectrum is
# a Maxwell.
CASES = {
    "U235-1.2MeV": ("U235", 1.2e6),
    "U235-14.3MeV": ("U235", 14.3e6),
    "U238-2.2MeV": ("U238", 2.2e6),
    "U238-4.5MeV": ("U238", 4.5e6),
    "U238-7.5MeV": ("U238", 7.5e6),
    "Pu239-1.2MeV": ("Pu239", 1.2e6),
    "Pu240-2.7MeV": ("Pu240", 2.7e6),
    "Pu241-1.2MeV": ("Pu241", 1.2e6),
    "Pu242-3.3MeV": ("Pu242", 3.3e6),
    # U236 fissions through partial reactions (MT 19, 20, 21, 38), whose later chances have Maxwell
    # spectra cut at E - u with u of 2.6, 10.4 and 15.9 MeV (M4 review).
    "U236-8MeV": ("U236", 8.0e6),
    "U236-12.5MeV": ("U236", 12.5e6),
}


def run(case):
    nuclide, e0 = CASES[case]
    material = openmc.Material()
    material.add_nuclide(nuclide, 1.0)
    material.set_density("atom/b-cm", 0.05)
    material.temperature = 293.6
    box = openmc.model.RectangularParallelepiped(-2, 2, -2, 2, -2, 2, boundary_type="vacuum")
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = PARTICLES
    settings.batches = BATCHES
    settings.seed = SEED
    settings.photon_transport = False
    settings.create_fission_neutrons = True
    settings.ptables = False
    settings.temperature = {"method": "nearest", "tolerance": 1.0}
    # Only first collisions count, so stop each history once it has left the source energy.
    settings.cutoff = {"energy_neutron": e0 * 0.999}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(), energy=openmc.stats.Discrete([e0], [1.0])
    )
    edges = np.geomspace(ENERGY_MIN, 3.2e7, int(np.log(3.2e7 / ENERGY_MIN) * BINS_PER_LETHARGY) + 1)
    tally = openmc.Tally(name="out")
    tally.filters = [
        openmc.EnergyFilter([e0 * (1 - 1e-9), e0 * (1 + 1e-9)]),
        openmc.EnergyoutFilter(edges),
    ]
    tally.scores = ["nu-fission"]
    tally.estimator = "analog"
    collisions = openmc.Tally(name="collisions")
    collisions.filters = [openmc.EnergyFilter([e0 * (1 - 1e-9), e0 * (1 + 1e-9)])]
    collisions.scores = ["total"]
    collisions.estimator = "analog"
    work = HERE / "results" / case
    work.mkdir(parents=True, exist_ok=True)
    model = openmc.Model(
        openmc.Geometry([openmc.Cell(fill=material, region=-box)]),
        openmc.Materials([material]),
        settings,
        openmc.Tallies([tally, collisions]),
    )
    statepoint = model.run(cwd=work, output=False)
    with openmc.StatePoint(statepoint) as sp:
        out = sp.get_tally(name="out")
        mean = out.mean.ravel()
        std = out.std_dev.ravel()
        hits = sp.get_tally(name="collisions")
        collided = float(hits.mean.ravel()[0])
        collided_std = float(hits.std_dev.ravel()[0])
    summary = {
        "case": case,
        "nuclide": nuclide,
        "source_energy": e0,
        "provenance": {
            "inputs_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "openmc_version": openmc.__version__,
            "library": "ENDF/B-VIII.0 HDF5, endfb-viii.0-hdf5",
            "seed": SEED,
            "particles": PARTICLES,
            "batches": BATCHES,
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "edges": edges.tolist(),
        # First collisions per source neutron, and fission neutrons per source neutron per outgoing
        # bin, which sum to collisions * nu * sigma_f / sigma_t at the source energy.
        "collisions": collided,
        "collisions_std": collided_std,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    (HERE / f"{case}.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    for case in sys.argv[1:] or CASES:
        run(case)
