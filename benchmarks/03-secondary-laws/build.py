"""OpenMC oracle for ZeroG's inelastic kernels: first-collision outgoing spectra per reaction.

A mono-energetic source in an infinite medium of one nuclide. An incoming-energy bin around the
source energy holds only first collisions, since no neutron gains energy here, so each reaction's
energy-out tally is OpenMC's own sampling of that law at that energy. Usage: build.py [case ...].
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import openmc

from zerog.data.scatter import read_scattering_laws

HERE = Path(__file__).parent
SEED = 20261102
PARTICLES = 2_000_000
BATCHES = 10
BINS_PER_LETHARGY = 50
ENERGY_MIN = 100.0  # eV, the bottom of the energy-out tally

# Between them these cover every neutron law in the 20 PWR and SFR nuclides surveyed for M3:
# centre-of-mass levels, Kalbach-Mann and correlated laws, and lab evaporation, tabular and
# correlated laws, with a reaction that holds two laws (Na23 MT 91).
CASES = {
    "O16-14MeV": ("O16", 14.0e6),
    "Fe56-2MeV": ("Fe56", 2.0e6),
    "Fe56-14MeV": ("Fe56", 14.0e6),
    "Na23-14MeV": ("Na23", 14.0e6),
    "Cr52-14MeV": ("Cr52", 14.0e6),
    "Pu241-14MeV": ("Pu241", 14.0e6),
    "U238-1MeV": ("U238", 1.0e6),
    "U238-14MeV": ("U238", 14.0e6),
    # 14 MeV is a table energy for U238 and Fe56, so these test mixing and unit-base scaling between
    # tables, where study 02's second collisions happen (M3 review).
    "U238-13.7MeV": ("U238", 13.7e6),
    "U238-13.25MeV": ("U238", 13.25e6),
    "Fe56-13.7MeV": ("Fe56", 13.7e6),
    "O16-13.3MeV": ("O16", 13.3e6),
}


def run(case):
    nuclide, e0 = CASES[case]
    mts = [r.mt for r in read_scattering_laws(nuclide) if r.mt != 2]
    material = openmc.Material()
    material.add_nuclide(nuclide, 1.0)
    material.set_density("atom/b-cm", 0.05)
    material.temperature = 293.6
    box = openmc.model.RectangularParallelepiped(
        -50, 50, -50, 50, -50, 50, boundary_type="reflective"
    )
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = PARTICLES
    settings.batches = BATCHES
    settings.seed = SEED
    settings.photon_transport = False
    settings.create_fission_neutrons = False
    settings.ptables = False
    settings.temperature = {"method": "nearest", "tolerance": 1.0}
    # Later collisions don't count, so stop each history once it has slowed well below the source.
    settings.cutoff = {"energy_neutron": ENERGY_MIN * 0.5}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(), energy=openmc.stats.Discrete([e0], [1.0])
    )
    edges = np.geomspace(
        ENERGY_MIN, e0 * 1.0001, int(np.log(e0 / ENERGY_MIN) * BINS_PER_LETHARGY) + 1
    )
    tally = openmc.Tally(name="out")
    tally.filters = [
        openmc.EnergyFilter([e0 * (1 - 1e-9), e0 * (1 + 1e-9)]),
        openmc.EnergyoutFilter(edges),
    ]
    tally.scores = [str(mt) for mt in mts]
    tally.estimator = "analog"
    work = HERE / "results" / case
    work.mkdir(parents=True, exist_ok=True)
    model = openmc.Model(
        openmc.Geometry([openmc.Cell(fill=material, region=-box)]),
        openmc.Materials([material]),
        settings,
        openmc.Tallies([tally]),
    )
    statepoint = model.run(cwd=work, output=False)
    with openmc.StatePoint(statepoint) as sp:
        out = sp.get_tally(name="out")
        mean = out.mean.reshape(len(edges) - 1, -1)
        std = out.std_dev.reshape(len(edges) - 1, -1)
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
        "mts": mts,
        # Events per source neutron, per outgoing bin, for each MT. Each source neutron has one
        # first collision, so a reaction's events sum to its share of the total cross section.
        "mean": {str(mt): mean[:, i].tolist() for i, mt in enumerate(mts)},
        "std": {str(mt): std[:, i].tolist() for i, mt in enumerate(mts)},
    }
    (HERE / f"{case}.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    for case in sys.argv[1:] or CASES:
        run(case)
