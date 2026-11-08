"""Shared definitions for study 02, read by both ZeroG and its OpenMC twin."""

CASES = {
    "light": {
        "densities": {"Na23": 0.02, "Fe56": 0.02, "O16": 0.01},
        "source_energy": 4.0e5,
    },
    "heavy": {
        "densities": {"U238": 0.02, "O16": 0.04},
        "source_energy": 4.0e4,
    },
}
# Part 2: the same media from 14 MeV, so inelastic scattering and (n,xn) join elastic scattering.
CASES["light-fast"] = {"densities": CASES["light"]["densities"], "source_energy": 14.0e6}
CASES["heavy-fast"] = {"densities": CASES["heavy"]["densities"], "source_energy": 14.0e6}
TEMPERATURE = "294K"
ENERGY_MIN = 20.0  # eV, above OpenMC's free gas threshold of 400 kT, about 10 eV at 294 K
FLUX_BINS_PER_LETHARGY = 40
