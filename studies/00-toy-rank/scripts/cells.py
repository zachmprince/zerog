"""Lattice cells and material definitions for the toy cases."""

import numpy as np
from slowdown import Nuclide, absorber2_like, geom_cells, u238_like

# Number densities [1/b-cm]
N_U_FUEL, N_O_FUEL = 0.0232, 0.0464  # UO2, 10.4 g/cc (all U treated as U-238-like)
N_H_MOD, N_O_MOD = 0.0495, 0.02476  # H2O, 0.74 g/cc

# Slab dimensions chosen to preserve mean chord lengths of a 0.41 cm radius pin
# at 1.26 cm pitch: fuel slab 0.41 cm thick, moderator slab 0.82 cm thick.
FUEL_T, MOD_T = 0.41, 0.82

H = Nuclide("H-1", 1.0, sig_s=20.4, sig_a_2200=0.332)
O = Nuclide("O-16", 16.0, sig_s=3.8)
U = Nuclide("U-238-like", 238.0, res=u238_like(), R_fm=9.40)
X = Nuclide("absorber-2", 240.0, res=absorber2_like(), R_fm=9.50)

NUCS_U = [H, O, U]
NUCS_UX = [H, O, U, X]


def mesh(regions):
    """regions: list of (material_tag, thickness, n_cells, dx_min, toward)."""
    dx, tag = [], []
    for name, L, n, dmin, toward in regions:
        w = geom_cells(L, n, dmin, toward)
        dx.append(w)
        tag += [name] * len(w)
    dx = np.concatenate(dx)
    xc = np.cumsum(dx) - 0.5 * dx
    return dx, xc, np.array(tag)


def single_pin():
    # half fuel (centerline at x=0) | half moderator, reflective both ends
    return mesh([("fuel", FUEL_T / 2, 70, 4e-4, "right"), ("mod", MOD_T / 2, 30, 2e-3, "left")])


def two_pin():
    # half pin A | moderator | full pin B | half moderator  (reflective)
    return mesh(
        [
            ("fuelA", FUEL_T / 2, 50, 5e-4, "right"),
            ("mod", MOD_T, 30, 2e-3, "both"),
            ("fuelB", FUEL_T, 80, 5e-4, "both"),
            ("mod", MOD_T / 2, 20, 2e-3, "left"),
        ]
    )


def densities(tag, nucs, fuel_spec):
    """fuel_spec: dict fuel_tag -> dict nuclide name -> density array or scalar."""
    K, Nx = len(nucs), len(tag)
    N = np.zeros((K, Nx))
    for k, nuc in enumerate(nucs):
        if nuc.name == "H-1":
            N[k, tag == "mod"] = N_H_MOD
        elif nuc.name == "O-16":
            N[k, tag == "mod"] = N_O_MOD
            N[k, tag != "mod"] = N_O_FUEL
    for ft, spec in fuel_spec.items():
        m = tag == ft
        for k, nuc in enumerate(nucs):
            if nuc.name in spec:
                N[k, m] = spec[nuc.name]
    return N
