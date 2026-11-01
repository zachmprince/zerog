"""Union grid size against tolerance for a PWR pin and an SFR mix. Writes results/grid_sizes.csv."""

import csv
from pathlib import Path

import numpy as np
from openmc.data import NATURAL_ABUNDANCE

from zerog.data.grid import (
    major_curves,
    thin_indices,
    thin_nuclide,
    thin_reactions,
    union_grid,
)
from zerog.data.neutron import read_reactions
from zerog.data.temperature import read_reactions_at

RESULTS = Path(__file__).resolve().parent.parent / "results"
ENERGY_MAX = 2.0e7
ATOL_MICRO = 1e-8  # barns
ATOL_MACRO = 1e-8  # 1/cm
ATOL_SENSITIVITY = (1e-6, 1e-4)  # 1/cm
TOLERANCES = (1e-2, 1e-3, 1e-4)
AVOGADRO = 0.6022140


def element(symbol, density):
    """Natural isotopes of an element, in atoms/barn-cm."""
    return {
        name: density * fraction
        for name, fraction in NATURAL_ABUNDANCE.items()
        if name.rstrip("0123456789") == symbol
    }


def alloy(rho, weights):
    """Atom densities from a density in g/cm3 and {symbol: (weight fraction, atomic mass)}."""
    out = {}
    for symbol, (weight, mass) in weights.items():
        out |= element(symbol, rho * weight * AVOGADRO / mass)
    return out


def uo2(rho, enrichment):
    n = rho * AVOGADRO / 270.0
    trace = {"U234": 2.7e-4, "U236": 1.4e-4}
    uranium = {name: n * f for name, f in trace.items()}
    uranium |= {"U235": n * enrichment, "U238": n * (1.0 - enrichment - sum(trace.values()))}
    return uranium | element("O", 2.0 * n)


def water(rho, boron_ppm):
    n = rho * AVOGADRO / 18.015
    boron = rho * boron_ppm * 1e-6 * AVOGADRO / 10.81
    return element("H", 2.0 * n) | element("O", n) | element("B", boron)


def mox(rho, pu_fraction):
    n = rho * AVOGADRO / 271.0
    pu = {"Pu239": 0.6, "Pu240": 0.25, "Pu241": 0.1, "Pu242": 0.05}
    return (
        {name: n * pu_fraction * f for name, f in pu.items()}
        | {"U238": n * (1.0 - pu_fraction)}
        | element("O", 2.0 * n)
    )


ZIRCALOY = alloy(
    6.56,
    {"Zr": (0.982, 91.22), "Sn": (0.015, 118.71), "Fe": (0.002, 55.85), "Cr": (0.001, 52.0)},
)
STEEL = alloy(
    7.9,
    {
        "Fe": (0.87, 55.85),
        "Cr": (0.09, 52.0),
        "Mo": (0.01, 95.95),
        "Mn": (0.005, 54.94),
        "Ni": (0.005, 58.69),
    },
)
# Round-number materials: (material, temperature, densities).
CASES = {
    "PWR pin": [
        ("UO2", "900K", uo2(10.257, 0.031)),
        ("Zircaloy-4", "600K", ZIRCALOY),
        ("Borated water", "600K", water(0.743, 1300.0)),
    ],
    "SFR mix": [
        ("MOX", "1200K", mox(10.5, 0.2)),
        ("Steel", "600K", STEEL),
        ("Sodium", "600K", {"Na23": 0.022}),
    ],
}
# Tabulated temperatures a real pin's fuel, or its cladding and coolant, would span.
PWR_SPANS = {"UO2": ("600K", "900K", "1200K"), "Zircaloy-4": ("294K", "600K")}
PWR_SPANS["Borated water"] = ("294K", "600K")
# Kelvin, between tabulated temperatures, so each nuclide's grid merges two library grids.
INTERMEDIATE = {"UO2": 750.0, "Zircaloy-4": 565.0, "Borated water": 565.0}


def on_energy(data, curves, energy):
    return np.array([np.interp(energy[::-1], data.energy[::-1], c[::-1])[::-1] for c in curves])


def thin_checked(energy, curves, rtol, atol):
    """Thin, then check the tested curves at every point, independently of the kernel."""
    required = np.zeros(energy.shape[0], dtype=np.bool_)
    grid = energy[thin_indices(energy, curves, required, rtol, atol)]
    for curve in curves:
        values = np.interp(grid[::-1], energy[::-1], curve[::-1])
        rebuilt = np.interp(energy[::-1], grid[::-1], values)[::-1]
        if np.any(np.abs(rebuilt - curve) > rtol * np.abs(curve) + atol * (1.0 + 1e-9)):
            raise AssertionError(f"thinned grid breaks rtol={rtol}")
    return grid


def all_reactions_grid(name, temperature, rtol):
    grid = thin_nuclide(name, temperature, rtol, ATOL_MICRO, ENERGY_MAX)
    data = read_reactions(name, temperature)
    mask = data.energy <= ENERGY_MAX
    energy = data.energy[mask][::-1]
    for curve in data.xs[:, mask]:
        exact = curve[::-1]
        rebuilt = np.interp(energy, grid[::-1], np.interp(grid[::-1], energy, exact))
        if np.any(np.abs(rebuilt - exact) > rtol * np.abs(exact) + ATOL_MICRO * (1.0 + 1e-9)):
            raise AssertionError(f"{name} {temperature} breaks rtol={rtol}")
    return grid


def total_major_grid(name, temperature, rtol):
    data = read_reactions(name, temperature)
    mask = data.energy <= ENERGY_MAX
    return thin_checked(data.energy[mask], major_curves(data)[:, mask], rtol, ATOL_MICRO)


def material_data(temperature, densities):
    datasets = {name: read_reactions(name, temperature) for name in densities}
    energy = union_grid([d.energy[d.energy <= ENERGY_MAX] for d in datasets.values()])
    return datasets, energy


def joint_grid(temperature, densities, rtol):
    """One grid for a material's major microscopic curves, so nuclides share points."""
    datasets, energy = material_data(temperature, densities)
    curves = np.vstack([on_energy(d, major_curves(d), energy) for d in datasets.values()])
    return thin_checked(energy, curves, rtol, ATOL_MICRO)


def macroscopic_grid(temperature, densities, rtol, atol=ATOL_MACRO):
    datasets, energy = material_data(temperature, densities)
    curves = np.zeros((4, energy.shape[0]))
    for name, d in datasets.items():
        curves += densities[name] * on_energy(d, major_curves(d), energy)
    return thin_checked(energy, curves, rtol, atol)


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for case, materials in CASES.items():
        full = [
            (lambda e: e[e <= ENERGY_MAX])(read_reactions(n, temperature).energy)
            for _, temperature, densities in materials
            for n in densities
        ]
        n_full = len(union_grid(full))
        print(f"{case}: unthinned union {n_full}")
        rows.append((case, "all", "unthinned", 0.0, n_full))
        for rtol in TOLERANCES:
            methods = {"all reactions": [], "total+major": [], "joint": [], "macroscopic": []}
            for material, temperature, densities in materials:
                per_material = {
                    "all reactions": [all_reactions_grid(n, temperature, rtol) for n in densities],
                    "total+major": [total_major_grid(n, temperature, rtol) for n in densities],
                    "joint": [joint_grid(temperature, densities, rtol)],
                    "macroscopic": [macroscopic_grid(temperature, densities, rtol)],
                }
                for method, grids in per_material.items():
                    methods[method] += grids
                    rows.append((case, material, method, rtol, len(union_grid(grids))))
            for method, grids in methods.items():
                n = len(union_grid(grids))
                rows.append((case, "all", method, rtol, n))
                print(f"{case} rtol={rtol:g} {method}: {n}")

    for rtol in TOLERANCES[:2]:
        spanned = []
        for material, _, densities in CASES["PWR pin"]:
            grids = [
                all_reactions_grid(n, temperature, rtol)
                for temperature in PWR_SPANS[material]
                for n in densities
            ]
            spanned += grids
            n = len(union_grid(grids))
            rows.append(("PWR pin, temperature span", material, "all reactions", rtol, n))
        n = len(union_grid(spanned))
        rows.append(("PWR pin, temperature span", "all", "all reactions", rtol, n))
        print(f"PWR pin, temperature span rtol={rtol:g}: {n}")

    for rtol in TOLERANCES[:2]:
        grids = [
            thin_reactions(read_reactions_at(n, INTERMEDIATE[material]), rtol, ATOL_MICRO)
            for material, _, densities in CASES["PWR pin"]
            for n in densities
        ]
        n = len(union_grid(grids))
        rows.append(("PWR pin, intermediate temperatures", "all", "all reactions", rtol, n))
        print(f"PWR pin, intermediate temperatures rtol={rtol:g}: {n}")

    for case, materials in CASES.items():
        for atol in ATOL_SENSITIVITY:
            grids = [macroscopic_grid(t, d, 1e-3, atol) for _, t, d in materials]
            n = len(union_grid(grids))
            rows.append((case, "all", f"macroscopic, floor {atol:g}", 1e-3, n))
            print(f"{case} macroscopic floor {atol:g} at 0.1%: {n}")

    with open(RESULTS / "grid_sizes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "material", "method", "rtol", "n_energy"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
