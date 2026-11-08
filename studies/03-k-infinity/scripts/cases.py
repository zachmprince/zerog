"""Shared definitions for study 03, read by both ZeroG and its OpenMC twin."""

# A homogenized sodium-cooled fast reactor cell: 35% oxide MOX (U0.8 Pu0.2)O2 at 10.4 g/cm3, 40%
# sodium at 0.85 g/cm3 and 25% steel (65% Fe, 17% Cr, 12% Ni by mass) at 7.9 g/cm3. The Pu vector
# is 55/25/10/6 Pu239/240/241/242 with 2% Am241 and 2% Pu238 folded into Pu239, U is depleted to
# 0.25% U235, and each steel element is its main isotope. Atoms/b-cm.
SFR = {
    "U235": 1.618e-5,
    "U238": 6.455e-3,
    "Pu239": 9.222e-4,
    "Pu240": 4.045e-4,
    "Pu241": 1.618e-4,
    "Pu242": 9.707e-5,
    "Am241": 3.236e-5,
    "O16": 1.6178e-2,
    "Na23": 8.907e-3,
    "Fe56": 1.3842e-2,
    "Cr52": 3.888e-3,
    "Ni58": 2.432e-3,
}
CASES = {"sfr": {"densities": SFR, "temperature": 900.0}}
TEMPERATURE = "900K"
ENERGY_MIN = 50.0  # eV, above OpenMC's free gas threshold of 400 kT, 31 eV at 900 K
FLUX_BINS_PER_LETHARGY = 20
