"""Order of the no-absorption error (exact F = 1) on uniform lethargy grids."""

import cells as C
import numpy as np
from slowdown import E_BOT, E_TOP, Nuclide, solve

dx, xc, tag = C.single_pin()
nucs = [Nuclide("H-1", 1.0, sig_s=20.4), C.O, Nuclide("U-238-like", 238.0, sig_s=11.1)]
N = C.densities(tag, nucs, {"fuel": {"U-238-like": C.N_U_FUEL}})
T = np.full(N.shape, 900.0)
previous = None
for du in (0.004, 0.002, 0.001):
    u = np.arange(0.0, np.log(E_TOP / E_BOT) + du, du)
    F, _ = solve(u, dx, nucs, N, T)
    err = np.abs(F - 1).max()
    ratio = previous / err if previous else float("nan")
    print(f"du={du} NE={len(u)} max|F-1|={err:.3e} ratio={ratio:.2f}")
    previous = err
