"""Sanity checks for the slowing-down solver."""

import time
from pathlib import Path

import cells as C
import numpy as np
from slowdown import E_TOP, Nuclide, lethargy_grid, solve, trap_weights

dx, xc, tag = C.single_pin()
fuel = tag == "fuel"

# 1) No absorption, constant cross sections: exact answer is F = 1 everywhere.
Unr = Nuclide("U-238-like", 238.0, sig_s=11.1)
Hs = Nuclide("H-1", 1.0, sig_s=20.4, sig_a_2200=0.0)
nucs = [Hs, C.O, Unr]
N = C.densities(tag, nucs, {"fuel": {"U-238-like": C.N_U_FUEL}})
u = lethargy_grid([(C.U.res, 238.0, 900.0)])
T = np.full(N.shape, 900.0)
t0 = time.time()
F, info = solve(u, dx, nucs, N, T)
print(
    f"[asymptotic] NE={len(u)} Nx={len(dx)}  time={time.time() - t0:.1f}s  "
    f"max|F-1|={np.abs(F - 1).max():.2e}  max iters={info['iters'].max()}"
)

# 2) Real resonances, uniform 900 K.
nucs = C.NUCS_U
N = C.densities(tag, nucs, {"fuel": {"U-238-like": C.N_U_FUEL}})
T = np.full(N.shape, 900.0)
t0 = time.time()
F, info = solve(u, dx, nucs, N, T)
print(f"[resonant]   time={time.time() - t0:.1f}s  max iters={info['iters'].max()}")
E = E_TOP * np.exp(-u)
vol = dx / dx.sum()
print(f"  cell-avg F at 1 eV (≈ resonance escape p): {np.sum(F[-1] * vol):.4f}")
i0 = np.argmin(np.abs(E - 6.674))
print(
    f"  F at 6.674 eV: fuel center {F[i0, 0]:.4f}, fuel surface {F[i0, fuel][-1]:.4f}, "
    f"moderator {F[i0, ~fuel].mean():.4f}"
)
# capture rate profile in fuel
wu = trap_weights(u)
k = 2
Sc = N[k][None, :] * info["sig_a"][k, info["zone"][k]].T  # (NE, Nx)
R = (wu[:, None] * Sc * F).sum(0)
print(f"  capture rate surface/center ratio: {R[fuel][-1] / R[0]:.2f}")
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
np.savez(RESULTS / "verify.npz", u=u, F=F, xc=xc, tag=tag)
