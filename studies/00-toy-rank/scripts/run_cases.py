"""Run the toy cases and measure the space-energy rank of the reference flux.

For each case:
  * weighted SVD of F(x,u) (inner product = integral over x and lethargy);
  * rank-M truncation errors: L2, worst per-cell capture rate (any absorber,
    any fuel cell), and total capture;
  * the same with energy split into macro-bands, each band with its own
    separated expansion; modes are pooled greedily by singular value, so
    "M" is the total number of space-energy mode pairs across all bands.
"""

import time
from pathlib import Path

import cells as C
import numpy as np
from slowdown import E_TOP, lethargy_grid, solve, trap_weights

OUT = Path(__file__).resolve().parents[1] / "results" / "results.npz"


def case_definitions():
    cases = {}

    dx, xc, tag = C.single_pin()
    R = C.FUEL_T / 2
    fuel = tag == "fuel"

    # A: single pin, uniform 900 K
    N = C.densities(tag, C.NUCS_U, {"fuel": {"U-238-like": C.N_U_FUEL}})
    T = np.full(N.shape, 900.0)
    cases["A"] = dict(
        label="A: one pin, 900 K",
        dx=dx,
        xc=xc,
        tag=tag,
        nucs=C.NUCS_U,
        N=N,
        T=T,
        res=[(C.U.res, 238.0, 900.0)],
    )

    # B: single pin, parabolic fuel temperature 1200 K (center) -> 600 K (surface),
    #    binned to 50 K zones (13 temperature-dependent cross-section sets)
    Tf = 600.0 + 600.0 * (1.0 - (xc / R) ** 2)
    Tf = np.round(Tf / 50.0) * 50.0
    T = np.full(N.shape, 900.0)
    T[2, fuel] = Tf[fuel]
    cases["B"] = dict(
        label="B: one pin, T profile",
        dx=dx,
        xc=xc,
        tag=tag,
        nucs=C.NUCS_U,
        N=N.copy(),
        T=T,
        res=[(C.U.res, 238.0, 600.0)],
    )

    # C: single pin, uniform 900 K, plus a second absorber concentrated at the rim
    Nx2 = 2e-4 + 6e-3 * np.exp(-(R - xc) / 0.012)
    N = C.densities(tag, C.NUCS_UX, {"fuel": {"U-238-like": C.N_U_FUEL, "absorber-2": Nx2[fuel]}})
    T = np.full(N.shape, 900.0)
    cases["C"] = dict(
        label="C: one pin + rim absorber",
        dx=dx,
        xc=xc,
        tag=tag,
        nucs=C.NUCS_UX,
        N=N,
        T=T,
        res=[(C.U.res, 238.0, 900.0), (C.X.res, 240.0, 900.0)],
    )

    # D: two different pins in one cell: A = 900 K U only; B = 600 K U + absorber-2
    dx, xc, tag = C.two_pin()
    tag_fuel = np.where(np.isin(tag, ["fuelA", "fuelB"]), tag, "mod")
    N = C.densities(
        tag_fuel,
        C.NUCS_UX,
        {
            "fuelA": {"U-238-like": C.N_U_FUEL},
            "fuelB": {"U-238-like": C.N_U_FUEL, "absorber-2": 1.5e-3},
        },
    )
    T = np.full(N.shape, 900.0)
    T[:, tag == "fuelB"] = 600.0
    cases["D"] = dict(
        label="D: two different pins",
        dx=dx,
        xc=xc,
        tag=tag_fuel,
        nucs=C.NUCS_UX,
        N=N,
        T=T,
        res=[(C.U.res, 238.0, 600.0), (C.X.res, 240.0, 600.0)],
    )
    return cases


def weighted_svd(F, wu, dx):
    B = np.sqrt(wu)[:, None] * F * np.sqrt(dx)[None, :]
    return np.linalg.svd(B, full_matrices=False)


def band_slices(u, edges_eV):
    E = E_TOP * np.exp(-u)
    out = []
    for hi, lo in zip(edges_eV[:-1], edges_eV[1:]):
        m = (hi >= E) & (lo < E) if lo > edges_eV[-1] else (hi >= E) & (lo <= E)
        out.append(np.where(m)[0])
    return out


def truncation_path(F, wu, dx, Sc, cellmask, bands):
    """Greedy pooled truncation across energy bands.
    Sc: list of (NE, Nx) macroscopic capture xs per absorber.
    cellmask: list of (Nx,) bool, where each absorber is present."""
    Ref = [(wu[:, None] * s * F).sum(0) for s in Sc]
    Svals, contrib = [], []
    for idx in bands:
        U, S, Vt = weighted_svd(F[idx], wu[idx], dx)
        sw = np.sqrt(wu[idx])[:, None]
        per = [(S[:, None] * Vt / np.sqrt(dx)[None, :]) * (U.T @ (sw * s[idx])) for s in Sc]
        Svals.append(S)
        contrib.append(np.stack(per, axis=1))  # (modes, n_abs, Nx)
    S_all = np.concatenate(Svals)
    C_all = np.concatenate(contrib, axis=0)
    order = np.argsort(-S_all)
    S_ord, C_ord = S_all[order], C_all[order]
    cum = np.cumsum(C_ord, axis=0)  # (M, n_abs, Nx)
    l2 = np.sqrt(np.clip(1.0 - np.cumsum(S_ord**2) / np.sum(S_ord**2), 0, None))
    cell_err = np.zeros(len(S_ord))
    tot_err = np.zeros(len(S_ord))
    for a, (r, m) in enumerate(zip(Ref, cellmask)):
        e = np.abs(cum[:, a, m] - r[m]) / r[m]
        cell_err = np.maximum(cell_err, e.max(axis=1))
        tot = (cum[:, a, :] * dx).sum(1)
        tot_ref = (r * dx).sum()
        tot_err = np.maximum(tot_err, np.abs(tot - tot_ref) / tot_ref)
    return dict(S=S_ord, l2=l2, cell=cell_err, tot=tot_err)


def angular_path(Psi, F, wu, dx, wmu, Sc, cellmask, Mmax=300, chunk=3000):
    """Rank-M truncation of the angular flux psi(x, mu, u) with (x, mu) modes
    and energy modes, i.e. the separation an energy-PGD would actually use.
    wmu: quadrature weights matching Psi's angle rows (sum to 2)."""
    NE, Na, Nx = Psi.shape
    P = Na * Nx
    sp = np.sqrt((wmu[:, None] * dx[None, :]).ravel())
    sw = np.sqrt(wu)

    def block(a, b):
        return Psi[a:b].reshape(b - a, P).astype(np.float64) * sw[a:b, None] * sp[None, :]

    Gm = np.zeros((P, P))
    for a in range(0, NE, chunk):
        B = block(a, min(a + chunk, NE))
        Gm += B.T @ B
    lam, V = np.linalg.eigh(Gm)
    lam, V = np.clip(lam[::-1], 0, None), V[:, ::-1]
    S = np.sqrt(lam)
    Mmax = min(Mmax, P)
    Vm = V[:, :Mmax]
    smode = np.einsum("a,aim->mi", wmu, (Vm / sp[:, None]).reshape(Na, Nx, Mmax))
    Cs = [np.zeros((Mmax, Nx)) for _ in Sc]
    for a in range(0, NE, chunk):
        b = min(a + chunk, NE)
        A = block(a, b) @ Vm
        for c, s in zip(Cs, Sc):
            c += A.T @ (sw[a:b, None] * s[a:b])
    Ref = [(wu[:, None] * s * F).sum(0) for s in Sc]
    l2 = np.sqrt(np.clip(1.0 - np.cumsum(lam) / lam.sum(), 0, None))[:Mmax]
    cell_err = np.zeros(Mmax)
    tot_err = np.zeros(Mmax)
    for c, r, m in zip(Cs, Ref, cellmask):
        cum = np.cumsum(smode * c, axis=0)
        cell_err = np.maximum(cell_err, (np.abs(cum[:, m] - r[m]) / r[m]).max(1))
        tot_err = np.maximum(tot_err, np.abs((cum * dx).sum(1) - (r * dx).sum()) / (r * dx).sum())
    return dict(S=S, l2=l2, cell=cell_err, tot=tot_err)


def converged_rank(err, tol):
    """Smallest M such that err stays below tol for every M' >= M."""
    above = np.where(err >= tol)[0]
    return 1 if len(above) == 0 else above[-1] + 2


BANDS = {
    "1 band": [5000.0, 1.0],
    "4 bands": [5000.0, 1000.0, 100.0, 10.0, 1.0],
    "8 bands": [5000.0, 3162.0, 1000.0, 316.2, 100.0, 31.62, 10.0, 3.162, 1.0],
}


def main():
    cases = case_definitions()
    store = {}
    print(
        f"{'case':28s} {'NE':>6s} {'Nx':>4s} {'time':>6s} | "
        f"{'bands':8s} {'L2<1e-3':>8s} {'cell<1%':>8s} {'cell<0.1%':>9s} {'tot<0.1%':>9s}"
    )
    for key, cs in cases.items():
        u = lethargy_grid(cs["res"])
        t0 = time.time()
        F, info = solve(u, cs["dx"], cs["nucs"], cs["N"], cs["T"], angular=True)
        dt = time.time() - t0
        Psi = info.pop("Psi")
        wmu = np.concatenate([info["wt"], info["wt"]])
        chk = np.abs((Psi * wmu[None, :, None]).sum(1) - F).max()
        assert chk < 1e-5, chk
        wu = trap_weights(u)
        Sc, masks = [], []
        for k, nuc in enumerate(cs["nucs"]):
            if nuc.res is None:
                continue
            micro = info["sig_a"][k][info["zone"][k]]  # (Nx, NE)
            Sc.append((cs["N"][k][:, None] * micro).T)
            masks.append(cs["N"][k] > 0)
        for bname, edges in BANDS.items():
            path = truncation_path(F, wu, cs["dx"], Sc, masks, band_slices(u, edges))
            r = [
                converged_rank(path["l2"], 1e-3),
                converged_rank(path["cell"], 1e-2),
                converged_rank(path["cell"], 1e-3),
                converged_rank(path["tot"], 1e-3),
            ]
            print(
                f"{cs['label']:28s} {len(u):6d} {len(cs['dx']):4d} {dt:5.1f}s | "
                f"{bname:8s} {r[0]:8d} {r[1]:8d} {r[2]:9d} {r[3]:9d}"
            )
            for m, v in path.items():
                store[f"{key}|{bname}|{m}"] = v
        path = angular_path(Psi, F, wu, cs["dx"], wmu, Sc, masks)
        r = [
            converged_rank(path["l2"], 1e-3),
            converged_rank(path["cell"], 1e-2),
            converged_rank(path["cell"], 1e-3),
            converged_rank(path["tot"], 1e-3),
        ]
        print(
            f"{cs['label']:28s} {'':6s} {'':4s} {'':6s} | "
            f"{'angular':8s} {r[0]:8d} {r[1]:8d} {r[2]:9d} {r[3]:9d}"
        )
        for m, v in path.items():
            store[f"{key}|angular|{m}"] = v
        del Psi
        # modes and flux for plotting
        U, S, Vt = weighted_svd(F, wu, cs["dx"])
        store[f"{key}|u"] = u
        store[f"{key}|xc"] = cs["xc"]
        store[f"{key}|dx"] = cs["dx"]
        store[f"{key}|tag"] = cs["tag"]
        store[f"{key}|label"] = cs["label"]
        if key == "A":
            store["A|F"] = F
            store["A|Emodes"] = U[:, :6] / np.sqrt(wu)[:, None]
            store["A|Xmodes"] = Vt[:6].T / np.sqrt(cs["dx"])[:, None]
            store["A|S"] = S
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **store)
    print("saved", OUT)


if __name__ == "__main__":
    main()
