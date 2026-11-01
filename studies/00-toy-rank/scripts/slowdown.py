"""
Pointwise-energy slowing-down solver for a 1D slab lattice cell.

Purpose: generate a reference space-energy flux F(x, u) through the resolved
resonance range so its rank (singular-value decay) can be measured. This is the
best case for ANY rank-M separated representation, PGD included, in the
weighted L2 norm used here.

Model
-----
* Lethargy u = ln(E_top / E), E from E_top (5 keV) down to E_bot (1 eV).
  F = E * phi is flux per unit lethargy.
* Neutrons enter from above: flux above E_top is asymptotic (F = 1) everywhere.
* Elastic scattering, target-at-rest kinematics, isotropic in the lab (toy).
* Each energy point is a one-group fixed-source S_N problem: Gauss-Legendre
  quadrature, step characteristics (flat source, exact attenuation), and an
  exact closure for the reflective/reflective boundary pair.
* The slowing-down integral is marched with trapezoid quadrature in lethargy;
  the current point's own (tiny) self-scatter is iterated inside the S_N solve.
* Resonances: single-level Breit-Wigner, Doppler broadened with the Faddeeva
  function (psi/chi line shapes), evaluated per temperature zone.
"""

import math

import numpy as np
from numba import njit
from scipy.optimize import brentq
from scipy.special import wofz

K_B = 8.617333262e-5  # eV/K
E_TOP, E_BOT = 5000.0, 1.0  # eV


# --------------------------------------------------------------------------
# Resonance data (synthetic, "U-238-like")
# --------------------------------------------------------------------------
# Lowest s-wave resonances with approximately U-238-like parameters
# (E0 [eV], Gamma_n [eV], Gamma_gamma [eV]). Approximate on purpose: this is a
# toy, not an evaluation.
U238_LOW = np.array(
    [
        (6.674, 1.493e-3, 23.0e-3),
        (20.872, 10.26e-3, 22.9e-3),
        (36.682, 33.55e-3, 22.9e-3),
        (66.032, 24.18e-3, 23.3e-3),
        (80.749, 1.87e-3, 23.0e-3),
        (102.56, 70.8e-3, 24.6e-3),
        (116.90, 25.5e-3, 22.4e-3),
        (165.29, 3.30e-3, 23.0e-3),
        (189.67, 171.0e-3, 22.0e-3),
        (208.51, 51.6e-3, 23.0e-3),
        (237.38, 26.6e-3, 23.0e-3),
        (273.66, 25.2e-3, 23.0e-3),
        (291.00, 16.4e-3, 23.0e-3),
    ]
)


def sample_ladder(E_start, E_stop, D, Gn0_mean, Gg, seed):
    """Statistical s-wave ladder: Wigner spacings, Porter-Thomas widths.
    Gamma_n = Gamma_n^0 * sqrt(E)."""
    rng = np.random.default_rng(seed)
    res, E = [], E_start
    while True:
        E += D * math.sqrt(-4.0 / math.pi * math.log(rng.uniform(1e-12, 1.0)))
        if E_stop < E:
            break
        Gn0 = Gn0_mean * rng.standard_normal() ** 2
        res.append((E, max(Gn0, 1e-3 * Gn0_mean) * math.sqrt(E), Gg))
    return np.array(res)


def u238_like():
    hi = sample_ladder(291.0, 1.05 * E_TOP, D=20.9, Gn0_mean=2.1e-3, Gg=23e-3, seed=238)
    return np.vstack([U238_LOW, hi])


def absorber2_like():
    """Second resonant absorber, mass 240, denser and wider ladder
    (a stand-in for a Pu isotope; fission lumped into absorption)."""
    return sample_ladder(7.0, 1.05 * E_TOP, D=12.0, Gn0_mean=1.6e-3, Gg=30e-3, seed=240)


def slbw(E, res, A, T, R_fm, g=1.0):
    """Doppler-broadened SLBW capture and elastic cross sections [barns]."""
    E = np.asarray(E, dtype=float)
    R = R_fm * 1e-13
    sc = np.zeros_like(E)
    se = np.full_like(E, 4.0 * math.pi * R * R * 1e24)
    for E0, Gn, Gg in res:
        G = Gn + Gg
        lam0 = 4.5525e-10 * (A + 1.0) / A / math.sqrt(E0)  # reduced wavelength (CM), cm
        sig0 = 4.0 * math.pi * lam0 * lam0 * g * Gn / G * 1e24  # peak, barns
        x = 2.0 * (E - E0) / G
        Delta = math.sqrt(4.0 * E0 * K_B * T / A)
        xi = G / Delta
        w = wofz(0.5 * xi * (x + 1j))
        psi = 0.5 * xi * math.sqrt(math.pi) * w.real
        chi = 0.5 * xi * math.sqrt(math.pi) * w.imag
        sc += sig0 * Gg / G * np.sqrt(E0 / E) * psi
        se += sig0 * Gn / G * psi + sig0 * (2.0 * R / lam0) * chi
    return sc, np.maximum(se, 0.5)  # floor guards SLBW interference dips


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------
def lethargy_grid(res_sets, du_max=0.004):
    """Union grid in lethargy: uniform base plus refinement around each
    resonance. res_sets: list of (res, A, T_min)."""
    E = [np.exp(-np.arange(0.0, math.log(E_TOP / E_BOT) + du_max, du_max)) * E_TOP]
    for res, A, Tmin in res_sets:
        for E0, Gn, Gg in res:
            if E0 > E_TOP:
                continue
            G = Gn + Gg
            hw = math.sqrt((0.5 * G) ** 2 + 4.0 * E0 * K_B * Tmin / A)
            core = E0 + hw * np.linspace(-6, 6, 49)
            wing = hw * np.geomspace(6, 400, 45)
            E.append(np.concatenate([core, E0 - wing, E0 + wing]))
    E = np.concatenate(E)
    E = E[(E >= E_BOT) & (E <= E_TOP)]
    u = np.unique(np.round(np.log(E_TOP / E), 9))
    keep = np.concatenate([[True], np.diff(u) > 2e-7])
    return u[keep]


def geom_cells(L, n, dx_min, toward):
    """n cell widths summing to L, geometric, smallest = dx_min at 'toward'
    end ('left', 'right' or 'both')."""
    if toward == "both":
        a = geom_cells(L / 2, n // 2, dx_min, "left")
        return np.concatenate([a[::-1], a])

    def total(r):
        return dx_min * (r**n - 1) / (r - 1) - L if abs(r - 1) > 1e-12 else dx_min * n - L

    r = brentq(total, 1.0 + 1e-9, 2.0)
    w = dx_min * r ** np.arange(n)
    return w if toward == "left" else w[::-1]


# --------------------------------------------------------------------------
# Transport kernels
# --------------------------------------------------------------------------
@njit(cache=True)
def sweep_reflective(St, Q, dx, mu, wt):
    """One-group slab S_N, step characteristic, reflective at both ends.
    Q is the isotropic angular source density (q/2). Returns scalar flux."""
    Nx = St.shape[0]
    phi = np.zeros(Nx)
    for m in range(mu.shape[0]):
        mm = mu[m]
        # zero-incoming passes give the boundary closure a = (A*B + C)/(1 - A^2)
        psi = 0.0
        tau = 0.0
        for i in range(Nx):
            t = St[i] * dx[i] / mm
            omt = -math.expm1(-t)
            psi = psi * (1.0 - omt) + Q[i] / St[i] * omt
            tau += t
        A = math.exp(-tau)
        B = psi
        psi = 0.0
        for i in range(Nx - 1, -1, -1):
            t = St[i] * dx[i] / mm
            omt = -math.expm1(-t)
            psi = psi * (1.0 - omt) + Q[i] / St[i] * omt
        C = psi
        psi = (A * B + C) / (1.0 - A * A)
        for i in range(Nx):
            t = St[i] * dx[i] / mm
            omt = -math.expm1(-t)
            s = Q[i] / St[i]
            phi[i] += wt[m] * (s + (psi - s) * omt / t)
            psi = psi * (1.0 - omt) + s * omt
        for i in range(Nx - 1, -1, -1):
            t = St[i] * dx[i] / mm
            omt = -math.expm1(-t)
            s = Q[i] / St[i]
            phi[i] += wt[m] * (s + (psi - s) * omt / t)
            psi = psi * (1.0 - omt) + s * omt
    return phi


@njit(cache=True)
def sweep_angular(St, Q, dx, mu, wt, out):
    """Same sweep, writing cell-average angular fluxes into out (2*Nmu, Nx):
    rows 0..Nmu-1 are +mu, rows Nmu..2Nmu-1 are -mu."""
    Nx = St.shape[0]
    Nm = mu.shape[0]
    for m in range(Nm):
        mm = mu[m]
        psi = 0.0
        tau = 0.0
        for i in range(Nx):
            t = St[i] * dx[i] / mm
            omt = -math.expm1(-t)
            psi = psi * (1.0 - omt) + Q[i] / St[i] * omt
            tau += t
        A = math.exp(-tau)
        B = psi
        psi = 0.0
        for i in range(Nx - 1, -1, -1):
            t = St[i] * dx[i] / mm
            omt = -math.expm1(-t)
            psi = psi * (1.0 - omt) + Q[i] / St[i] * omt
        C = psi
        psi = (A * B + C) / (1.0 - A * A)
        for i in range(Nx):
            t = St[i] * dx[i] / mm
            omt = -math.expm1(-t)
            s = Q[i] / St[i]
            out[m, i] = s + (psi - s) * omt / t
            psi = psi * (1.0 - omt) + s * omt
        for i in range(Nx - 1, -1, -1):
            t = St[i] * dx[i] / mm
            omt = -math.expm1(-t)
            s = Q[i] / St[i]
            out[Nm + m, i] = s + (psi - s) * omt / t
            psi = psi * (1.0 - omt) + s * omt


@njit(cache=True)
def march(u, sig_s, sig_a, zone, N, alpha, eps, hidx, hw, dx, mu, wt, tol, maxit, Psi):
    """March the slowing-down problem from u[0] (E_top) to u[-1] (E_bot).
    sig_s, sig_a : (K, Z, NE) microscopic [b] per nuclide and temperature zone
    zone, N      : (K, Nx) zone index and number density [1/b-cm] per cell
    alpha, eps   : (K,) kinematic alpha and max lethargy gain (inf for H)
    hidx, hw     : (K, NE) bracket index/weight of u_n - eps_k in the grid
                   (-1: falls above E_top, -2: eps infinite)."""
    K, Nx = N.shape
    NE = u.shape[0]
    F = np.empty((NE, Nx))
    G = np.zeros((K, NE, Nx))  # G_k(u_n) = int_{-inf}^{u_n} Sigma_s,k F e^{u'} du'
    Ss0 = np.zeros((K, Nx))
    for k in range(K):
        for i in range(Nx):
            Ss0[k, i] = N[k, i] * sig_s[k, zone[k, i], 0]
    St = np.zeros(Nx)
    q = np.zeros(Nx)
    c = np.zeros(Nx)
    Gk = np.zeros((K, Nx))
    iters = np.zeros(NE, dtype=np.int64)
    for n in range(NE):
        St[:] = 0.0
        q[:] = 0.0
        c[:] = 0.0
        eu = math.exp(u[n])
        h = u[n] - u[n - 1] if n > 0 else 0.0
        for k in range(K):
            j = hidx[k, n]
            for i in range(Nx):
                z = zone[k, i]
                Ssk = N[k, i] * sig_s[k, z, n]
                St[i] += Ssk + N[k, i] * sig_a[k, z, n]
                if n == 0:
                    Gk[k, i] = Ss0[k, i]  # F = 1 above E_top
                else:
                    fprev = N[k, i] * sig_s[k, z, n - 1] * F[n - 1, i] * math.exp(u[n - 1])
                    Gk[k, i] = G[k, n - 1, i] + 0.5 * h * fprev
                    c[i] += 0.5 * h * Ssk / (1.0 - alpha[k])
                if j == -2:
                    Glow = 0.0
                elif j == -1:
                    Glow = Ss0[k, i] * math.exp(u[n] - eps[k])
                else:
                    w = hw[k, n]
                    hj = u[j + 1] - u[j]
                    fj = N[k, i] * sig_s[k, z, j] * F[j, i] * math.exp(u[j])
                    fj1 = N[k, i] * sig_s[k, z, j + 1] * F[j + 1, i] * math.exp(u[j + 1])
                    ft = fj + w * (fj1 - fj)
                    Glow = G[k, j, i] + 0.5 * w * hj * (fj + ft)
                q[i] += (Gk[k, i] - Glow) / eu / (1.0 - alpha[k])
        # within-point self-scatter by source iteration (c/St ~ du/2, tiny)
        Fi = F[n - 1].copy() if n > 0 else np.ones(Nx)
        for it in range(maxit):
            Fn = sweep_reflective(St, 0.5 * (q + c * Fi), dx, mu, wt)
            err = 0.0
            for i in range(Nx):
                d = abs(Fn[i] - Fi[i]) / abs(Fn[i])
                if d > err:
                    err = d
            Fi = Fn
            if err < tol:
                break
        iters[n] = it + 1
        F[n] = Fi
        if Psi.shape[0] == NE:
            buf = np.empty((2 * mu.shape[0], Nx))
            sweep_angular(St, 0.5 * (q + c * Fi), dx, mu, wt, buf)
            for a in range(buf.shape[0]):
                for i in range(Nx):
                    Psi[n, a, i] = buf[a, i]
        for k in range(K):
            for i in range(Nx):
                if n == 0:
                    G[k, n, i] = Gk[k, i]
                else:
                    G[k, n, i] = (
                        Gk[k, i] + 0.5 * h * N[k, i] * sig_s[k, zone[k, i], n] * F[n, i] * eu
                    )
    return F, iters


# --------------------------------------------------------------------------
# Problem assembly
# --------------------------------------------------------------------------
class Nuclide:
    """name, mass A, and either constant xs or a resonance ladder."""

    def __init__(self, name, A, sig_s=None, sig_a_2200=0.0, res=None, R_fm=None):
        self.name, self.A = name, A
        self.sig_s, self.sig_a_2200 = sig_s, sig_a_2200
        self.res, self.R_fm = res, R_fm

    def xs(self, E, T):
        if self.res is None:
            ss = np.full_like(E, self.sig_s)
            sa = self.sig_a_2200 * np.sqrt(0.0253 / E)
            return ss, sa
        sa, ss = slbw(E, self.res, self.A, T, self.R_fm)
        return ss, sa


def solve(u, dx, nuclides, N, T, n_mu=16, tol=1e-11, maxit=50, angular=False):
    """nuclides: list of K Nuclide; N: (K, Nx) densities; T: (K, Nx) temperatures
    (cells with the same temperature share a cross-section zone).
    angular=True also returns cell-average angular fluxes (float32, NE x 2Nmu x Nx)."""
    E = E_TOP * np.exp(-u)
    K, Nx = N.shape
    zone = np.zeros((K, Nx), dtype=np.int64)
    temps = []
    for k in range(K):
        tk, zk = np.unique(np.round(T[k], 3), return_inverse=True)
        temps.append(tk)
        zone[k] = zk
    Z = max(len(t) for t in temps)
    sig_s = np.zeros((K, Z, len(u)))
    sig_a = np.zeros((K, Z, len(u)))
    for k, nuc in enumerate(nuclides):
        for z, Tz in enumerate(temps[k]):
            sig_s[k, z], sig_a[k, z] = nuc.xs(E, Tz)
    alpha = np.array([((nuc.A - 1.0) / (nuc.A + 1.0)) ** 2 for nuc in nuclides])
    eps = np.array([math.inf if a == 0 else -math.log(a) for a in alpha])
    assert np.diff(u).max() < eps.min(), "grid step must stay below the smallest lethargy gain"
    hidx = np.full((K, len(u)), -2, dtype=np.int64)
    hw = np.zeros((K, len(u)))
    for k in range(K):
        if math.isinf(eps[k]):
            continue
        ut = u - eps[k]
        j = np.searchsorted(u, ut, side="right") - 1
        j = np.minimum(j, len(u) - 2)
        pre = ut < u[0]
        jj = np.maximum(j, 0)
        hw[k] = (ut - u[jj]) / (u[jj + 1] - u[jj])
        hidx[k] = np.where(pre, -1, jj)
    mu, wt = np.polynomial.legendre.leggauss(n_mu)
    pos = mu > 0
    Psi = np.zeros((len(u), n_mu, Nx), np.float32) if angular else np.zeros((1, 1, 1), np.float32)
    F, iters = march(
        u,
        sig_s,
        sig_a,
        zone,
        N,
        alpha,
        eps,
        hidx,
        hw,
        dx,
        mu[pos].copy(),
        wt[pos].copy(),
        tol,
        maxit,
        Psi,
    )
    info = dict(
        sig_s=sig_s, sig_a=sig_a, zone=zone, iters=iters, mu=mu[pos].copy(), wt=wt[pos].copy()
    )
    if angular:
        info["Psi"] = Psi
    return F, info


def trap_weights(u):
    w = np.zeros_like(u)
    d = np.diff(u)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w
