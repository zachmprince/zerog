"""Figures for the toy rank study (reads results.npz from run_cases.py)."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from slowdown import E_TOP

STUDY = Path(__file__).resolve().parents[1]
R = np.load(STUDY / "results" / "results.npz", allow_pickle=True)
OUT = STUDY / "figures"

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CASE_COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
CASE_LBL = {
    "A": "A  one pin, 900 K",
    "B": "B  one pin, 600–1200 K profile",
    "C": "C  one pin + rim absorber",
    "D": "D  two different pins",
}
BLUE = {"1 band": "#0d366b", "4 bands": "#2a78d6", "8 bands": "#86b6ef"}

plt.rcParams.update(
    {
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    }
)


def conv_rank(err, tol):
    above = np.where(err >= tol)[0]
    return 1 if len(above) == 0 else above[-1] + 2


def subtitle(ax, s):
    ax.text(0, 1.015, s, transform=ax.transAxes, color=INK2, fontsize=10, va="bottom")


# ---------------------------------------------------------------- Figure 1
fig, axs = plt.subplots(3, 1, figsize=(7, 12.5), gridspec_kw=dict(hspace=0.55))

ax = axs[0]
for k in "ABCD":
    e = R[f"{k}|angular|cell"]
    M = np.arange(1, len(e) + 1)
    ax.semilogy(M, e, color=CASE_COL[k], label=CASE_LBL[k], solid_capstyle="round")
    m = conv_rank(e, 1e-3)
    ax.plot(m, 1e-3, "o", ms=8, color=CASE_COL[k], mec=SURF, mew=2, zorder=5)
for tol, txt in [(1e-2, "1%"), (1e-3, "0.1%")]:
    ax.axhline(tol, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(44.5, tol * 1.15, txt, color=INK2, ha="right", va="bottom", fontsize=10)
ax.set_xlim(0, 45)
ax.set_ylim(1e-5, 2)
ax.set_xlabel("modes M  (space-angle × energy pairs)")
ax.set_ylabel("worst per-cell capture error")
ax.set_title("Capture rates converge in ~15–30 modes", pad=24)
subtitle(ax, "rank-M truncation of ψ(x,μ,E); dots = modes needed for 0.1%")
ax.legend(loc="upper right", fontsize=9.5)

ax = axs[1]
for k in "ABCD":
    S = R[f"{k}|angular|S"]
    ax.semilogy(np.arange(1, 61), S[:60] / S[0], color=CASE_COL[k], label=CASE_LBL[k])
ax.set_xlim(0, 60)
ax.set_ylim(1e-6, 1.5)
ax.set_xlabel("mode index m")
ax.set_ylabel("σₘ / σ₁")
ax.set_title("Singular values decay fast in every case", pad=24)
subtitle(ax, f"vs. {len(R['A|u']):,}–{len(R['D|u']):,} pointwise energies")
ax.legend(loc="upper right", fontsize=9.5)

ax = axs[2]
for b in ["1 band", "4 bands", "8 bands"]:
    e = R[f"A|{b}|cell"]
    M = np.arange(1, len(e) + 1)
    ax.semilogy(M, e, color=BLUE[b], label=b)
    m = conv_rank(e, 1e-3)
    ax.plot(m, 1e-3, "o", ms=8, color=BLUE[b], mec=SURF, mew=2, zorder=5)
    ax.annotate(
        f"{b}: {m}", (m, 1e-3), xytext=(7, 7), textcoords="offset points", color=INK, fontsize=10
    )
ax.axhline(1e-3, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.set_xlim(0, 60)
ax.set_ylim(1e-5, 2)
ax.set_xlabel("total modes across all energy bands")
ax.set_ylabel("worst per-cell capture error")
ax.set_title("Splitting energy into bands costs more modes", pad=24)
subtitle(ax, "case A, scalar flux; modes pooled across bands by singular value")
ax.legend(loc="upper right", fontsize=9.5)

fig.savefig(OUT / "fig1_rank.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
u, xc, tag = R["A|u"], R["A|xc"], R["A|tag"]
E = E_TOP * np.exp(-u)
F, Em, Xm = R["A|F"], R["A|Emodes"], R["A|Xmodes"]
win = (E > 4) & (E < 120)
fuel = tag == "fuel"
x_if = xc[fuel].max() + 0.5 * R["A|dx"][fuel][-1]

fig = plt.figure(figsize=(7, 12.5))
gs = fig.add_gridspec(5, 2, height_ratios=[1.5, 1, 1, 1, 1], hspace=0.62, wspace=0.28)
ax = fig.add_subplot(gs[0, :])
pts = [
    (0, "fuel center", "#0d366b"),
    (np.where(fuel)[0][-1], "fuel surface", "#2a78d6"),
    (len(xc) - 1, "moderator", "#86b6ef"),
]
for i, lab, c in pts:
    ax.semilogx(E[win], F[win, i], color=c, lw=1.4, label=lab)
ax.set_ylim(0, 1.05)
ax.set_xlim(4, 120)
ax.set_xlabel("energy (eV)")
ax.set_ylabel("E·φ (flux / lethargy)")
ax.set_title("Reference flux, case A", pad=24)
subtitle(ax, "pointwise slowing-down solve; deep spatial self-shielding in the fuel")
ax.legend(loc="lower left", bbox_to_anchor=(0.2, 0.03), fontsize=9.5, ncol=1, handlelength=1.4)

for m in range(4):
    em = Em[:, m] / np.abs(Em[win, m]).max()
    xm = Xm[:, m] / np.abs(Xm[:, m]).max()
    s = np.sign(xm[np.argmax(np.abs(xm))])
    em, xm = em * s, xm * s
    a1 = fig.add_subplot(gs[m + 1, 0])
    a1.semilogx(E[win], em[win], color="#2a78d6", lw=1.2)
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_xlim(4, 120)
    a1.set_ylabel(f"mode {m + 1}")
    if m == 0:
        a1.set_title("Energy modes εₘ(E)", fontsize=12, pad=8)
    if m == 3:
        a1.set_xlabel("energy (eV)")
    else:
        a1.set_xticklabels([])
    a2 = fig.add_subplot(gs[m + 1, 1])
    a2.axvspan(0, x_if, color="#f0efec", lw=0, zorder=0)
    a2.plot(xc, xm, color="#2a78d6", lw=2)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xlim(0, xc.max())
    if m == 0:
        a2.set_title("Spatial modes Xₘ(x)", fontsize=12, pad=8)
        a2.text(
            x_if / 2,
            0.06,
            "fuel",
            transform=a2.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=INK2,
            fontsize=9,
        )
    if m == 3:
        a2.set_xlabel("x (cm)")
    else:
        a2.set_xticklabels([])
fig.savefig(OUT / "fig2_modes.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("ok")
