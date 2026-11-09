"""Compare an infinite-medium k-infinity solve with its OpenMC twin's summary.

A summary is what benchmarks/05-fast-k-infinity/build.py writes: k, per-nuclide rate tallies and a
flux tally per energy bin, all per source neutron. ZeroG's eigenvalue flux produces one fission
neutron, which is emitted as 1 / k, so it is k times the twin's.
"""

from dataclasses import dataclass

import numpy as np

from zerog.energy.fission import production_rates
from zerog.energy.grid import panel_rate


@dataclass(frozen=True)
class RateComparison:
    score: str
    nuclide: str
    ours: float
    twin: float
    std: float

    @property
    def relative(self):
        return self.ours / self.twin - 1.0

    @property
    def sigmas(self):
        return (self.ours - self.twin) / self.std


@dataclass(frozen=True)
class TwinComparison:
    k: float
    k_twin: float
    k_std: float
    rates: list
    bin_low: np.ndarray
    bin_high: np.ndarray
    flux: np.ndarray  # per bin, per source neutron
    flux_twin: np.ndarray
    flux_std: np.ndarray

    @property
    def pcm(self):
        return (self.k - self.k_twin) * 1e5

    @property
    def tallied(self):
        """Bins the twin scored in; bins at the ends can be empty."""
        return self.flux_std > 0.0

    @property
    def z(self):
        t = self.tallied
        return np.where(t, (self.flux - self.flux_twin) / np.where(t, self.flux_std, 1.0), 0.0)

    def report(self, top_bins=3):
        """Lines as study 03 prints them."""
        lines = [f"  k {self.k:.6f} vs {self.k_twin:.6f} +- {self.k_std:.6f}: {self.pcm:+.1f} pcm"]
        for r in self.rates:
            lines.append(
                f"  {r.score:10s} {r.nuclide:6s} {r.ours:.6e} vs {r.twin:.6e} "
                f"({r.std / r.twin * 100:.3f}%): {r.relative * 100:+.3f}% ({r.sigmas:+.1f} sd)"
            )
        t = self.tallied
        z = self.z
        relative = np.where(t, self.flux / np.where(t, self.flux_twin, 1.0) - 1.0, 0.0)
        lines.append(
            f"  flux bins {t.sum()} of {t.shape[0]} tallied: "
            f"rms z {np.sqrt(np.mean(z[t] ** 2)):.2f}, max |z| {np.abs(z).max():.1f}, "
            f"median sd {np.median(self.flux_std[t] / self.flux_twin[t]) * 100:.3f}%"
        )
        for b in np.argsort(np.abs(z))[-top_bins:]:
            lines.append(
                f"    bin {self.bin_low[b]:.4g}-{self.bin_high[b]:.4g} eV: "
                f"{relative[b] * 100:+.3f}%, z {z[b]:+.1f}"
            )
        lines.append(f"  total flux {self.flux.sum() / self.flux_twin.sum() * 100 - 100:+.3f}%")
        return "\n".join(lines)


def compare(mix, k, flux, summary, shield=None, fission=None):
    """Compare a zerog.energy.mix.Mix's eigenvalue solution (k, flux) with a twin summary.

    Rates are (n,gamma), fission and nu-fission for the mix's nuclides; others are skipped. With
    probability tables, pass the solution's shield and shielded fission kernels: OpenMC scores
    (n,gamma) there as the band's absorption less fission, and fission as the band's.
    """
    from zerog.energy.urr import SHIELD_CAPTURE, SHIELD_FISSION, shielded

    ops = mix.ops
    energy = mix.energy
    scaled = k * flux
    kernels = fission
    if kernels is None:
        from zerog.energy.fission import shield_fission

        kernels = shield_fission(mix.fission, ops.urr, shield, energy)
    produced = production_rates(kernels, mix.densities, energy, scaled, ops.rate_weights)
    rates = []
    for row in summary["rates"]:
        if row["nuclide"] not in mix.names or row["mean"] <= 0.0:
            continue
        i = mix.names.index(row["nuclide"])
        if row["score"] == "(n,gamma)":
            curve = shielded(ops.urr, shield, i, SHIELD_CAPTURE, mix.terms[i].capture, energy)
            ours = mix.densities[i] * panel_rate(curve, scaled, ops.rate_weights)
        elif row["score"] == "fission":
            curve = shielded(ops.urr, shield, i, SHIELD_FISSION, mix.fission_xs(i), energy)
            ours = mix.densities[i] * panel_rate(curve, scaled, ops.rate_weights)
        elif row["score"] == "nu-fission":
            ours = produced[i]
        else:
            continue
        rates.append(
            RateComparison(row["score"], row["nuclide"], ours, row["mean"], row["std. dev."])
        )

    low = np.array(summary["edges_low"])
    high = np.array(summary["edges_high"])
    ours = np.empty(low.shape[0])
    for b in range(low.shape[0]):
        mask = (energy <= high[b]) & (energy >= low[b])
        piece = energy[mask]
        ours[b] = np.sum(
            0.5 * (scaled[mask][:-1] + scaled[mask][1:]) * np.log(piece[:-1] / piece[1:])
        )
    return TwinComparison(
        k=k,
        k_twin=summary["k"],
        k_std=summary["k_std"],
        rates=rates,
        bin_low=low,
        bin_high=high,
        flux=ours,
        flux_twin=np.array(summary["flux"]),
        flux_std=np.array(summary["flux_std"]),
    )
