"""k-infinity of the 0D energy problem by power iteration on the fission source.

Each iteration solves solve_energy with the fission source of the last flux divided by k. In an
infinite medium the source shape depends on the flux only through how chi varies with incoming
energy, which is weak, so the iteration converges in a few solves.
"""

from dataclasses import dataclass

import numpy as np

from zerog.energy.fission import fission_hats, fission_source, production_rates, shield_fission
from zerog.energy.inelastic import restrict
from zerog.energy.subproblem import solve_energy

MAX_ITERATIONS = 100
K_TOLERANCE = 1e-9  # relative change in k between iterations
SOURCE_TOLERANCE = 1e-8  # largest relative change in the restricted production, per component


@dataclass(frozen=True)
class EigenSolution:
    """k, the flux normalized to one fission neutron produced, and iteration counts."""

    k: float
    flux: np.ndarray
    iterations: int
    sweeps: int
    history: np.ndarray  # k after each iteration
    block_marches: int = 0
    shield: np.ndarray | None = None  # as EnergySolution.shield, for the last solve
    fission: object = None  # the fission kernels shielded by it


def solve_k(ops, fission, removal, scatter, streaming, production, anisotropy="separable"):
    """k-infinity and its flux. removal, scatter and streaming are as for solve_energy;
    production (terms) holds each term's effective density for fission emission.

    The flux is normalized so that it produces one fission neutron, so the source it solves with
    is that production's emission divided by k.
    """
    production = np.asarray(production, dtype=float)
    flux = np.ones(ops.energy.shape[0])
    produced = production_rates(fission, production, ops.energy, flux, ops.rate_weights).sum()
    if produced <= 0.0:
        raise ValueError("this problem produces no fission neutrons")
    flux /= produced
    k = 1.0
    shape, _ = fission_hats(fission, production, ops.energy, flux, ops.rate_weights)
    history = []
    sweeps = 0
    marches = 0
    current = fission
    for iteration in range(1, MAX_ITERATIONS + 1):
        source = fission_source(ops, current, production, flux) / k
        # The last flux's inelastic emission starts the lagged sweeps, since it changes little.
        initial = (
            restrict(ops.inelastic, ops.energy, flux, ops.rate_weights) if iteration > 1 else None
        )
        solution = solve_energy(ops, removal, scatter, streaming, source, anisotropy, initial)
        sweeps += solution.sweeps
        marches += solution.block_marches
        # Where probability tables apply, production follows the bands' shielded fission.
        current = shield_fission(fission, ops.urr, solution.shield, ops.energy)
        produced = production_rates(
            current, production, ops.energy, solution.flux, ops.rate_weights
        ).sum()
        # The last flux produced one neutron, emitted as 1 / k.
        k_new = k * produced
        flux = solution.flux / produced
        new_shape, _ = fission_hats(current, production, ops.energy, flux, ops.rate_weights)
        history.append(k_new)
        change = np.abs(new_shape - shape).max() / np.abs(new_shape).max()
        if abs(k_new - k) <= K_TOLERANCE * k_new and change <= SOURCE_TOLERANCE:
            return EigenSolution(
                k_new,
                flux,
                iteration,
                sweeps,
                np.array(history),
                marches,
                solution.shield,
                current,
            )
        k = k_new
        shape = new_shape
    raise RuntimeError(f"power iteration did not converge in {MAX_ITERATIONS} iterations")
