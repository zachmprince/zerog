"""A thin Krylov interface, so that physics code never calls a solver library directly.

Operators are callables on flat float arrays. Only SciPy's GMRES sits behind it for now; petsc4py
can replace it here without touching physics code. Spatial subproblems may have effective
scattering ratios above one or below zero, so nothing here assumes a symmetric or definite
operator.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

RESTART = 50
MAX_ITERATIONS = 500


class KrylovError(RuntimeError):
    """A Krylov solve that did not reach its tolerance."""


def solve(
    apply,
    rhs,
    precondition=None,
    x0=None,
    atol=0.0,
    rtol=1e-10,
    restart=RESTART,
    max_iterations=MAX_ITERATIONS,
):
    """Solve apply(x) = rhs by GMRES. Returns x and the number of applications of the operator.

    precondition, if given, applies an approximate inverse. The solve stops once the residual is
    below max(atol, rtol * |rhs|), and raises KrylovError if it can't get there in max_iterations
    restart cycles.
    """
    rhs = np.asarray(rhs, dtype=float)
    n = rhs.shape[0]
    count = [0]

    def counted(x):
        count[0] += 1
        return apply(x)

    operator = LinearOperator((n, n), matvec=counted, dtype=float)
    preconditioner = None
    if precondition is not None:
        preconditioner = LinearOperator((n, n), matvec=precondition, dtype=float)
    if not np.any(rhs) and x0 is None:
        return np.zeros(n), 0
    x, info = gmres(
        operator,
        rhs,
        x0=x0,
        rtol=rtol,
        atol=atol,
        restart=min(restart, n),
        maxiter=max_iterations,
        M=preconditioner,
    )
    if info != 0:
        residual = np.linalg.norm(rhs - apply(x))
        raise KrylovError(f"GMRES stopped with info {info} at residual {residual:.3e}")
    return x, count[0]
