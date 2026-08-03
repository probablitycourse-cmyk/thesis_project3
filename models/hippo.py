"""
HiPPO-LegS matrices and the NPLR (Normal Plus Low-Rank) decomposition
used to initialise S4 layers.

The continuous-time HiPPO-LegS ODE is

    dx/dt = (1/t) * (A x + B u(t))

with

    A[n,k] = -sqrt((2n+1)(2k+1))   if k < n
    A[n,n] = -(n+1)
    A[n,k] = 0                     if k > n
    B[n]   = sqrt(2n+1)

`nplr_decompose` rewrites A as

    A = V (Lambda - 0.5 q q*) V*

where V is unitary, Lambda is diagonal, and q is a single vector. This form
is what makes the fast frequency-domain kernel of S4 possible.
"""

from __future__ import annotations

import numpy as np


def hippo_legs_matrices(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the constant HiPPO-LegS matrices A (N, N) and B (N,)."""
    n = np.arange(N)
    k = np.arange(N)
    row_n, col_k = np.meshgrid(n, k, indexing="ij")

    A = -np.sqrt((2 * row_n + 1) * (2 * col_k + 1))
    A = np.where(col_k < row_n, A, 0.0)
    A[np.arange(N), np.arange(N)] = -(np.arange(N) + 1)

    B = np.sqrt(2 * np.arange(N) + 1)
    return A, B


def nplr_decompose(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose the HiPPO-LegS matrix into Normal-Plus-Low-Rank form.

    Returns
    -------
    Lambda : (N,) complex   diagonal part (Re(Lambda) == -0.5 for every entry)
    q      : (N,) complex   rank-1 correction vector (equals V* B exactly)
    V      : (N, N) complex unitary basis
    """
    A, _ = hippo_legs_matrices(N)
    p = np.sqrt(2 * np.arange(N) + 1)          # note: B == p exactly

    # Adding 0.5 * p p^T makes the off-diagonal part antisymmetric; subtracting
    # 0.5 * I then zeroes the diagonal, so S is exactly skew-symmetric.
    S = -(A + 0.5 * np.outer(p, p)) - 0.5 * np.eye(N)

    eigvals, V = np.linalg.eig(S)
    Lambda = -eigvals - 0.5
    q = V.conj().T @ p
    return Lambda, q, V


def verify_nplr(N: int = 32, tol: float = 1e-10) -> float:
    """Reconstruct A from (Lambda, q, V) and return the max absolute error."""
    A, _ = hippo_legs_matrices(N)
    Lambda, q, V = nplr_decompose(N)
    A_recon = (
        V @ np.diag(Lambda) @ V.conj().T
        - 0.5 * V @ np.outer(q, q.conj()) @ V.conj().T
    ).real
    err = float(np.max(np.abs(A - A_recon)))
    assert err < tol, f"NPLR reconstruction error {err:.2e} exceeds {tol:.0e}"
    return err


if __name__ == "__main__":
    for n in (8, 16, 32, 64):
        print(f"N={n:3d}  NPLR reconstruction error = {verify_nplr(n):.3e}")
