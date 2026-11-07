"""Classical reference: Lanczos (Krylov) propagation of one state with m matrix-vector products."""
from __future__ import annotations

import numpy as np


def lanczos_propagate(h: np.ndarray, psi: np.ndarray, t: float, m: int) -> np.ndarray:
    """Approximate exp(-i H t) psi in an m-dimensional Krylov space (full re-orthogonalisation)."""
    d = len(psi)
    m = min(m, d)
    q = np.zeros((m, d), dtype=complex)
    alpha = np.zeros(m)
    beta = np.zeros(m)
    q[0] = psi / np.linalg.norm(psi)
    for j in range(m):
        w = h @ q[j]
        alpha[j] = np.real(np.vdot(q[j], w))
        w = w - alpha[j] * q[j] - (beta[j - 1] * q[j - 1] if j > 0 else 0)
        for k in range(j + 1):
            w = w - np.vdot(q[k], w) * q[k]
        if j + 1 < m:
            beta[j] = np.linalg.norm(w)
            if beta[j] < 1e-13:
                m = j + 1
                break
            q[j + 1] = w / beta[j]
    tri = np.diag(alpha[:m]) + np.diag(beta[: m - 1], 1) + np.diag(beta[: m - 1], -1)
    ev, evec = np.linalg.eigh(tri)
    coeff = evec @ (np.exp(-1j * ev * t) * evec[0].conj())
    return np.linalg.norm(psi) * (q[:m].T @ coeff)
