"""Operator-error metrics (global phase is always optimised away)."""
from __future__ import annotations

import numpy as np


def spectral_error(u: np.ndarray, v: np.ndarray) -> float:
    """min over phi of the spectral norm ||V - exp(i phi) U||.

    Exact: the eigenphases of U^dag V lie on an arc of width w; the optimal phase centres the arc
    and the error is 2 sin(w / 4).
    """
    w = u.conj().T @ v
    ph = np.sort(np.angle(np.linalg.eigvals(w)))
    gaps = np.diff(np.concatenate([ph, [ph[0] + 2 * np.pi]]))
    width = 2 * np.pi - gaps.max()
    return float(2 * np.sin(width / 4))


def trace_infidelity(u: np.ndarray, v: np.ndarray) -> float:
    """1 - |Tr(U^dag V)|^2 / d^2 (entanglement infidelity)."""
    d = u.shape[0]
    return float(1 - abs(np.trace(u.conj().T @ v)) ** 2 / d**2)
