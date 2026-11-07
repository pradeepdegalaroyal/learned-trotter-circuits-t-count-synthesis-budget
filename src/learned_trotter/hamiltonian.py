"""Disordered mixed-field Ising chain and its exact propagator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I = np.eye(2, dtype=complex)


@dataclass(frozen=True)
class Instance:
    """One disordered chain: H = sum_j J_j Z_j Z_{j+1} + sum_j hz_j Z_j + sum_j hx_j X_j."""

    n: int
    J: np.ndarray   # (n-1,) Ising couplings
    hz: np.ndarray  # (n,) longitudinal fields
    hx: np.ndarray  # (n,) transverse fields


def make_instance(n: int, seed: int, disorder: float = 0.25) -> Instance:
    """Draw a chain with multiplicative Gaussian disorder around (J, hx, hz) = (1, 1.05, 0.5)."""
    rng = np.random.default_rng(1000 + seed)
    j = 1.0 * (1 + disorder * rng.standard_normal(n - 1))
    hx = 1.05 * (1 + disorder * rng.standard_normal(n))
    hz = 0.5 * (1 + disorder * rng.standard_normal(n))
    return Instance(n, j, hz, hx)


def _kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def _site_op(n: int, site: int, op: np.ndarray) -> np.ndarray:
    return _kron_all([op if k == site else _I for k in range(n)])


def dense_hamiltonian(inst: Instance) -> np.ndarray:
    """Dense 2^n x 2^n Hamiltonian (qubit 0 = most significant bit)."""
    n = inst.n
    h = np.zeros((2**n, 2**n), dtype=complex)
    for j in range(n - 1):
        ops = [_I] * n
        ops[j], ops[j + 1] = _Z, _Z
        h += inst.J[j] * _kron_all(ops)
    for j in range(n):
        h += inst.hz[j] * _site_op(n, j, _Z) + inst.hx[j] * _site_op(n, j, _X)
    return h


def exact_propagator(inst: Instance, t: float) -> np.ndarray:
    """exp(-i H t) from a dense eigendecomposition (the classical ground truth)."""
    h = dense_hamiltonian(inst)
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * w * t)) @ v.conj().T
