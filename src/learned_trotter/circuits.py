"""Layered circuits (dense simulation in torch) and rotation counting.

Layer kinds
-----------
X   : n rotations exp(-i phi_j X_j / 2)
D   : (n-1) ZZ rotations exp(-i theta_j Z_j Z_{j+1} / 2) and n Z rotations (all commute: one layer)
RY  : n rotations about Y;  RZ : n rotations about Z
CZE / CZO : fixed Clifford CZ layers on even / odd bonds (no rotations, no T cost)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

Spec = tuple[str, ...]


def n_angles(kind: str, n: int) -> int:
    """Number of free rotation angles in one layer."""
    return {"X": n, "D": 2 * n - 1, "RY": n, "RZ": n, "CZE": 0, "CZO": 0}[kind]


def spec_angles(spec: Spec, n: int) -> int:
    """Total number of angles of a circuit."""
    return sum(n_angles(k, n) for k in spec)


def rotation_count(spec: Spec, n: int) -> int:
    """Number of non-Clifford rotations R in the circuit (CZ layers are Clifford)."""
    return spec_angles(spec, n)


def strang_spec(k: int) -> Spec:
    """X D X D ... D X with k diagonal layers (k Strang steps, adjacent half-steps merged)."""
    return tuple(["X"] + ["D", "X"] * k)


def hea_spec(layers: int) -> Spec:
    """Generic hardware-efficient brickwork: (RY RZ CZE CZO) x layers, then RY RZ."""
    return tuple(["RY", "RZ", "CZE", "CZO"] * layers + ["RY", "RZ"])


@lru_cache(maxsize=None)
def _tables(n: int) -> dict[str, torch.Tensor]:
    idx = np.arange(2**n)
    bits = np.stack([(idx >> (n - 1 - j)) & 1 for j in range(n)], axis=1)  # (D, n)
    s = 1 - 2 * bits
    szz = s[:, :-1] * s[:, 1:]

    def cz(bonds: range) -> np.ndarray:
        ph = np.zeros(2**n)
        for j in bonds:
            ph += bits[:, j] * bits[:, j + 1]
        return np.where(ph % 2 == 0, 1.0, -1.0)

    return {
        "s": torch.tensor(s, dtype=torch.float64),
        "szz": torch.tensor(szz, dtype=torch.float64),
        "cze": torch.tensor(cz(range(0, n - 1, 2)), dtype=torch.complex128),
        "czo": torch.tensor(cz(range(1, n - 1, 2)), dtype=torch.complex128),
    }


def _apply_1q(v: torch.Tensor, mat: torch.Tensor, j: int, n: int) -> torch.Tensor:
    """Left-multiply the dense operator ``v`` by a 2x2 matrix acting on qubit ``j`` (cheap contraction)."""
    d = v.shape[0]
    w = v.reshape(2**j, 2, (2 ** (n - j - 1)) * d)
    return torch.einsum("ab,ibk->iak", mat, w).reshape(d, d)


def _x_mats(ang: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(ang / 2).to(torch.complex128), torch.sin(ang / 2).to(torch.complex128)
    return torch.stack([c, -1j * s, -1j * s, c], dim=-1).reshape(-1, 2, 2)


def _ry_mats(ang: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(ang / 2).to(torch.complex128), torch.sin(ang / 2).to(torch.complex128)
    return torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)


def apply_layer(v: torch.Tensor, kind: str, ang: torch.Tensor, n: int) -> torch.Tensor:
    """Left-multiply the dense operator ``v`` by one layer."""
    t = _tables(n)
    if kind in ("X", "RY"):
        mats = _x_mats(ang) if kind == "X" else _ry_mats(ang)
        for j in range(n):
            v = _apply_1q(v, mats[j], j, n)
        return v
    if kind == "D":
        phase = -(t["szz"] @ ang[: n - 1] + t["s"] @ ang[n - 1:]) / 2
        return torch.exp(1j * phase)[:, None] * v
    if kind == "RZ":
        phase = -(t["s"] @ ang) / 2
        return torch.exp(1j * phase)[:, None] * v
    if kind == "CZE":
        return t["cze"][:, None] * v
    if kind == "CZO":
        return t["czo"][:, None] * v
    raise ValueError(kind)


_PAULI = [
    torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
]


def _layer_qubit_groups(kind: str, n: int) -> list[tuple[int, ...]]:
    """Qubits touched by each rotation of a layer (for the Pauli noise model)."""
    if kind == "D":
        return [(j, j + 1) for j in range(n - 1)] + [(j,) for j in range(n)]
    if kind in ("X", "RY", "RZ"):
        return [(j,) for j in range(n)]
    return []


def build_unitary(
    angles: torch.Tensor,
    spec: Spec,
    n: int,
    pauli_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Dense unitary of the circuit.

    With ``pauli_p`` > 0 every rotation is followed (at the end of its layer) by a uniformly random
    non-identity Pauli on its qubits with probability p: one Monte Carlo trajectory of a
    depolarising channel per rotation.
    """
    v = torch.eye(2**n, dtype=torch.complex128)
    pos = 0
    eye = torch.eye(2, dtype=torch.complex128)
    for kind in spec:
        k = n_angles(kind, n)
        v = apply_layer(v, kind, angles[pos: pos + k], n)
        pos += k
        if pauli_p > 0 and k > 0:
            assert rng is not None
            per_qubit: dict[int, torch.Tensor] = {}
            for qs in _layer_qubit_groups(kind, n):
                if rng.random() < pauli_p:
                    while True:
                        draw = rng.integers(0, 4, size=len(qs))
                        if draw.any():
                            break
                    for q, d in zip(qs, draw):
                        if d > 0:
                            per_qubit[q] = _PAULI[d - 1] @ per_qubit.get(q, eye)
            for q, m in per_qubit.items():
                v = _apply_1q(v, m, q, n)
    return v
