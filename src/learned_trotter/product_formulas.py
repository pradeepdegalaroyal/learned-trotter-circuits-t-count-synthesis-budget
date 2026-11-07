"""Strang (2nd-order) and Suzuki (4th-order) product formulas as layered angle vectors."""
from __future__ import annotations

import numpy as np
import torch

from .circuits import Spec, strang_spec
from .hamiltonian import Instance

SUZUKI_P = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))


def _dlayer(inst: Instance, tau: float) -> np.ndarray:
    return np.concatenate([2 * inst.J * tau, 2 * inst.hz * tau])


def _xlayer(inst: Instance, tau: float) -> np.ndarray:
    return 2 * inst.hx * tau


def sequence_from_steps(inst: Instance, taus: list[float]) -> tuple[Spec, torch.Tensor]:
    """Concatenate Strang steps X(tau/2) D(tau) X(tau/2), merging the touching X halves."""
    xs: list[np.ndarray] = [_xlayer(inst, taus[0] / 2)]
    ds: list[np.ndarray] = []
    for i, tau in enumerate(taus):
        ds.append(_dlayer(inst, tau))
        nxt = tau / 2 + (taus[i + 1] / 2 if i + 1 < len(taus) else 0.0)
        xs.append(_xlayer(inst, nxt))
    parts: list[np.ndarray] = [xs[0]]
    for d, x in zip(ds, xs[1:]):
        parts += [d, x]
    return strang_spec(len(taus)), torch.tensor(np.concatenate(parts), dtype=torch.float64)


def strang(inst: Instance, t: float, steps: int) -> tuple[Spec, torch.Tensor]:
    """Second-order formula with ``steps`` diagonal layers."""
    return sequence_from_steps(inst, [t / steps] * steps)


def suzuki4(inst: Instance, t: float, repeats: int) -> tuple[Spec, torch.Tensor]:
    """Fourth-order Suzuki recursion; each repeat is 5 Strang steps (so K = 5 * repeats)."""
    tau = t / repeats
    one = [SUZUKI_P * tau] * 2 + [(1 - 4 * SUZUKI_P) * tau] + [SUZUKI_P * tau] * 2
    return sequence_from_steps(inst, one * repeats)
