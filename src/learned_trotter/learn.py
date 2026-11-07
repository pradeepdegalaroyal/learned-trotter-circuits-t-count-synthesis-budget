"""Fit circuit angles to a target propagator with L-BFGS (dense simulation)."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from .circuits import Spec, build_unitary


def trace_loss(v: torch.Tensor, u_target: torch.Tensor) -> torch.Tensor:
    """1 - |Tr(U^dag V)| / d (phase-invariant, zero iff V = e^{i phi} U)."""
    d = v.shape[0]
    return 1 - torch.abs(torch.sum(u_target.conj() * v)) / d


def fit(
    spec: Spec,
    n: int,
    u_target: np.ndarray,
    theta0: np.ndarray,
    to_angles: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_iter: int = 150,
) -> tuple[np.ndarray, float, float]:
    """Minimise the trace loss over ``theta`` (angles = ``to_angles(theta)``, identity by default).

    Returns (theta, final loss, wall seconds).
    """
    ut = torch.tensor(u_target, dtype=torch.complex128)
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    f = to_angles if to_angles is not None else (lambda x: x)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=30,
                            tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = trace_loss(build_unitary(f(theta), spec, n), ut)
        loss.backward()
        return loss

    t0 = time.perf_counter()
    opt.step(closure)
    with torch.no_grad():
        loss = float(trace_loss(build_unitary(f(theta), spec, n), ut))
    return theta.detach().numpy().copy(), loss, time.perf_counter() - t0
