"""Fault-tolerant cost model: every non-Clifford rotation is synthesised into Clifford+T.

Assumption (stated here, not taken from the literature brief): a rotation synthesised to
spectral-norm precision eps costs ``c * log2(1/eps)`` T gates. Errors add (triangle inequality), so
a circuit with R rotations and algorithmic error eps_alg meets a total error target eps* with
per-rotation precision eps_s = (eps* - eps_alg) / R. The ratio between two methods depends only
weakly on ``c``.
"""
from __future__ import annotations

import math


def t_count(r: int, eps_alg: float, eps_target: float, c: float = 3.0) -> float:
    """T-count to reach ``eps_target`` in total, or +inf if eps_alg already exceeds the target."""
    budget = eps_target - eps_alg
    if budget <= 0:
        return math.inf
    return r * c * math.log2(r / budget)
