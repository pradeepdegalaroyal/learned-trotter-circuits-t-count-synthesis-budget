"""Experiment tasks (each task = one disordered chain instance) and aggregation helpers.

All tasks are top-level functions returning plain records so they can run in a process pool.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from .circuits import (Spec, build_unitary, hea_spec, n_angles, rotation_count, spec_angles,
                       strang_spec)
from .ftcost import t_count
from .hamiltonian import dense_hamiltonian, exact_propagator, make_instance
from .krylov import lanczos_propagate
from .learn import fit
from .metrics import spectral_error, trace_infidelity
from .product_formulas import strang, suzuki4

STRANG_K = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40)
SUZUKI_R = (1, 2, 3, 4, 6, 8)
EPS_TARGETS = (0.2, 0.1, 0.05)


def _eval(u: np.ndarray, spec: Spec, ang: Any, n: int) -> tuple[float, float]:
    v = build_unitary(torch.as_tensor(ang, dtype=torch.float64), spec, n).numpy()
    return spectral_error(u, v), trace_infidelity(u, v)


def accuracy_task(n: int, t: float, seed: int, ks: tuple[int, ...], max_iter: int) -> dict[str, Any]:
    """Product formulas (all budgets) and Strang-initialised learned circuits for one instance."""
    torch.set_num_threads(1)
    inst = make_instance(n, seed)
    u = exact_propagator(inst, t)
    rows: list[dict[str, Any]] = []
    params: dict[str, list[float]] = {}

    def add(method: str, k: int, spec: Spec, e: float, f: float, secs: float = 0.0) -> None:
        rows.append(dict(n=n, t=t, seed=seed, method=method, K=k, R=rotation_count(spec, n),
                         eps_spec=e, infid=f, fit_seconds=secs))

    for k in STRANG_K:
        spec, ang = strang(inst, t, k)
        add("strang", k, spec, *_eval(u, spec, ang, n))
    for r in SUZUKI_R:
        spec, ang = suzuki4(inst, t, r)
        add("suzuki4", 5 * r, spec, *_eval(u, spec, ang, n))
    for k in ks:
        spec, ang = strang(inst, t, k)
        th, _, secs = fit(spec, n, u, ang.numpy(), max_iter=max_iter)
        add("learned", k, spec, *_eval(u, spec, th, n), secs)
        params[f"{n}|{t}|{seed}|{k}"] = th.tolist()
    return {"rows": rows, "params": params}


def _shared_map(spec: Spec, n: int) -> np.ndarray:
    """Index of the shared scale for every angle: X layer -> 1 scale, D layer -> 2 scales (ZZ, Z)."""
    idx, nxt = [], 0
    for kind in spec:
        if kind == "X":
            idx += [nxt] * n
            nxt += 1
        elif kind == "D":
            idx += [nxt] * (n - 1) + [nxt + 1] * n
            nxt += 2
    return np.array(idx)


def ablation_task(n: int, t: float, k: int, seed: int, max_iter: int) -> list[dict[str, Any]]:
    """Parameterisation / initialisation ablation at a fixed rotation budget."""
    torch.set_num_threads(1)
    inst = make_instance(n, seed)
    u = exact_propagator(inst, t)
    spec, base = strang(inst, t, k)
    rng = np.random.default_rng(seed + 77)
    rows: list[dict[str, Any]] = []

    def add(arm: str, spec_: Spec, e: float, f: float, secs: float, npar: int) -> None:
        rows.append(dict(n=n, t=t, seed=seed, K=k, arm=arm, R=rotation_count(spec_, n),
                         n_params=npar, eps_spec=e, infid=f, fit_seconds=secs))

    add("Strang (no learning)", spec, *_eval(u, spec, base, n), 0.0, 0)
    th, _, s = fit(spec, n, u, base.numpy(), max_iter=max_iter)
    add("free angles, Strang start", spec, *_eval(u, spec, th, n), s, len(th))
    th, _, s = fit(spec, n, u, rng.standard_normal(len(base)), max_iter=max_iter)
    add("free angles, random start", spec, *_eval(u, spec, th, n), s, len(th))
    smap = torch.tensor(_shared_map(spec, n))
    nscale = int(smap.max()) + 1
    th, _, s = fit(spec, n, u, np.ones(nscale), to_angles=lambda x: base * x[smap], max_iter=max_iter)
    ang = base * torch.tensor(th)[smap]
    add("shared scales (3 per step)", spec, *_eval(u, spec, ang, n), s, nscale)
    r_ref = rotation_count(spec, n)
    layers = max(1, r_ref // (2 * n) - 1)
    hspec = hea_spec(layers)
    th, _, s = fit(hspec, n, u, 0.5 * rng.standard_normal(spec_angles(hspec, n)), max_iter=max_iter)
    add("generic brickwork, random start", hspec, *_eval(u, hspec, th, n), s, len(th))
    return rows


def _noise_arms(inst: Any, t: float, k: int, learned: np.ndarray) -> dict[str, tuple[Spec, torch.Tensor]]:
    return {
        "Strang, same R": strang(inst, t, k),
        "Strang, 2K steps": strang(inst, t, 2 * k),
        "Suzuki-4, r=2": suzuki4(inst, t, 2),
        "learned": (strang_spec(k), torch.tensor(learned, dtype=torch.float64)),
    }


def noise_task(n: int, t: float, k: int, seed: int, learned: list[float], sigmas: tuple[float, ...],
               draws: int, pauli_ps: tuple[float, ...], traj: int) -> dict[str, Any]:
    """Coherent angle-noise and Pauli (depolarising) noise at fixed circuits."""
    torch.set_num_threads(1)
    inst = make_instance(n, seed)
    u = exact_propagator(inst, t)
    arms = _noise_arms(inst, t, k, np.array(learned))
    rng = np.random.default_rng(5000 + seed)
    ang_rows: list[dict[str, Any]] = []
    pauli_rows: list[dict[str, Any]] = []
    d = 2**n
    for arm, (spec, ang) in arms.items():
        r = rotation_count(spec, n)
        for sigma in sigmas:
            for dr in range(draws):
                noisy = ang + sigma * torch.tensor(rng.standard_normal(len(ang)))
                e, _ = _eval(u, spec, noisy, n)
                ang_rows.append(dict(n=n, t=t, seed=seed, arm=arm, R=r, sigma=sigma, draw=dr, eps_spec=e))
        for p in pauli_ps:
            fids = []
            for _ in range(traj):
                v = build_unitary(ang, spec, n, pauli_p=p, rng=rng).numpy()
                fids.append(abs(np.trace(u.conj().T @ v)) ** 2 / d**2)
            pauli_rows.append(dict(n=n, t=t, seed=seed, arm=arm, R=r, p=p, infid=1 - float(np.mean(fids))))
    return {"angle": ang_rows, "pauli": pauli_rows}


def krylov_task(n: int, t: float, k: int, seed: int, learned: list[float], ms: tuple[int, ...],
                n_states: int) -> list[dict[str, Any]]:
    """Classical Lanczos reference versus circuit state errors on random product states."""
    torch.set_num_threads(1)
    inst = make_instance(n, seed)
    h = dense_hamiltonian(inst)
    u = exact_propagator(inst, t)
    rng = np.random.default_rng(900 + seed)
    states = []
    for _ in range(n_states):
        qs = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
        psi = np.array([1.0 + 0j])
        for q in qs:
            psi = np.kron(psi, q / np.linalg.norm(q))
        states.append(psi)
    rows = []
    for m in ms:
        err = np.mean([np.linalg.norm(lanczos_propagate(h, p, t, m) - u @ p) for p in states])
        rows.append(dict(n=n, t=t, seed=seed, method="lanczos", budget=m, budget_unit="matvecs",
                         state_error=float(err)))
    for arm, (spec, ang) in _noise_arms(inst, t, k, np.array(learned)).items():
        if arm == "Suzuki-4, r=2":
            continue
        v = build_unitary(ang, spec, n).numpy()
        phase = np.vdot(u.reshape(-1), v.reshape(-1))
        phase /= abs(phase)
        err = np.mean([np.linalg.norm(v @ p - phase * (u @ p)) for p in states])
        rows.append(dict(n=n, t=t, seed=seed, method=arm, budget=rotation_count(spec, n),
                         budget_unit="rotations", state_error=float(err)))
    return rows


# ----------------------------------------------------------------------------- aggregation
def t_cost_table(acc: pd.DataFrame, targets: tuple[float, ...] = EPS_TARGETS, c: float = 3.0) -> pd.DataFrame:
    """Per instance and target: cheapest circuit (min T over budgets) for each method family."""
    out = []
    for (n, t, seed), g in acc.groupby(["n", "t", "seed"]):
        for target in targets:
            best: dict[str, tuple[float, int, int]] = {}
            for method, gm in g.groupby("method"):
                for _, r in gm.iterrows():
                    tc = t_count(int(r.R), float(r.eps_spec), target, c)
                    if method not in best or tc < best[method][0]:
                        best[method] = (tc, int(r.K), int(r.R))
            tro = [best[m] for m in ("strang", "suzuki4") if m in best]
            if tro:
                best["best product formula"] = min(tro)
            for method, (tc, k, r) in best.items():
                out.append(dict(n=n, t=t, seed=seed, eps_target=target, method=method, T=tc, K=k, R=r))
    return pd.DataFrame(out)


def rotations_to_reach(acc_mean: pd.DataFrame, method: str, eps: float) -> float:
    """Log-log interpolated rotation count at which a method's mean error crosses ``eps``."""
    g = acc_mean[acc_mean.method == method].sort_values("R")
    r, e = g.R.to_numpy(float), g.eps_spec.to_numpy(float)
    for i in range(len(r) - 1):
        if (e[i] - eps) * (e[i + 1] - eps) <= 0 and e[i] != e[i + 1]:
            w = (math.log(eps) - math.log(e[i])) / (math.log(e[i + 1]) - math.log(e[i]))
            return float(math.exp(math.log(r[i]) + w * (math.log(r[i + 1]) - math.log(r[i]))))
    return math.nan


def hypothesis_checks(acc: pd.DataFrame, tc: pd.DataFrame, ab: pd.DataFrame, ang: pd.DataFrame,
                      pauli: pd.DataFrame, n: int, t: float, k: int, eps: float) -> dict[str, Any]:
    """Numerical verdicts for H1-H4 of docs/RESEARCH_GAP.md (all numbers derived from the result tables)."""
    out: dict[str, Any] = {}
    m = acc.groupby(["n", "t", "method", "K"]).eps_spec.mean()
    h1 = []
    for tt in sorted(acc[acc.n == n].t.unique()):
        for kk in (k, k + 1):
            if (n, tt, "learned", kk) in m.index:
                h1.append(dict(t=float(tt), K=kk, strang_error=float(m[(n, tt, "strang", kk)]),
                               learned_error=float(m[(n, tt, "learned", kk)]),
                               ratio=float(m[(n, tt, "strang", kk)] / m[(n, tt, "learned", kk)])))
    out["H1_error_ratio_equal_R"] = dict(rows=h1, min_ratio=min(r["ratio"] for r in h1),
                                         holds=bool(min(r["ratio"] for r in h1) >= 2.0))
    g = tc[(tc.n == n) & (tc.t == t) & (np.isclose(tc.eps_target, eps))]
    mean_t = g.groupby("method")["T"].apply(lambda s: float(s.mean()) if np.isfinite(s).all() else math.nan)
    ratio = float(mean_t["best product formula"] / mean_t["learned"])
    out["H2_t_count_ratio"] = dict(n=n, t=t, eps_target=eps, best_product_formula_T=float(mean_t["best product formula"]),
                                   learned_T=float(mean_t["learned"]), ratio=ratio,
                                   holds=bool(ratio >= 1.5))
    am = ang.groupby(["arm", "sigma"]).eps_spec.mean()
    pm = pauli.groupby(["arm", "p"]).infid.mean()
    pauli_ok = all(pm[("learned", p)] <= pm[("Strang, 2K steps", p)] for p in pauli.p.unique() if p > 0)
    ang_ratio = max(float(am[("learned", s)] / am[("Strang, same R", s)]) for s in ang.sigma.unique())
    out["H3_robustness"] = dict(pauli_learned_le_equal_error_strang_at_all_p=bool(pauli_ok),
                                max_angle_error_ratio_learned_over_equal_R_strang=ang_ratio,
                                holds=bool(pauli_ok and ang_ratio <= 2.0))
    a = ab.groupby("arm").eps_spec.mean()
    base = float(a["free angles, Strang start"])
    ratios = {name: float(a[name] / base) for name in a.index if name not in ("free angles, Strang start",)}
    out["H4_ablation_error_ratio_vs_method"] = dict(
        ratios=ratios,
        holds=bool(all(v >= 2.0 for kname, v in ratios.items() if kname != "Strang (no learning)")))
    return out


def headline_tables(acc: pd.DataFrame, tc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compact tables quoted in the README: error at equal rotation count, and T-count comparison."""
    rows = []
    g = acc[acc.method.isin(["strang", "learned"])].groupby(["n", "t", "K", "method"]).eps_spec.agg(["mean", "std"])
    for (n, t, k) in sorted({(a, b, c) for a, b, c, d in g.index if d == "learned"}):
        s, le = g.loc[(n, t, k, "strang")], g.loc[(n, t, k, "learned")]
        r = int(acc[(acc.n == n) & (acc.t == t) & (acc.K == k) & (acc.method == "learned")].R.iloc[0])
        rows.append(dict(n=n, t=t, K=k, R=r, strang_mean=s["mean"], strang_std=s["std"], learned_mean=le["mean"],
                         learned_std=le["std"], error_ratio=s["mean"] / le["mean"]))
    err = pd.DataFrame(rows)
    out = []
    for (n, t, e), gg in tc.groupby(["n", "t", "eps_target"]):
        a = gg[gg.method == "best product formula"]["T"]
        z = gg[gg.method == "learned"]["T"]
        zf = z[np.isfinite(z)]
        out.append(dict(n=n, t=t, eps_target=e, n_instances=len(z), learned_reached=len(zf),
                        pf_T_mean=a.mean(), pf_T_std=a.std(), learned_T_mean=zf.mean() if len(zf) else math.nan,
                        learned_T_std=zf.std() if len(zf) > 1 else math.nan,
                        ratio=a.mean() / zf.mean() if len(zf) == len(z) and len(z) else math.nan))
    return err, pd.DataFrame(out)
