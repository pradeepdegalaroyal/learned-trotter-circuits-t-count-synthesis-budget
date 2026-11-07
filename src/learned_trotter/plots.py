"""Figures (earth-tone palette; grouped hatched bars, small multiples, empirical CDFs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BROWN, TEAL, OCHRE, SAND, OLIVE, TERRA = "#6b4423", "#1f7a78", "#c8922a", "#d8c7a3", "#5b6b3a", "#b04e2c"
COL = {"strang": BROWN, "suzuki4": OCHRE, "learned": TEAL, "best product formula": BROWN}
LAB = {"strang": "Strang (2nd order)", "suzuki4": "Suzuki (4th order)", "learned": "learned angles",
       "best product formula": "best product formula"}
DPI = 200


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_accuracy(acc: pd.DataFrame, path: Path, n: int, eps_line: float = 0.1) -> None:
    """Small multiples: spectral-norm error against rotation count, one panel per evolution time."""
    ts = sorted(acc[acc.n == n].t.unique())
    fig, axes = plt.subplots(1, len(ts), figsize=(4.0 * len(ts), 3.6), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], ts):
        g = acc[(acc.n == n) & (acc.t == t)]
        for m in ("strang", "suzuki4", "learned"):
            gm = g[g.method == m].groupby("R").eps_spec.agg(["mean", "std"]).reset_index()
            ax.errorbar(gm.R, gm["mean"], yerr=gm["std"].fillna(0), color=COL[m], marker="osd"[("strang", "suzuki4", "learned").index(m)],
                        ms=4, lw=1.6, capsize=2, label=LAB[m])
        ax.axhline(eps_line, color="k", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([50, 100, 200, 400, 800]); ax.set_xticklabels(["50", "100", "200", "400", "800"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"n = {n} qubits, t = {t:g}")
        ax.set_xlabel("non-Clifford rotations R (count)")
        ax.grid(alpha=0.25, which="both")
    axes[0][0].set_ylabel("spectral-norm error (dimensionless)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Operator error versus rotation budget (mean ± std over instances)", y=1.02)
    _save(fig, path)


def _bars(ax: plt.Axes, groups: list[str], series: dict[str, tuple[list[float], list[float], list[int]]],
          hatches: dict[str, str], width: float = 0.26) -> None:
    x = np.arange(len(groups))
    for i, (name, (mean, std, nreach)) in enumerate(series.items()):
        pos = x + (i - (len(series) - 1) / 2) * width
        vals = np.nan_to_num(mean, nan=0.0)
        ax.bar(pos, vals, width, yerr=np.nan_to_num(std), color=COL[name], edgecolor="k", hatch=hatches[name],
               capsize=2, label=LAB[name], alpha=0.9)
        for p, m, k in zip(pos, mean, nreach):
            if np.isnan(m):
                ax.text(p, ax.get_ylim()[1] * 0.02 if ax.get_yscale() == "linear" else 1, "not\nreached", ha="center",
                        fontsize=6, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(groups)


def fig_tcount(tc: pd.DataFrame, path: Path, eps: float, n_ref: int = 6) -> None:
    """Grouped hatched bars: T-count to reach a total error target, by system size and by time."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"wspace": 0.28})
    hatches = {"strang": "//", "suzuki4": "..", "learned": "xx"}
    for ax, key, sel, xl in ((axes[0], "n", tc[(tc.t == 1.0)], "number of qubits n (t = 1)"),
                             (axes[1], "t", tc[tc.n == n_ref], f"evolution time t (n = {n_ref})")):
        sel = sel[sel.eps_target == eps]
        groups = sorted(sel[key].unique())
        series = {}
        for m in hatches:
            means, stds, nr = [], [], []
            for gval in groups:
                v = sel[(sel[key] == gval) & (sel.method == m)]["T"].to_numpy(float)
                fin = v[np.isfinite(v)]
                means.append(fin.mean() if len(fin) == len(v) and len(v) else np.nan)
                stds.append(fin.std() if len(fin) > 1 else 0.0)
                nr.append(len(fin))
            series[m] = (means, stds, nr)
        _bars(ax, [f"{g:g}" for g in groups], series, hatches)
        ax.set_xlabel(xl); ax.set_ylabel("T gates (count)")
        ax.set_title(f"target spectral-norm error {eps:g}")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fault-tolerant cost including rotation synthesis (c = 3 T per bit of precision)", y=1.02)
    _save(fig, path)


def fig_ecdf(ang: pd.DataFrame, path: Path, sigmas: tuple[float, ...]) -> None:
    """Empirical CDF of the operator error under coherent angle noise (instances x draws)."""
    arms = list(ang.arm.unique())
    colors = [SAND, BROWN, OCHRE, TEAL]
    styles = ["-", "--", "-.", "-"]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.0 * len(sigmas), 3.6), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], sigmas):
        for a, c, ls in zip(arms, colors, styles):
            v = np.sort(ang[(ang.arm == a) & np.isclose(ang.sigma, s)].eps_spec.to_numpy())
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, ls=ls, lw=2, label=a)
        ax.set_xscale("log")
        ax.set_title(f"angle noise sigma = {s:g} rad")
        ax.set_xlabel("spectral-norm error (dimensionless)")
        ax.grid(alpha=0.25, which="both")
    axes[0][0].set_ylabel("empirical CDF (probability)")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Robustness to coherent rotation-angle errors", y=1.02)
    _save(fig, path)


def fig_ablation(ab: pd.DataFrame, path: Path) -> None:
    """Grouped hatched bars: error of each parameterisation at the same rotation budget."""
    arms = list(dict.fromkeys(ab.arm))
    g = ab.groupby("arm").eps_spec.agg(["mean", "std"]).loc[arms]
    npar = ab.groupby("arm").n_params.first().loc[arms]
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    colors = [SAND, TEAL, TERRA, OCHRE, OLIVE]
    hatches = ["", "xx", "//", "..", "\\\\"]
    for i, a in enumerate(arms):
        ax.bar(i, g.loc[a, "mean"], yerr=0 if np.isnan(g.loc[a, "std"]) else g.loc[a, "std"], color=colors[i % 5],
               edgecolor="k", hatch=hatches[i % 5], capsize=3)
        top = g.loc[a, "mean"] + (0 if np.isnan(g.loc[a, "std"]) else g.loc[a, "std"])
        ax.text(i, top * 1.12, f"{int(npar[a])} parameters", ha="center", fontsize=7)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a.replace(", ", ",\n") for a in arms], fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel("spectral-norm error (dimensionless)")
    ax.set_title("Ablation at fixed rotation budget (mean ± std over instances)")
    ax.grid(alpha=0.25, axis="y", which="both")
    _save(fig, path)


def fig_noise_sweeps(ang: pd.DataFrame, pauli: pd.DataFrame, path: Path) -> None:
    """Small multiples: mean error against coherent (left) and Pauli (right) noise strength."""
    arms = list(ang.arm.unique())
    colors = [SAND, BROWN, OCHRE, TEAL]
    marks = ["o", "s", "d", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.3})
    for a, c, mk in zip(arms, colors, marks):
        g = ang[ang.arm == a].groupby("sigma").eps_spec.agg(["mean", "std"]).reset_index()
        axes[0].errorbar(g.sigma.clip(lower=1e-4), g["mean"], yerr=g["std"], color=c, marker=mk, ms=4, capsize=2, lw=1.6,
                         label=f"{a} (R = {int(ang[ang.arm == a].R.iloc[0])})")
        h = pauli[pauli.arm == a].groupby("p").infid.agg(["mean", "std"]).reset_index()
        axes[1].errorbar(h.p.clip(lower=1e-4), h["mean"], yerr=h["std"], color=c, marker=mk, ms=4, capsize=2, lw=1.6)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[0].set_xlabel("angle noise sigma (rad; 0 drawn at 1e-4)")
    axes[0].set_ylabel("spectral-norm error (dimensionless)")
    axes[1].set_xlabel("Pauli error probability p (0 drawn at 1e-4)")
    axes[1].set_ylabel("process infidelity (dimensionless)")
    axes[0].set_title("coherent angle errors"); axes[1].set_title("incoherent Pauli errors")
    axes[0].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    _save(fig, path)
