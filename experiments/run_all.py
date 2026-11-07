"""Reproduce every result and figure of the study.

    python experiments/run_all.py            # full study (about 5 minutes on 4 CPU workers)
    python experiments/run_all.py --quick    # smoke run (< 1 minute), written to a temporary folder
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from learned_trotter import plots, study  # noqa: E402

FULL = dict(
    acc=[(4, 1.0, 5, (3, 4, 5, 6)), (6, 0.5, 3, (3, 4, 5, 6)), (6, 1.0, 5, (3, 4, 5, 6)),
         (6, 1.5, 3, (3, 4, 5, 6)), (8, 1.0, 3, (5, 6))],
    main=(6, 1.0, 5), n_seeds_main=5, max_iter=80, workers=4,
    sigmas=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05), draws=20, pauli_ps=(0.0, 1e-3, 3e-3, 1e-2), traj=40,
    ecdf_sigmas=(0.005, 0.02), ms=(4, 6, 8, 10, 12, 16, 20), n_states=8, tc_eps=0.1,
)
QUICK = dict(
    acc=[(4, 0.5, 2, (3, 4)), (4, 1.0, 2, (3, 4))],
    main=(4, 1.0, 4), n_seeds_main=2, max_iter=30, workers=2,
    sigmas=(0.0, 0.01, 0.05), draws=4, pauli_ps=(0.0, 1e-2), traj=6,
    ecdf_sigmas=(0.01, 0.05), ms=(4, 8, 12), n_states=3, tc_eps=0.1,
)


def run_pool(ex, fn, arg_list):
    futs = [ex.submit(fn, *a) for a in arg_list]
    return [f.result() for f in futs]


def summarise(acc: pd.DataFrame, tc: pd.DataFrame, ab: pd.DataFrame, ang: pd.DataFrame,
              pauli: pd.DataFrame, kry: pd.DataFrame) -> dict:
    out: dict = {}
    mean = acc.groupby(["n", "t", "method", "K", "R"]).eps_spec.mean().reset_index()
    rows = []
    for (n, t), g in mean.groupby(["n", "t"]):
        for k in sorted(g[g.method == "learned"].K.unique()):
            lr = g[(g.method == "learned") & (g.K == k)].iloc[0]
            r_s = study.rotations_to_reach(g, "strang", float(lr.eps_spec))
            rows.append(dict(n=int(n), t=float(t), K=int(k), R_learned=int(lr.R), eps_learned=float(lr.eps_spec),
                             R_strang_same_error=r_s, rotation_ratio=r_s / lr.R if np.isfinite(r_s) else None))
    out["rotation_ratio_at_equal_error"] = rows
    summ = []
    for (n, t, e, m), g in tc.groupby(["n", "t", "eps_target", "method"]):
        fin = g["T"][np.isfinite(g["T"])]
        summ.append(dict(n=int(n), t=float(t), eps_target=float(e), method=m, n_instances=len(g), n_reached=len(fin),
                         T_mean=float(fin.mean()) if len(fin) else None,
                         T_std=float(fin.std()) if len(fin) > 1 else None))
    out["t_count_summary"] = summ
    out["ablation"] = ab.groupby("arm", sort=False).agg(eps_mean=("eps_spec", "mean"), eps_std=("eps_spec", "std"),
                                                       n_params=("n_params", "first"), R=("R", "first"),
                                                       fit_seconds=("fit_seconds", "mean")).reset_index().to_dict("records")
    out["fit_seconds_by_n"] = acc[acc.method == "learned"].groupby("n").fit_seconds.agg(["mean", "max"]).reset_index().to_dict("records")
    out["angle_noise"] = ang.groupby(["arm", "sigma"]).eps_spec.agg(["mean", "std"]).reset_index().to_dict("records")
    out["pauli_noise"] = pauli.groupby(["arm", "p"]).infid.agg(["mean", "std"]).reset_index().to_dict("records")
    out["state_error_reference"] = kry.groupby(["method", "budget", "budget_unit"]).state_error.agg(["mean", "std"]).reset_index().to_dict("records")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--plots-only", action="store_true", help="redraw figures from the CSVs in results/")
    args = ap.parse_args()
    cfg = QUICK if args.quick else FULL
    out = Path(args.out_dir) if args.out_dir else (Path(tempfile.mkdtemp(prefix="lt_quick_")) if args.quick else ROOT)
    res, fig = out / "results", out / "figures"
    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        n, _, _ = cfg["main"]
        rd = pd.read_csv
        acc, tc, ab = rd(res / "accuracy_vs_rotations.csv"), rd(res / "t_count.csv"), rd(res / "ablation.csv")
        ang, pauli = rd(res / "noise_angle.csv"), rd(res / "noise_pauli.csv")
        he, ht = study.headline_tables(acc, tc)
        he.to_csv(res / "headline_error_table.csv", index=False)
        ht.to_csv(res / "headline_t_count_table.csv", index=False)
        plots.fig_accuracy(acc, fig / "fig1_error_vs_rotations.png", n=n)
        plots.fig_tcount(tc, fig / "fig2_t_count.png", eps=cfg["tc_eps"], n_ref=n)
        plots.fig_ecdf(ang, fig / "fig3_angle_noise_ecdf.png", cfg["ecdf_sigmas"])
        plots.fig_ablation(ab, fig / "fig4_ablation.png")
        plots.fig_noise_sweeps(ang, pauli, fig / "fig5_noise_sweeps.png")
        return
    t0 = time.perf_counter()
    w, mi = cfg["workers"], cfg["max_iter"]
    print(f"output: {out}", flush=True)
    ex = ProcessPoolExecutor(max_workers=w)

    # 1. accuracy vs rotation budget
    jobs = [(n, t, s, ks, mi) for (n, t, ns, ks) in cfg["acc"] for s in range(ns)]
    parts = run_pool(ex, study.accuracy_task, jobs)
    acc = pd.DataFrame([r for p in parts for r in p["rows"]])
    store = {k: v for p in parts for k, v in p["params"].items()}
    acc.to_csv(res / "accuracy_vs_rotations.csv", index=False)
    (res / "learned_parameters.json").write_text(json.dumps(store))
    print(f"[1/5] accuracy done  {time.perf_counter() - t0:.0f}s", flush=True)

    tc = study.t_cost_table(acc)
    tc.to_csv(res / "t_count.csv", index=False)

    n, t, k = cfg["main"]
    seeds = range(cfg["n_seeds_main"])
    # 2. ablation
    ab = pd.DataFrame([r for rows in run_pool(ex, study.ablation_task, [(n, t, k, s, mi) for s in seeds]) for r in rows])
    ab.to_csv(res / "ablation.csv", index=False)
    print(f"[2/5] ablation done  {time.perf_counter() - t0:.0f}s", flush=True)

    # 3. noise sweeps on the trained circuits
    nz = run_pool(ex, study.noise_task, [(n, t, k, s, store[f"{n}|{t}|{s}|{k}"], cfg["sigmas"], cfg["draws"],
                                     cfg["pauli_ps"], cfg["traj"]) for s in seeds])
    ang = pd.DataFrame([r for p in nz for r in p["angle"]])
    pauli = pd.DataFrame([r for p in nz for r in p["pauli"]])
    ang.to_csv(res / "noise_angle.csv", index=False)
    pauli.to_csv(res / "noise_pauli.csv", index=False)
    print(f"[3/5] noise done  {time.perf_counter() - t0:.0f}s", flush=True)

    # 4. classical Krylov reference
    kry = pd.DataFrame([r for rows in run_pool(ex, study.krylov_task, [(n, t, k, s, store[f"{n}|{t}|{s}|{k}"], cfg["ms"],
                                                                    cfg["n_states"]) for s in seeds]) for r in rows])
    kry.to_csv(res / "classical_krylov_reference.csv", index=False)
    print(f"[4/5] krylov done  {time.perf_counter() - t0:.0f}s", flush=True)

    ex.shutdown()

    # 5. summary + figures
    summary = summarise(acc, tc, ab, ang, pauli, kry)
    summary["config"] = {k_: (list(v) if isinstance(v, tuple) else v) for k_, v in cfg.items() if k_ != "acc"}
    summary["config"]["acc"] = [list(map(lambda x: list(x) if isinstance(x, tuple) else x, a)) for a in cfg["acc"]]
    summary["hypothesis_checks"] = study.hypothesis_checks(acc, tc, ab, ang, pauli, n, t, k, cfg["tc_eps"])
    summary["runtime_seconds"] = time.perf_counter() - t0
    (res / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    he, ht = study.headline_tables(acc, tc)
    he.to_csv(res / "headline_error_table.csv", index=False)
    ht.to_csv(res / "headline_t_count_table.csv", index=False)
    plots.fig_accuracy(acc, fig / "fig1_error_vs_rotations.png", n=n)
    plots.fig_tcount(tc, fig / "fig2_t_count.png", eps=cfg["tc_eps"], n_ref=n)
    plots.fig_ecdf(ang, fig / "fig3_angle_noise_ecdf.png", cfg["ecdf_sigmas"])
    plots.fig_ablation(ab, fig / "fig4_ablation.png")
    plots.fig_noise_sweeps(ang, pauli, fig / "fig5_noise_sweeps.png")
    print(f"[5/5] done in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
