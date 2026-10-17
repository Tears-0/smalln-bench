"""Selection stability (spec §6.3, F20) and the stability-recovery test (F21).

Kuncheva index of the top-50 proteins across all R(R-1)/2 pairs of repeats at
fixed (task, configuration, N, label arm). The total feature count n is the
number of proteins in the task passing the reference presence filter (>= 50%
observed) on the full data. Kuncheva is 0 for chance overlap and 1 for
identical lists.

Configurations: those with a protein ranking (F8: not TabPFN, not *-path).
Configs 17-18 share their external fit with 15-16 (identical rankings) and are
not repeated.

F21 (decided before looking): Spearman rho between mean Kuncheva and mean
precision@50 across (configuration, N) cells with N <= 30, true labels, computed
within each task. Stability is a usable proxy only if rho >= 0.5 in EVERY task;
otherwise reported as a negative with the per-task values.

    python -m bench.stability
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import loaders
from .configs import BY_ID
from .paths import RESULTS

K = 50
F21_MAX_N, F21_RHO = 30, 0.5


def n_total(task_name):
    t = loaders.load(task_name)
    return int(((~np.isnan(t.X)).mean(0) >= 0.5).sum())


def kuncheva_matrix(lists, n, k=K):
    """Mean Kuncheva index over all pairs, via an indicator matrix."""
    ids = np.unique(np.concatenate([np.asarray(l[:k]) for l in lists]))
    pos = {p: i for i, p in enumerate(ids)}
    B = np.zeros((len(lists), len(ids)), dtype=np.float32)
    for i, l in enumerate(lists):
        B[i, [pos[p] for p in l[:k]]] = 1
    r = B @ B.T
    iu = np.triu_indices(len(lists), 1)
    ki = (r[iu] * n - k * k) / (k * (n - k))
    return float(ki.mean()), float(np.percentile(ki, 25)), float(np.percentile(ki, 75))


def compute(df=None):
    df = df if df is not None else pd.read_parquet(RESULTS / "main_all_runs.parquet")
    keep = [c for c in df.config_id.unique() if BY_ID[c].protein_ranking and BY_ID[c].smote != "leaky"]
    d = df[df.config_id.isin(keep)].dropna(subset=["top100"])
    ntot = {t: n_total(t) for t in d.task.unique()}
    rows = []
    for (task, cfg, cid, N, arm), g in d.groupby(["task", "config", "config_id", "N", "arm"]):
        lists = [np.asarray(x) for x in g.top100]
        m, q1, q3 = kuncheva_matrix(lists, ntot[task])
        rows.append(dict(task=task, config=cfg, config_id=cid, N=N, arm=arm, kuncheva=m, k_q1=q1, k_q3=q3,
                         n_repeats=len(lists), n_total=ntot[task], p50_mean=float(g.p50.mean())))
    return pd.DataFrame(rows)


def f21(st):
    t = st[(st.arm == "true") & (st.N <= F21_MAX_N)]
    out = {}
    for task, g in t.groupby("task"):
        rho, _ = spearmanr(g.kuncheva, g.p50_mean)
        out[task] = dict(rho=float(rho), cells=len(g))
    usable = all(v["rho"] >= F21_RHO for v in out.values())
    return out, usable


def f21_fixed_n(st):
    """Diagnostic added 2026-10-15 (spec §16 A8), NOT part of the registered test.
    The registered rho pools cells across N, and both stability and recovery rise with
    N; a lab sits at one N. Rank stability and recovery within each N, then correlate."""
    t = st[(st.arm == "true") & (st.N <= F21_MAX_N)].copy()
    out = {}
    for task, g in t.groupby("task"):
        g = g.assign(rk=g.groupby("N").kuncheva.rank(), rp=g.groupby("N").p50_mean.rank())
        out[task] = dict(rho_fixed_N=float(spearmanr(g.rk, g.rp)[0]),
                         rho_N_recovery=float(spearmanr(g.N, g.p50_mean)[0]),
                         rho_N_stability=float(spearmanr(g.N, g.kuncheva)[0]),
                         per_N={int(N): float(spearmanr(h.kuncheva, h.p50_mean)[0]) for N, h in g.groupby("N")})
    return out


if __name__ == "__main__":
    st = compute()
    st.to_parquet(RESULTS / "stability.parquet")
    pd.set_option("display.width", 230)
    for arm in ("true", "perm"):
        print(f"\nMean Kuncheva index of top-{K}, {arm} labels (0 = chance overlap, 1 = identical lists)")
        print(st[st.arm == arm].pivot_table(index=["task", "config"], columns="N", values="kuncheva").round(2).to_string())
    res, usable = f21(st)
    print("\nF21 stability-recovery test (true labels, N <= 30, per task):",
          {k: {"rho": round(v["rho"], 3), "cells": v["cells"]} for k, v in res.items()})
    print("Registered F21 outcome -- stability is a usable proxy for recovery:", usable)
    fx = f21_fixed_n(st)
    print("Diagnostic (A8), rho at fixed N:", {k: round(v["rho_fixed_N"], 2) for k, v in fx.items()})
    import json
    (RESULTS / "stability_f21.json").write_text(json.dumps(dict(registered=res, registered_usable=usable,
                                                                fixed_n_diagnostic=fx), indent=2))
