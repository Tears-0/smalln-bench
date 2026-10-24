"""Leave-pair-out for the fusion arm (post hoc, prompted by review).

The main run gave FUS no leave-pair-out pass, for cost, so its positive pooled-LOOCV bias at
N <= 12 was never checked by the honest estimator. This runner adds, for the same draws,
labels and in-fold pipeline as config 5:

  - leave-pair-out AUC for every unit (N = 6, 9, 12), with per-pair test probabilities;
  - for the first R_DIAG repeats of each cell, a re-run of the pooled-LOOCV folds that keeps
    each held-out probability, its label and the training class counts, so the direction of
    the bias can be read off directly. The batched trainer draws its shuffles per batch, so
    these re-runs are fresh replicates of the stored primary fits, not bit-identical copies.

Seeds follow bench.run: preprocessing seed("impute", *coords, kind, i), model
seed("model", 5, *coords, kind, i).

    python -m bench.fusion_lpo --tasks B2,B5 --arms perm
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import cv, embeddings, gpu
from .fusion import CONFIG_ID, MAX_MODELS, train_batch
from .paths import RUNS
from .preprocess import prepare
from .run import code_hash, context
from .subsample import draw, permuted_labels, seed

NS = (6, 9, 12)
R_DIAG = 25
OUT = RUNS / "fus_lpo"


def _unit_fits(task_name, N, r, arm, diag):
    t, ctx, _ = context(task_name)
    S, H = draw(t, N, r)
    yS = t.y[S].astype(int)
    if arm == "perm":
        yS = permuted_labels(yS, task_name, N, r)
    coords = (task_name, N, r, arm)
    folds = {"lpo": cv.lpo_folds(yS, N)}
    if diag:
        folds["prim"] = cv.primary_folds(yS, N, coords)
    fits = []
    for kind, fl in folds.items():
        for i, (tr_pos, te_pos) in enumerate(fl):
            p = prepare(ctx, S[tr_pos], S[te_pos], yS[tr_pos], "ref", seed("impute", *coords, kind, i),
                        restrict_seq=True, tag=f"{kind}:honest")
            fits.append(dict(kind=kind, i=i, Ztr=p.Ztr.astype(np.float32), ytr=yS[tr_pos],
                             Zte=p.Zte.astype(np.float32), feat=p.feat,
                             seed=seed("model", CONFIG_ID, *coords, kind, i)))
    # how atypical each pooled-LOOCV held-out sample looks on its fold's input proteins:
    # mean |z| of the held-out sample, of the training class-1 samples, of the training class-0 samples
    zstat = [[float(np.abs(f["Zte"]).mean()), float(np.abs(f["Ztr"][f["ytr"] == 1]).mean()),
              float(np.abs(f["Ztr"][f["ytr"] == 0]).mean())] for f in fits if f["kind"] == "prim"]
    return fits, dict(yS=yS, folds=folds, zstat=zstat)


def run_cell(task_name, N, units, out_file):
    E_task = np.nan_to_num(embeddings.load(task_name))
    per_unit, groups = {}, {}
    for u in units:
        fits, meta = _unit_fits(task_name, N, *u, diag=u[0] < R_DIAG)
        per_unit[u] = meta
        for f in fits:
            groups.setdefault(f["Ztr"].shape[0], []).append((u, f))
    res = {}
    for n_train, items in sorted(groups.items()):
        for b in range(0, len(items), MAX_MODELS):
            chunk = items[b:b + MAX_MODELS]
            t0 = time.time()
            for (u, f), (prob, _) in zip(chunk, train_batch([f for _, f in chunk], E_task)):
                res[(u, f["kind"], f["i"])] = prob
            print(f"   [{time.strftime('%H:%M:%S')}] {task_name} N={N} n_train={n_train} "
                  f"models {b + 1}-{b + len(chunk)}/{len(items)} in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for u, meta in per_unit.items():
        r, arm = u
        yS, folds = meta["yS"], meta["folds"]
        lpo = [res[(u, "lpo", i)] for i in range(len(folds["lpo"]))]
        row = dict(task=task_name, N=N, r=r, arm=arm, config_id=CONFIG_ID, config="FUS",
                   auc_int_lpo=cv.lpo_auc(yS, folds["lpo"], lpo),
                   # per pair: (P(class 1) of the positive, P(class 1) of the negative)
                   lpo_pairs=[[float(s[np.flatnonzero(yS[te] == 1)[0]]), float(s[np.flatnonzero(yS[te] == 0)[0]])]
                              for (_, te), s in zip(folds["lpo"], lpo)])
        if "prim" in folds:
            prim = [res[(u, "prim", i)] for i in range(len(folds["prim"]))]
            row["auc_int_rerun"] = cv.primary_auc(yS, N, folds["prim"], prim)
            # per held-out sample: label, P(class 1), training class-1 count, training size
            row["prim_heldout"] = [[int(yS[j]), float(p_j), int(yS[tr].sum()), int(len(tr))]
                                   for (tr, te), s in zip(folds["prim"], prim) for j, p_j in zip(te, s)]
            row["prim_zstat"] = meta["zstat"]
        rows.append(row)
    pd.DataFrame(rows).assign(code=code_hash()).to_parquet(out_file)


def run(tasks, Ns, arms, R=100, block=25):
    print(f"GPU cap {gpu.guard('fusion'):.1f} GB", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"plan_{'_'.join(tasks)}_{'_'.join(arms)}.json").write_text(json.dumps(
        dict(tasks=tasks, N=list(Ns), arms=list(arms), R=R, block=block, R_DIAG=R_DIAG,
             code=code_hash(), started=time.strftime("%F %T")), indent=2))
    for arm in arms:
        for name in tasks:
            for N in Ns:
                for b in range(0, R, block):
                    f = OUT / f"{name}_N{N}_{arm}_r{b:03d}.parquet"
                    if f.exists():
                        continue
                    units = [(r, arm) for r in range(b, min(b + block, R))]
                    print(f"[{time.strftime('%H:%M:%S')}] cell {name} N={N} {arm} r{b}-{units[-1][0]}", flush=True)
                    run_cell(name, N, units, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="B2,B5")
    ap.add_argument("--N", default=",".join(map(str, NS)))
    ap.add_argument("--arms", default="perm,true")
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--block", type=int, default=25)
    a = ap.parse_args()
    run(a.tasks.split(","), tuple(int(x) for x in a.N.split(",")), tuple(a.arms.split(",")), a.R, a.block)
