"""Grid execution (paper §3.3-3.5): one parquet row per (config, task, N, r, arm).

A unit is one (task, N, r, arm). Within a unit, preprocessing for each
(variant, fold) is computed once and shared by every configuration that uses
it, so configurations are paired on identical imputed data.

    python -m bench.run --runner cpu    --out main --workers 20
    python -m bench.run --runner tabpfn --out main
    python -m bench.run --runner fus    --out main      (see bench.fusion)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, loaders, metrics, references
from .configs import BY_ID, CONFIGS
from .models import fit_predict, smote
from .paths import DERIVED, ROOT, RUNS
from .preprocess import TaskCtx, leaky_features, prepare
from .subsample import GRID, draw, eligible, permuted_labels, seed

ARMS = ("true", "perm")


def code_hash():
    h = hashlib.md5()
    for p in sorted((ROOT / "bench").glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


@lru_cache(maxsize=None)
def context(name):
    t = loaders.load(name)
    mp = DERIVED / f"reactome_{name}.npz"
    if mp.exists():
        M = np.load(mp)["M"]
    else:
        from .reference import membership_matrix
        M, ids = membership_matrix(t.uniprot)
        np.savez_compressed(mp, M=M, pathway_ids=ids)
    ctx = TaskCtx(X=t.X, has_seq=t.has_seq, M=M)
    return t, ctx, references.load(name)


def _sign_S(X, S, yS):
    """Sign of the S_r mean difference per protein (directional agreement, paper §3.6)."""
    import warnings
    a, b = X[S][yS == 1], X[S][yS == 0]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.nanmean(a, 0) - np.nanmean(b, 0)
    return np.sign(np.nan_to_num(d))


def run_unit(task_name, N, r, arm, config_ids):
    t, ctx, ref = context(task_name)
    S, H = draw(t, N, r)
    yS = t.y[S].astype(int)
    if arm == "perm":
        yS = permuted_labels(yS, task_name, N, r)
    yH = t.y[H].astype(int)
    coords = (task_name, N, r, arm)
    prim = cv.primary_folds(yS, N, coords)
    lpo = cv.lpo_folds(yS, N)
    sign_S = _sign_S(t.X, S, yS)

    cache = {}

    def prep(variant, kind, i, tr_pos, te, fixed=None, fixed_key=None):
        key = (variant, kind, i, fixed_key)
        if key not in cache:
            tr = S[tr_pos]
            cache[key] = prepare(ctx, tr, te, yS[tr_pos], variant, seed("impute", *coords, kind, i),
                                 fixed=fixed, tag=f"{kind}:{fixed_key or 'honest'}")
        return cache[key]

    leaky_feat = {}

    def fixed_for(variant):
        if variant not in leaky_feat:
            leaky_feat[variant] = leaky_features(ctx, S, yS, variant, seed("impute", *coords, "S"))
        return leaky_feat[variant]

    def cv_scores(cfg, folds, kind, fixed=None):
        out = []
        for i, (tr_pos, te_pos) in enumerate(folds):
            p = prep(cfg.variant, kind, i, tr_pos, S[te_pos], fixed=fixed,
                     fixed_key="leaky" if fixed is not None else None)
            s, _ = fit_predict(cfg.model, p.Ztr, yS[tr_pos], p.Zte, p.t_feat,
                               seed("model", cfg.id, *coords, kind, i, fixed is not None),
                               smote_in_fold=(cfg.smote == "infold"))
            out.append(s)
        return out

    smote_S = {}

    def leaky_smote_scores(cfg, folds, kind):
        """SMOTE on all of S_r before the split: preprocessing fitted on S_r, synthetic
        points from every original sample stay in every training fold (paper §3.4)."""
        if cfg.variant not in smote_S:
            p = prepare(ctx, S, S, yS, cfg.variant, seed("impute", *coords, "S"), tag="leaky-smote-on-S")
            Zaug, yaug = smote(p.Ztr, yS, seed("smote", *coords))
            smote_S[cfg.variant] = (p, Zaug, yaug)
        p, Zaug, yaug = smote_S[cfg.variant]
        syn_Z, syn_y = Zaug[N:], yaug[N:]
        out = []
        for i, (tr_pos, te_pos) in enumerate(folds):
            Ztr = np.vstack([p.Ztr[tr_pos], syn_Z]); ytr = np.concatenate([yS[tr_pos], syn_y])
            s, _ = fit_predict(cfg.model, Ztr, ytr, p.Ztr[te_pos], p.t_feat,
                               seed("model", cfg.id, *coords, kind, i))
            out.append(s)
        return out

    ext_cache = {}

    def external(cfg):
        key = (cfg.model, cfg.variant, cfg.smote is not None)
        if key not in ext_cache:
            p = prep(cfg.variant, "ext", 0, np.arange(N), H)
            s, imp = fit_predict(cfg.model, p.Ztr, yS, p.Zte, p.t_feat,
                                 seed("model", cfg.shares_external_with or cfg.id, *coords, "ext", 0),
                                 smote_in_fold=cfg.smote is not None)
            ext_cache[key] = (cv.external_auc(yH, s), p, imp)
        return ext_cache[key]

    rows = []
    for cid in config_ids:
        cfg = BY_ID[cid]
        t0 = time.perf_counter()
        if cfg.smote == "leaky":
            auc_int = cv.primary_auc(yS, N, prim, leaky_smote_scores(cfg, prim, "prim"))
            auc_lpo = cv.lpo_auc(yS, lpo, leaky_smote_scores(cfg, lpo, "lpo")) if lpo and cfg.lpo else np.nan
            auc_leaky = np.nan
        else:
            auc_int = cv.primary_auc(yS, N, prim, cv_scores(cfg, prim, "prim"))
            auc_lpo = cv.lpo_auc(yS, lpo, cv_scores(cfg, lpo, "lpo")) if lpo and cfg.lpo else np.nan
            auc_leaky = (cv.primary_auc(yS, N, prim, cv_scores(cfg, prim, "prim", fixed=fixed_for(cfg.variant)))
                         if cfg.leaky_filter else np.nan)
        auc_ext, p_ext, imp = external(cfg)
        row = dict(task=task_name, N=N, r=r, arm=arm, config_id=cid, config=cfg.name,
                   auc_int=auc_int, auc_int_lpo=auc_lpo, auc_int_leaky=auc_leaky, auc_ext=auc_ext,
                   n_S_pos=int((yS == 1).sum()), n_H_pos=int((yH == 1).sum()), n_H_neg=int((yH == 0).sum()),
                   n_feat=int(p_ext.Ztr.shape[1]), primary=cv.primary_kind(N), seconds=time.perf_counter() - t0)
        if cfg.protein_ranking and imp is not None:
            rank = metrics.full_ranking(p_ext, imp)
            row.update(metrics.recovery(rank, ref, sign_S))
            row["top100"] = rank[:100].astype(np.int32).tolist()
        rows.append(row)
    return rows


# ------------------------------------------------------------------ driver
def do_block(name, N, arm, b0, b1, f, ids):
    """One output file: repeats b0..b1-1 of one (task, N, arm). Module level so
    joblib workers can unpickle it."""
    rows = [row for r in range(b0, b1) for row in run_unit(name, N, r, arm, ids)]
    pd.DataFrame(rows).assign(code=code_hash()).to_parquet(f)
    return Path(f).name
def plan(tasks, R, Ns=GRID):
    cells = []
    for name in tasks:
        t, _, _ = context(name)
        for N in Ns:
            if eligible(t, N, R):
                cells.append((name, N))
    return cells


def _out(out, runner, name, N, arm, block):
    d = RUNS / out / runner
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}_N{N}_{arm}_r{block:03d}.parquet"


def run(runner, out, tasks, R, workers, configs=None, block=10, Ns=GRID, arms=ARMS, r0=0):
    ids = configs or [c.id for c in CONFIGS if c.runner == runner]
    cells = plan(tasks, R, Ns)
    meta = dict(runner=runner, configs=ids, R=R, r0=r0, cells=cells, arms=list(arms), code=code_hash(),
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (RUNS / out).mkdir(parents=True, exist_ok=True)
    (RUNS / out / f"plan_{runner}.json").write_text(json.dumps(meta, indent=2))
    print("cells:", cells, flush=True)
    jobs = []
    for name, N in cells:
        for arm in arms:
            for b in range(r0, r0 + R, block):
                f = _out(out, runner, name, N, arm, b)
                if not f.exists():
                    jobs.append((name, N, arm, b, min(b + block, r0 + R), f))
    print(f"{len(jobs)} blocks to run", flush=True)
    if runner == "tabpfn":
        from . import gpu
        print(f"GPU cap {gpu.guard('tabpfn'):.1f} GB", flush=True)
    if runner == "cpu":
        from joblib import Parallel, delayed
        for i, name_done in enumerate(Parallel(n_jobs=workers, return_as="generator_unordered", verbose=0)(
                delayed(do_block)(*j, ids) for j in jobs)):
            print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(jobs)} {name_done}", flush=True)
    else:
        for i, (name, N, arm, b0, b1, f) in enumerate(jobs):
            rows = [row for r in range(b0, b1) for row in run_unit(name, N, r, arm, ids)]
            pd.DataFrame(rows).assign(code=code_hash()).to_parquet(f)
            print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(jobs)} {f.name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", choices=("cpu", "tabpfn"), required=True)
    ap.add_argument("--out", default="main")
    ap.add_argument("--tasks", default=",".join(loaders.TASKS))
    ap.add_argument("--N", default=",".join(map(str, GRID)))
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--r0", type=int, default=0)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--arms", default="true,perm")
    ap.add_argument("--configs", default="")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    # Call through the imported module, not __main__, so joblib workers can unpickle do_block.
    from bench import run as _mod
    _mod.run(a.runner, a.out, a.tasks.split(","), a.R, a.workers,
        configs=[int(x) for x in a.configs.split(",")] if a.configs else None,
        block=a.block, Ns=tuple(int(x) for x in a.N.split(",")), arms=tuple(a.arms.split(",")), r0=a.r0)
