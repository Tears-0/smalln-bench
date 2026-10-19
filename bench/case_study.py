"""Descriptive illustration on the motivating nine-sample study (spec §16 A11).

Added after the benchmark results were known; descriptive only, no claim rests on it.
The study has no held-out samples, so optimism cannot be measured. What can be shown
is how the two internal estimators behave on these real data.

For each binary contrast available in the study (3 vs 3, so N = 6, a benchmark grid
point), every one of the C(6,3) = 20 balanced labellings is analysed: the true
labelling and the 19 rearrangements. Each gets pooled LOOCV AUC and leave-pair-out
AUC from the same reference pipeline and the same five models as the benchmark
(the fusion network included, with leave-pair-out, since this is its home data).

    python -m bench.case_study      -> results/case_study/case_study_auc.csv
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
import torch

from . import cv, fusion, gpu
from .models import fit_predict
from .paths import CASE_STUDY, RESULTS
from .preprocess import TaskCtx, prepare
from .subsample import seed

OUT = RESULTS / "case_study"
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = {"Control": ["F61", "F62", "F63"], "8 h": ["F64", "F65", "F66"], "24 h": ["F67", "F68", "F69"]}
CONTRASTS = [("24 h", "Control"), ("8 h", "Control"), ("24 h", "8 h")]   # positive class first
CPU_MODELS = ["ccp", "en", "rf", "tabpfn"]


def labellings(n=6, k=3):
    for pos in itertools.combinations(range(n), k):
        y = np.zeros(n, dtype=int)
        y[list(pos)] = 1
        yield y


def overlap_with_truth(y, y_true):
    """Number of true positives among the labelling's positives (3 = true, 0 = fully swapped)."""
    return int(((y == 1) & (y_true == 1)).sum())


def run():
    gpu.guard("tabpfn")
    df = pd.read_csv(CASE_STUDY / "not_imputed.csv")
    E = torch.load(CASE_STUDY / "esm2_t12_35M_cls_embeddings.pt", weights_only=True).float().numpy()
    rows = []
    for pos, neg in CONTRASTS:
        t0 = time.time()
        cols = GROUPS[pos] + GROUPS[neg]
        X = df[cols].to_numpy(np.float32).T                      # [6 samples, 2841 proteins]
        ctx = TaskCtx(X=X, has_seq=np.ones(X.shape[1], bool), M=None)
        y_true = np.array([1, 1, 1, 0, 0, 0])
        fus_fits = []
        for li, y in enumerate(labellings()):
            meta = dict(contrast=f"{pos} vs {neg}", labelling=li, overlap=overlap_with_truth(y, y_true),
                        is_true=bool((y == y_true).all()))
            for kind, folds in (("loocv", cv.primary_folds(y, 6, ("case", pos, neg, li))), ("lpo", cv.lpo_folds(y, 6))):
                preps = [prepare(ctx, tr, te, y[tr], "ref", seed("case-impute", pos, neg, li, kind, i), tag=f"case:{kind}")
                         for i, (tr, te) in enumerate(folds)]
                for m in CPU_MODELS:
                    sc = [fit_predict(m, p.Ztr, y[tr], p.Zte, p.t_feat, seed("case-model", m, pos, neg, li, kind, i))[0]
                          for i, ((tr, te), p) in enumerate(zip(folds, preps))]
                    auc = cv.primary_auc(y, 6, folds, sc) if kind == "loocv" else cv.lpo_auc(y, folds, sc)
                    rows.append(dict(meta, model=m, estimator=kind, auc=auc))
                for i, ((tr, te), p) in enumerate(zip(folds, preps)):
                    fus_fits.append(dict(Ztr=p.Ztr.astype(np.float32), ytr=y[tr], Zte=p.Zte.astype(np.float32),
                                         feat=p.feat, seed=seed("case-model", "fus", pos, neg, li, kind, i),
                                         key=(li, kind, i), y=y, folds=folds, meta=meta))
        # fusion network: batch every fit of this contrast that shares a training size
        res = {}
        for n_train in sorted({f["Ztr"].shape[0] for f in fus_fits}):
            group = [f for f in fus_fits if f["Ztr"].shape[0] == n_train]
            for f, (prob, _) in zip(group, fusion.train_batch(group, E)):
                res[f["key"]] = prob
        for li, y in enumerate(labellings()):
            for kind in ("loocv", "lpo"):
                fs = [f for f in fus_fits if f["key"][0] == li and f["key"][1] == kind]
                folds = fs[0]["folds"]
                sc = [res[f["key"]] for f in fs]
                auc = cv.primary_auc(y, 6, folds, sc) if kind == "loocv" else cv.lpo_auc(y, folds, sc)
                rows.append(dict(fs[0]["meta"], model="fus", estimator=kind, auc=auc))
        print(f"{pos} vs {neg}: done in {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "case_study_auc.csv", index=False)
    return R


def summarise(R):
    out = []
    for (c, m, e), g in R.groupby(["contrast", "model", "estimator"]):
        true = float(g[g.is_true].auc.iloc[0])
        other = g[~g.is_true].auc
        out.append(dict(contrast=c, model=m, estimator=e, true_auc=true, others_median=other.median(),
                        others_min=other.min(), others_max=other.max(),
                        exact_p=float((g.auc >= true).mean())))
    S = pd.DataFrame(out)
    S.to_csv(OUT / "case_study_summary.csv", index=False)
    return S


if __name__ == "__main__":
    R = run()
    S = summarise(R)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))
