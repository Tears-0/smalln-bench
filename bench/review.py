"""Analyses added after the results were known, in response to review (spec §16 A13).

None of these is registered; all are descriptive. Outputs go to results/analysis/review_*.csv.

    python -m bench.review effect_null      # median |d| of the top 100 under permuted labels
    python -m bench.review smote            # leaky-SMOTE optimism split into filter and SMOTE shares
    python -m bench.review kendall          # Kendall's W over B2 and B5, with and without FUS
    python -m bench.review kfold_tail       # P(optimism > 0.1) under k-fold, per config and pooled
    python -m bench.review recovery_h       # recovery against a reference computed on H_r only
    python -m bench.review missingness      # per-sample missing fraction as the only score
    python -m bench.review fus_lpo          # leave-pair-out for the fusion arm (bench.fusion_lpo)
    python -m bench.review all
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import loaders
from .analysis import OUT, SECONDARY, kendall_w, load
from .limma import moderated_t
from .metrics import precision_at
from .paths import CASE_STUDY, RUNS
from .subsample import GRID, R_MAIN, draw, permuted_labels, seed

BASELINES = ["CCP", "EN", "RF", "TabPFN"]
N_PERM_FULL = 20


def _cohen_d(X, y):
    a, b = X[y == 1], X[y == 0]
    na, nb = np.sum(~np.isnan(a), 0), np.sum(~np.isnan(b), 0)
    with np.errstate(all="ignore"):
        ma, mb = np.nanmean(a, 0), np.nanmean(b, 0)
        va, vb = np.nanvar(a, 0, ddof=1), np.nanvar(b, 0, ddof=1)
        sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        d = (ma - mb) / sp
    d[(na < 2) | (nb < 2)] = np.nan
    return d


def _med_d_top100(X, y, t):
    """The task-selection statistic exactly as bench.references computes it: median |d|
    over the top 100 proteins by |t| (NaN t last), ignoring proteins whose d is undefined."""
    d = _cohen_d(X, y)
    idx = np.argsort(-np.where(np.isfinite(t), np.abs(t), -np.inf), kind="stable")[:100]
    return float(np.nanmedian(np.abs(d[idx])))


# ---------------------------------------------------------------- effect-size null
def effect_null():
    """Full cohort: true labels vs N_PERM_FULL label permutations, both ranked by R limma
    exactly as the task table. Subsamples: what a laboratory at N would compute on its own
    data (numpy moderated t), true vs permuted labels, 100 draws."""
    from .references import _limma_r
    rows = []
    for name in ("B1", "B2", "B5"):
        T = loaders.load(name)
        X, y = T.X.astype(float), T.y.astype(int)
        rows.append(dict(task=name, scope="full", N=len(y), arm="true", median_abs_d=_med_d_top100(X, y, _limma_r(X, y)[0])))
        rng = np.random.default_rng(seed("effect-null", name))
        v = []
        for _ in range(N_PERM_FULL):
            yp = rng.permutation(y)
            v.append(_med_d_top100(X, yp, _limma_r(X, yp)[0]))
        rows.append(dict(task=name, scope="full", N=len(y), arm="perm", median_abs_d=float(np.median(v)),
                         q05=float(np.quantile(v, 0.05)), q95=float(np.quantile(v, 0.95))))
        for N in GRID:
            for arm in ("true", "perm"):
                v = []
                for r in range(R_MAIN):
                    try:
                        S, _ = draw(T, N, r)
                    except ValueError:
                        break
                    yS = y[S] if arm == "true" else permuted_labels(y[S], name, N, r)
                    v.append(_med_d_top100(X[S], yS, moderated_t(X[S], yS)))
                if v:
                    rows.append(dict(task=name, scope="subsample", N=N, arm=arm, median_abs_d=float(np.median(v)),
                                     q05=float(np.quantile(v, 0.05)), q95=float(np.quantile(v, 0.95))))
        print(f"effect_null {name} done", flush=True)
    E = pd.DataFrame(rows)
    E.to_csv(OUT / "review_effect_null.csv", index=False)
    return E


# ---------------------------------------------------------------- leaky SMOTE split
def smote():
    """Configs 17/18 fit preprocessing and SMOTE on all of S_r. Configs 15/16 run in-fold
    SMOTE, and their leaky pass refits only the supervised filter on S_r. On the same draws,
    auc_int(17/18) - auc_int_leaky(15/16) is the share that SMOTE (with imputation and
    z-scoring on S_r) adds on top of the filter leak."""
    D = load(None)
    rows = []
    for (task, arm), g in D.groupby(["task", "arm"]):
        W = g.pivot_table(index=["N", "r"], columns="config", values=["auc_int", "auc_int_leaky", "auc_ext"])
        for model in ("EN", "RF"):
            leaky, infold = f"{model}-smote-leaky", f"{model}-smote"
            for N, w in W.groupby(level="N"):
                total = w[("auc_int", leaky)] - w[("auc_ext", leaky)]
                filt = w[("auc_int_leaky", infold)] - w[("auc_ext", infold)]
                extra = w[("auc_int", leaky)] - w[("auc_int_leaky", infold)]
                rows.append(dict(task=task, arm=arm, model=model, N=N, total_optimism=total.median(),
                                 filter_leak_optimism=filt.median(), smote_extra=extra.median(),
                                 smote_extra_q25=extra.quantile(0.25), smote_extra_q75=extra.quantile(0.75),
                                 p_smote_extra_pos=float((extra > 0).mean()),
                                 auc_int_leaky_smote=w[("auc_int", leaky)].median()))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "review_smote_split.csv", index=False)
    return S


# ---------------------------------------------------------------- Kendall's W
def kendall():
    T = load("true")
    T = T[T.config.isin(BASELINES + ["FUS"]) & T.task.isin(SECONDARY)]
    W = T.pivot_table(index=["task", "N", "r"], columns="config", values="auc_ext")
    rows = []
    for N in sorted(W.index.get_level_values("N").unique()):
        for label, cols in (("with FUS", BASELINES + ["FUS"]), ("without FUS", BASELINES)):
            mr = W[cols].rank(axis=1, ascending=False, method="average").groupby(level=["task", "N"]).mean()
            tasks = [t for t in SECONDARY if (t, N) in mr.index]
            if len(tasks) < 2:
                continue
            M = mr.loc[[(t, N) for t in tasks]]
            rows.append(dict(N=N, set=label, W=kendall_w(M.rank(axis=1).to_numpy()),
                             **{f"order_{t}": " > ".join(M.loc[(t, N)].sort_values().index) for t in tasks}))
    K = pd.DataFrame(rows)
    K.to_csv(OUT / "review_kendall.csv", index=False)
    return K


# ---------------------------------------------------------------- k-fold tail
def kfold_tail():
    """Empirical P(AUC_int - AUC_ext > 0.1) under the k-fold primary estimator, true labels."""
    T = load("true")
    T = T[(T.primary != "loocv") & T.config.isin(BASELINES) & T.task.isin(SECONDARY)].assign(
        O=lambda d: d.auc_int - d.auc_ext)
    per = T.groupby(["task", "N", "config"]).O.apply(lambda o: (o > 0.1).mean()).rename("p_gt_0.1").reset_index()
    pooled = T.groupby(["task", "N"]).O.agg(p_gt_01=lambda o: (o > 0.1).mean(), median="median",
                                           q25=lambda o: o.quantile(0.25), q75=lambda o: o.quantile(0.75)).reset_index()
    per.to_csv(OUT / "review_kfold_tail_per_config.csv", index=False)
    pooled.to_csv(OUT / "review_kfold_tail_pooled.csv", index=False)
    return per, pooled


# ---------------------------------------------------------------- recovery vs H_r reference
def recovery_h():
    """Precision@50 against a moderated-t ranking computed on H_r alone (true labels),
    so the reference shares no sample with S_r. Sensitivity check for the full-cohort
    reference, which S_r overlaps by up to 31%."""
    D = load(None).dropna(subset=["p50"])
    D = D[D.top100.apply(lambda v: v is not None and len(v) >= 50)]
    rows = []
    for name in sorted(D.task.unique()):
        T = loaders.load(name)
        X, y = T.X.astype(float), T.y.astype(int)
        g_task = D[D.task == name]
        for (N, r), g in g_task.groupby(["N", "r"]):
            _, H = draw(T, N, r)
            t = moderated_t(X[H], y[H])
            ref = np.argsort(-np.where(np.isfinite(t), np.abs(t), -np.inf), kind="stable")
            for row in g.itertuples(index=False):
                rows.append(dict(task=name, N=N, r=r, arm=row.arm, config=row.config,
                                 p50_full=row.p50, p50_H=precision_at(np.asarray(row.top100), ref, 50)))
        print(f"recovery_h {name} done", flush=True)
    R = pd.DataFrame(rows)
    S = R.groupby(["task", "config", "N", "arm"])[["p50_full", "p50_H"]].mean().reset_index()
    S.to_csv(OUT / "review_recovery_h.csv", index=False)
    return S


# ---------------------------------------------------------------- missingness only
def missingness():
    """AUC of the per-sample fraction of missing values, with no model and no fitting."""
    rows = []
    for name in ("B1", "B2", "B5"):
        T = loaders.load(name)
        mf = np.isnan(T.X).mean(1)
        rows.append(dict(scope="benchmark", contrast=name, pos_mean=mf[T.y == 1].mean(), neg_mean=mf[T.y == 0].mean(),
                         auc=roc_auc_score(T.y, mf)))
    df = pd.read_csv(CASE_STUDY / "not_imputed.csv")
    G = {"Control": ["F61", "F62", "F63"], "8 h": ["F64", "F65", "F66"], "24 h": ["F67", "F68", "F69"]}
    mf = {k: [float(df[c].isna().mean()) for c in v] for k, v in G.items()}
    for pos, neg in (("24 h", "Control"), ("8 h", "Control"), ("24 h", "8 h")):
        s = np.array(mf[pos] + mf[neg]); y = np.r_[np.ones(3), np.zeros(3)]
        rows.append(dict(scope="case study", contrast=f"{pos} vs {neg}", pos_mean=np.mean(mf[pos]),
                         neg_mean=np.mean(mf[neg]), auc=roc_auc_score(y, s),
                         pos_values=mf[pos], neg_values=mf[neg]))
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "review_missingness.csv", index=False)
    return M


# ---------------------------------------------------------------- FUS leave-pair-out
def fus_lpo():
    files = sorted((RUNS / "fus_lpo").glob("*_N*_*_r*.parquet"))
    L = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    M = load(None)
    M = M[M.config == "FUS"][["task", "N", "r", "arm", "auc_int", "auc_ext"]]
    L = L.merge(M, on=["task", "N", "r", "arm"], how="left", validate="one_to_one")
    L["O_lpo"] = L.auc_int_lpo - L.auc_ext
    L["O_loocv"] = L.auc_int - L.auc_ext
    S = L.groupby(["task", "N", "arm"]).agg(
        n=("r", "size"), lpo_median=("auc_int_lpo", "median"), lpo_mean=("auc_int_lpo", "mean"),
        O_lpo_median=("O_lpo", "median"), O_lpo_q25=("O_lpo", lambda v: v.quantile(0.25)),
        O_lpo_q75=("O_lpo", lambda v: v.quantile(0.75)), loocv_median=("auc_int", "median"),
        O_loocv_median=("O_loocv", "median"), rerun_median=("auc_int_rerun", "median")).reset_index()
    S.to_csv(OUT / "review_fus_lpo.csv", index=False)

    # direction of the pooled-LOOCV bias: is the held-out sample assigned to the class that
    # is the minority in its training fold?
    H = []
    for row in L.dropna(subset=["auc_int_rerun"]).itertuples(index=False):
        for (lab, p1, n1_tr, n_tr), (z_te, z1, z0) in zip(row.prim_heldout, row.prim_zstat):
            minority = 1 if n1_tr < n_tr - n1_tr else (0 if n1_tr > n_tr - n1_tr else -1)
            H.append(dict(task=row.task, N=row.N, arm=row.arm, label=int(lab), p1=p1, minority=minority,
                          p_minority=p1 if minority == 1 else 1 - p1,
                          z_heldout=z_te, z_train_minority=z1 if minority == 1 else z0,
                          z_train_majority=z0 if minority == 1 else z1))
    H = pd.DataFrame(H)
    H = H[H.minority >= 0]
    D = H.groupby(["task", "N", "arm"]).agg(
        n=("p1", "size"), p_minority_mean=("p_minority", "mean"),
        frac_assigned_minority=("p_minority", lambda v: (v > 0.5).mean()),
        frac_confident=("p1", lambda v: ((v < 0.05) | (v > 0.95)).mean()),
        z_heldout=("z_heldout", "median"), z_train_minority=("z_train_minority", "median"),
        z_train_majority=("z_train_majority", "median")).reset_index()
    D.to_csv(OUT / "review_fus_direction.csv", index=False)
    return S, D


if __name__ == "__main__":
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = dict(effect_null=effect_null, smote=smote, kendall=kendall, kfold_tail=kfold_tail,
                recovery_h=recovery_h, missingness=missingness, fus_lpo=fus_lpo)
    for name, fn in jobs.items():
        if what in (name, "all") and not (what == "all" and name == "fus_lpo"):
            res = fn()
            print(f"\n=== {name}")
            for r in (res if isinstance(res, tuple) else (res,)):
                print(r.round(3).to_string(index=False))
