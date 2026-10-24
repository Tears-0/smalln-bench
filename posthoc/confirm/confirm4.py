"""Confirmation run 4 (plan: confirm4_plan.md): a transferred TP53 signature as prior direction.

    SMALLN_REPO=<smalln-bench> python confirm4.py <smalln-bench> [sig] [baselines] [transfer]
"""
import os, sys, time, warnings
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

REPO = os.environ.get("SMALLN_REPO", str(__import__("pathlib").Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO)
from bench import cv, loaders                                        # noqa: E402
from bench.loaders import NONSILENT                                  # noqa: E402
from bench.paths import DERIVED, RAW                                 # noqa: E402
from bench.preprocess import prepare                                 # noqa: E402
from bench.run import context                                        # noqa: E402
from bench.subsample import draw, seed                               # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from sklearn.metrics import roc_auc_score                            # noqa: E402

HERE = Path(__file__).parent
SIG = HERE / "sig4"
GRID = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
CELLS = [("B2", 6), ("B2", 12), ("B2", 30), ("B5", 6), ("B5", 12), ("B5", 30), ("B5", 50)]
R0, R = 400, 100
CPTAC_EXT = ["Brca", "Ccrcc", "Coad", "Gbm", "Hnscc", "Lscc", "Luad", "Ov", "Ucec"]


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def ridge(X, y):
    warnings.simplefilter("ignore")
    return LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=5000).fit(X, y)


def _z(X):
    obs = ~np.isnan(X)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1)
    Z = (X - mu) / np.where(sd > 0, sd, np.nan)
    Z[:, obs.mean(0) < 0.5] = np.nan
    return np.nan_to_num(Z)


# ------------------------------------------------------------------ external data and signatures
def external(task):
    """Returns X (samples x task proteins, z within group), y, group labels, notes."""
    t = loaders.load(task)
    Xs, ys, gs, notes = [], [], [], []
    if task == "B2":
        import cptac
        ids = pd.Index([s.split(".")[0] for s in t.protein_ids])
        for name in CPTAC_EXT:
            ds = getattr(cptac, name)()
            P = ds.get_proteomics("bcm" if name == "Coad" else "umich")
            P = P[~P.index.astype(str).str.endswith(".N")]
            sm = ds.get_somatic_mutation("harmonized")
            if sm.empty:
                sm = ds.get_somatic_mutation("washu")
            seqd = set(sm.index.astype(str))
            mut = set(sm.index[(sm["Gene"] == "TP53") & sm["Mutation"].astype(str).str.contains(NONSILENT)].astype(str))
            tid = P.index.astype(str)
            keep = np.array([s in seqd for s in tid])
            cols = pd.Index(P.columns.get_level_values(1).astype(str).str.split(".").str[0])
            uniq = ~cols.duplicated()
            Z = pd.DataFrame(_z(P.to_numpy(float)[keep][:, uniq]), columns=cols[uniq]).reindex(columns=ids)
            matched = int(Z.notna().any().sum())
            y = np.array([s in mut for s in tid[keep]], dtype=int)
            if matched == 0:
                notes.append(f"{name}: 0 task proteins matched by Ensembl protein ID -> excluded")
                continue
            Xs.append(Z.fillna(0).to_numpy()); ys.append(y); gs += [name] * len(y)
            notes.append(f"{name}: {len(y)} tumours ({int(y.sum())} mutant / {int((1 - y).sum())} wild-type), "
                         f"{matched} task proteins matched")
    else:
        m = pd.read_csv(RAW / "ProCan-DepMapSanger_protein_matrix_8498_averaged.txt", sep="\t", low_memory=False)
        m.index = m.pop("Project_Identifier").str.split(";").str[0]
        ml = pd.read_csv(RAW / "model_list_20261010.csv", low_memory=False).set_index("model_id")
        calls = pd.read_csv(DERIVED / "kras_tp53_calls.csv", dtype=str)
        seqd = set(pd.read_csv(DERIVED / "models_with_mutation_data.csv")["model_id"])
        mut = set(calls.loc[(calls.gene_symbol == "TP53") & (calls.cancer_driver == "t"), "model_id"])
        lines = [i for i in m.index if i in ml.index and i in seqd and isinstance(ml.loc[i, "tissue"], str)
                 and ml.loc[i, "tissue"] != "Haematopoietic and Lymphoid"]
        for tissue, g in ml.loc[lines].groupby("tissue"):
            ids_ = list(g.index)
            if len(ids_) < 3:
                notes.append(f"{tissue}: {len(ids_)} lines -> too few to standardise, excluded")
                continue
            y = np.array([i in mut for i in ids_], dtype=int)
            Xs.append(_z(m.loc[ids_].to_numpy(float))); ys.append(y); gs += [tissue] * len(y)
            notes.append(f"{tissue}: {len(y)} lines ({int(y.sum())} mutant / {int((1 - y).sum())} wild-type)")
    return np.vstack(Xs), np.concatenate(ys), np.array(gs), notes


def stage_sig():
    SIG.mkdir(exist_ok=True)
    qc = []
    for task in ("B2", "B5"):
        X, y, g, notes = external(task)
        for n in notes:
            log(f"{task} external {n}")
        beta = ridge(X, y).coef_[0]
        np.save(SIG / f"{task}_beta.npy", beta)
        log(f"{task}: signature from {len(y)} external samples ({int(y.sum())} mutant), {len(set(g))} groups")
        for grp in sorted(set(g)):
            te = g == grp
            if min(y[te].sum(), (1 - y[te]).sum()) < 10:
                continue
            auc = roc_auc_score(y[te], ridge(X[~te], y[~te]).decision_function(X[te]))
            qc.append(dict(task=task, held_out_group=grp, n=int(te.sum()), mutant=int(y[te].sum()), auc=auc))
            log(f"{task} leave-one-group-out: {grp} AUC {auc:.3f} (n = {int(te.sum())})")
    pd.DataFrame(qc).to_csv(HERE / "confirm4_signature_qc.csv", index=False)


# ------------------------------------------------------------------ task units
@lru_cache(maxsize=None)
def beta(task):
    return np.load(SIG / f"{task}_beta.npy")


def fit_score(Ztr, ytr, Zte, bsel, tau):
    nb = np.linalg.norm(bsel)
    if tau == 0.0 or nb == 0:
        Xtr, Xte = Ztr, Zte
    else:
        b = bsel / nb
        Xtr, Xte = np.column_stack([Ztr, tau * Ztr @ b]), np.column_stack([Zte, tau * Zte @ b])
    return ridge(Xtr, ytr).decision_function(Xte)


def unit(task, N, r):
    t, ctx, _ = context(task)
    S, H = draw(t, N, r)
    yS, yH = t.y[S].astype(int), t.y[H].astype(int)
    coords = (task, N, r, "true")
    B = beta(task)
    folds = cv.lpo_folds(yS, N) if N <= 12 else cv.primary_folds(yS, N, coords)
    inner = [prepare(ctx, S[tr], S[te], yS[tr], "ref", seed("impute", *coords, "inner4", i))
             for i, (tr, te) in enumerate(folds)]
    inner_auc = {}
    for tau in GRID:
        sc = [fit_score(p.Ztr, yS[tr], p.Zte, B[p.feat], tau) for (tr, te), p in zip(folds, inner)]
        inner_auc[tau] = cv.lpo_auc(yS, folds, sc) if N <= 12 else cv.primary_auc(yS, N, folds, sc)
    best = max(inner_auc.values())
    tau_star = min(tt for tt in GRID if inner_auc[tt] >= best - 1e-12)
    po = prepare(ctx, S, H, yS, "ref", seed("impute", *coords, "ext", 0))
    ext = {tt: cv.external_auc(yH, fit_score(po.Ztr, yS, po.Zte, B[po.feat], tt)) for tt in {tau_star, 0.0, 1.0}}
    # zero-shot: S_r standardisation only, no S_r labels
    XS = t.X[S].astype(float)
    keep = (~np.isnan(XS)).mean(0) >= 0.5
    mu, sd = np.nanmean(XS[:, keep], 0), np.nanstd(XS[:, keep], 0, ddof=1)
    zH = np.nan_to_num((t.X[H][:, keep] - mu) / np.where(sd > 0, sd, np.nan))
    zero = cv.external_auc(yH, zH @ B[keep])
    return dict(task=task, N=N, r=r, tau=tau_star, inner_star=inner_auc[tau_star], inner_0=inner_auc[0.0],
                transfer=ext[tau_star], ridge=ext[0.0], tau1=ext[1.0], zeroshot=zero)


def stage_transfer():
    from joblib import Parallel, delayed
    import confirm4 as _mod
    out = HERE / "confirm4_transfer.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for task, N in CELLS:
        if len(done) and ((done.task == task) & (done.N == N)).any():
            continue
        t0 = time.time()
        res = Parallel(n_jobs=16)(delayed(_mod.unit)(task, N, r) for r in range(R0, R0 + R))
        done = pd.concat([done, pd.DataFrame(res)], ignore_index=True); done.to_csv(out, index=False)
        log(f"transfer {task} N={N}: {time.time() - t0:.0f}s")


def stage_baselines():
    import confirm as c1                       # run 1's verified external-only baseline runner
    from bench import gpu
    gpu.guard("tabpfn")
    out = HERE / "confirm4_baselines.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for task, N in CELLS:
        if len(done) and ((done.task == task) & (done.N == N)).any():
            continue
        done = pd.concat([done, pd.DataFrame(c1.baselines(task, N, range(R0, R0 + R)))], ignore_index=True)
        done.to_csv(out, index=False)
        log(f"baselines {task} N={N} done")


if __name__ == "__main__":
    stages = sys.argv[2:] or ["sig", "baselines", "transfer"]
    for s in stages:
        {"sig": stage_sig, "baselines": stage_baselines, "transfer": stage_transfer}[s]()
    log("all stages done")
