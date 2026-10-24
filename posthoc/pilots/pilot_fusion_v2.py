"""Pilot: does FusionV2 (embedding-informed linear fusion) beat the frozen fusion
network on held-out samples? Same draws, same in-fold reference preprocessing and
the same held-out sets as the benchmark (its elastic net is re-run as a check);
every hyperparameter (C = 1, k = 32) was fixed before running.

Exploratory and post hoc: not part of the paper. Run from anywhere:

    python pilot_fusion_v2.py <path to smalln-bench> 100 pilot.csv

Columns: Ridge = FusionV2(use_esm=False); EIL = FusionV2 (default);
EIL_esm_only = FusionV2(use_residual=False); *_loo = leave-one-out standardised
training rows; EIL_en = elastic-net penalty instead of ridge.
"""
import sys, time, warnings
import numpy as np
import pandas as pd

REPO = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from bench import loaders, cv                                   # noqa: E402
from bench.subsample import draw, seed, permuted_labels         # noqa: E402
from bench.preprocess import TaskCtx, prepare                   # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.exceptions import ConvergenceWarning               # noqa: E402

warnings.simplefilter("ignore", ConvergenceWarning)
K_PC = 32          # embedding principal components (fixed a priori)
C = 1.0            # same inverse penalty as the benchmark's elastic net
R = int(sys.argv[2]) if len(sys.argv) > 2 else 50


def esm_pcs(E, has_seq, k=K_PC):
    """Label-free, sample-free: PCA of the task's protein embeddings, PCs scaled to unit variance."""
    X = E[has_seq]
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    pcs = (E - mu) @ Vt[:k].T
    pcs /= pcs[has_seq].std(0)
    pcs[~has_seq] = 0.0
    return pcs


def loo_standardise(Z):
    """Each training row re-standardised with the other n-1 rows' mean and SD (from in-sample z)."""
    n = Z.shape[0]
    mu = -Z / (n - 1)
    var = (n - 1 - Z ** 2 * n / (n - 1)) / (n - 2)
    var = np.maximum(var, 0.05)
    return (Z - mu) / np.sqrt(var)


def logit_fit(X, y, l1_ratio=0.0, sd=0):
    if l1_ratio == 0.0:
        m = LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", max_iter=5000)
    else:
        m = LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=10000, random_state=sd % (2**32))
    return m.fit(X, y)


def run(task, Ns, arms=("true",)):
    t = loaders.load(task)
    ctx = TaskCtx(X=t.X, has_seq=t.has_seq)
    E = np.nan_to_num(np.load(f"{REPO}/data/derived/esm2_cls_{task}.npy"))
    PC = esm_pcs(E, t.has_seq)
    rows = []
    for N in Ns:
        for arm in arms:
            for r in range(R):
                S, H = draw(t, N, r)
                yS = t.y[S].astype(int)
                if arm == "perm":
                    yS = permuted_labels(yS, task, N, r)
                yH = t.y[H].astype(int)
                coords = (task, N, r, arm)
                sd_imp = seed("impute", *coords, "ext", 0)
                pe = prepare(ctx, S, H, yS, "ref", sd_imp)                          # EN input (as benchmark)
                pf = prepare(ctx, S, H, yS, "ref", sd_imp, restrict_seq=True)       # FUS input (as benchmark)
                out = dict(task=task, N=N, r=r, arm=arm)
                # benchmark elastic net, re-run (reproduction check)
                m = logit_fit(pe.Ztr, yS, 0.5, seed("model", 2, *coords, "ext", 0))
                out["EN_rerun"] = cv.external_auc(yH, m.decision_function(pe.Zte))
                Ztr, Zte = pf.Ztr, pf.Zte
                Pm = PC[pf.feat] / np.sqrt(Ztr.shape[1])
                Str, Ste = Ztr @ Pm, Zte @ Pm                                         # abundance-weighted embedding sums
                Zl = loo_standardise(Ztr)
                Sl = Zl @ Pm
                for name, Xtr, Xte, l1 in [
                    ("Ridge", Ztr, Zte, 0.0),
                    ("EIL", np.hstack([Ztr, Str]), np.hstack([Zte, Ste]), 0.0),
                    ("EIL_esm_only", Str, Ste, 0.0),
                    ("Ridge_loo", Zl, Zte, 0.0),
                    ("EIL_loo", np.hstack([Zl, Sl]), np.hstack([Zte, Ste]), 0.0),
                    ("EIL_en", np.hstack([Ztr, Str]), np.hstack([Zte, Ste]), 0.5),
                ]:
                    m = logit_fit(Xtr, yS, l1, seed("model", 2, *coords, "ext", 0))
                    out[name] = cv.external_auc(yH, m.decision_function(Xte))
                rows.append(out)
            print(f"{time.strftime('%H:%M:%S')} {task} N={N} {arm} done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    D = pd.concat([run("B2", [6, 12, 30]), run("B5", [6, 12, 30, 50]),
                   run("B2", [12], ("perm",)), run("B5", [12], ("perm",))], ignore_index=True)
    D.to_csv(sys.argv[3], index=False)
    print("seconds", round(time.time() - t0))
