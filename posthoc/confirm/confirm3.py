"""Confirmation run 3 (plan: confirm3_plan.md): embedding strength tau chosen by nested CV.

    python confirm3.py <smalln-bench>
"""
import os, sys, time, warnings
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

REPO = os.environ.get("SMALLN_REPO", str(__import__("pathlib").Path(__file__).resolve().parents[2]))           # env, not argv, so joblib workers can import this module
sys.path.insert(0, REPO)
from bench import cv                                                # noqa: E402
from bench.preprocess import prepare                                # noqa: E402
from bench.run import context                                       # noqa: E402
from bench.subsample import draw, seed                              # noqa: E402
from sklearn.linear_model import LogisticRegression                 # noqa: E402
from joblib import Parallel, delayed                                # noqa: E402

HERE = Path(__file__).parent
EMB = HERE / "emb2"
GRID = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LABELS = ("esmpc32", "esmpc32_shuf", "coexp32", "coexp32_shuf")
CELLS = [("B2", 6), ("B2", 12), ("B2", 30), ("B5", 6), ("B5", 12), ("B5", 30), ("B5", 50)]
R0, R = 300, 100


@lru_cache(maxsize=None)
def emb(task, label):
    return np.load(EMB / f"{task}_{label}.npy")


def score(Ztr, ytr, Zte, Asel, tau):
    warnings.simplefilter("ignore")
    if tau == 0.0:
        Xtr, Xte = Ztr, Zte
    else:
        Pm = Asel / np.sqrt(Ztr.shape[1])
        Xtr, Xte = np.hstack([Ztr, tau * Ztr @ Pm]), np.hstack([Zte, tau * Zte @ Pm])
    m = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=5000).fit(Xtr, ytr)
    return m.decision_function(Xte)


def unit(task, N, r):
    t, ctx, _ = context(task)
    S, H = draw(t, N, r)
    yS, yH = t.y[S].astype(int), t.y[H].astype(int)
    coords = (task, N, r, "true")
    folds = cv.lpo_folds(yS, N) if N <= 12 else cv.primary_folds(yS, N, coords)
    inner = [prepare(ctx, S[tr], S[te], yS[tr], "ref", seed("impute", *coords, "inner", i), restrict_seq=True)
             for i, (tr, te) in enumerate(folds)]                       # preprocessing refitted per inner fold
    po = prepare(ctx, S, H, yS, "ref", seed("impute", *coords, "ext", 0), restrict_seq=True)
    rows = []
    for label in LABELS:
        A = emb(task, label)
        inner_auc = {}
        for tau in GRID:
            sc = [score(p.Ztr, yS[tr], p.Zte, A[p.feat], tau) for (tr, te), p in zip(folds, inner)]
            inner_auc[tau] = cv.lpo_auc(yS, folds, sc) if N <= 12 else cv.primary_auc(yS, N, folds, sc)
        best = max(inner_auc.values())
        tau_star = min(tt for tt in GRID if inner_auc[tt] >= best - 1e-12)   # ties -> smallest tau
        ext = {tt: cv.external_auc(yH, score(po.Ztr, yS, po.Zte, A[po.feat], tt)) for tt in {tau_star, 0.0, 1.0}}
        rows.append(dict(task=task, N=N, r=r, emb=label, tau=tau_star, inner_star=inner_auc[tau_star],
                         inner_0=inner_auc[0.0], ext_star=ext[tau_star], ext_0=ext[0.0], ext_1=ext[1.0]))
    return rows


if __name__ == "__main__":
    import confirm3 as _mod                # workers unpickle functions from the importable module
    out = HERE / "confirm3.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for task, N in CELLS:
        if len(done) and ((done.task == task) & (done.N == N)).any():
            continue
        t0 = time.time()
        res = Parallel(n_jobs=16)(delayed(_mod.unit)(task, N, r) for r in range(R0, R0 + R))
        done = pd.concat([done, pd.DataFrame([x for rs in res for x in rs])], ignore_index=True)
        done.to_csv(out, index=False)
        print(f"{time.strftime('%F %T')} {task} N={N}: {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)
