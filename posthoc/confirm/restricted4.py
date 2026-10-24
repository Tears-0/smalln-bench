"""Run 4, post hoc: how much does restricting the signature to the 500 kept proteins weaken it on its own?

Run 4's transfer-tau model (confirm4.py) used the external signature only on the 500 proteins the in-fold
filter kept on S_r. This scores held-out samples with that restricted signature alone (no S_r labels in the
score; the reference pipeline's in-fold imputation and z-scoring), next to the zero-shot score on all
proteins, on run 4's own draws (repeats 400-499, N = 12 and 30).

    SMALLN_REPO=<smalln-bench> python restricted4.py
"""
import sys
import numpy as np
import pandas as pd

import confirm4 as c4                                    # sets up the repository path
from bench import cv                                     # noqa: E402
from bench.preprocess import prepare                     # noqa: E402
from bench.run import context                            # noqa: E402
from bench.subsample import draw, seed                   # noqa: E402

rows = []
for task in ("B2", "B5"):
    t, ctx, _ = context(task)
    B = c4.beta(task)
    for N in (12, 30):
        for r in range(400, 500):
            S, H = draw(t, N, r)
            yS, yH = t.y[S].astype(int), t.y[H].astype(int)
            po = prepare(ctx, S, H, yS, "ref", seed("impute", task, N, r, "true", "ext", 0))
            XS = t.X[S].astype(float)
            keep = (~np.isnan(XS)).mean(0) >= 0.5
            mu, sd = np.nanmean(XS[:, keep], 0), np.nanstd(XS[:, keep], 0, ddof=1)
            zH = np.nan_to_num((t.X[H][:, keep] - mu) / np.where(sd > 0, sd, np.nan))
            rows.append(dict(task=task, N=N, r=r, zeroshot_all=cv.external_auc(yH, zH @ B[keep]),
                             restricted_500=cv.external_auc(yH, po.Zte @ B[po.feat])))
D = pd.DataFrame(rows)
T = pd.read_csv(c4.HERE / "confirm4_transfer.csv")
D = D.merge(T[["task", "N", "r", "zeroshot", "ridge", "transfer"]], on=["task", "N", "r"], validate="one_to_one")
d = float((D.zeroshot_all - D.zeroshot).abs().max())
if d > 1e-12:
    sys.exit(f"zero-shot does not reproduce run 4 (max |diff| {d:.3g})")
D.drop(columns="zeroshot").to_csv(c4.HERE / "confirm4_restricted.csv", index=False)
print("zero-shot reproduces run 4 exactly")
print(D.groupby(["task", "N"])[["zeroshot_all", "restricted_500", "ridge", "transfer"]].median().round(3).to_string())
