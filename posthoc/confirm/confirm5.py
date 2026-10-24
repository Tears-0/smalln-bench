"""Confirmation run 5 (plan: confirm5_plan.md): zero-shot transferred TP53 signature vs the baselines.

    SMALLN_REPO=<smalln-bench> python confirm5.py <smalln-bench>
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = os.environ.get("SMALLN_REPO", str(__import__("pathlib").Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO)
from bench import cv                                                 # noqa: E402
from bench.run import context                                        # noqa: E402
from bench.subsample import draw                                     # noqa: E402

HERE = Path(__file__).parent
CELLS = [("B2", 6), ("B2", 12), ("B2", 30), ("B5", 6), ("B5", 12), ("B5", 30), ("B5", 50)]
R0, R = 500, 100


def zeroshot(task, N, rs):
    t, _, _ = context(task)
    B = np.load(HERE / "sig4" / f"{task}_beta.npy")
    rows = []
    for r in rs:
        S, H = draw(t, N, r)
        XS = t.X[S].astype(float)
        keep = (~np.isnan(XS)).mean(0) >= 0.5
        mu, sd = np.nanmean(XS[:, keep], 0), np.nanstd(XS[:, keep], 0, ddof=1)
        zH = np.nan_to_num((t.X[H][:, keep] - mu) / np.where(sd > 0, sd, np.nan))
        rows.append(dict(task=task, N=N, r=r, zeroshot=cv.external_auc(t.y[H].astype(int), zH @ B[keep])))
    return rows


if __name__ == "__main__":
    import confirm as c1
    from bench import gpu
    gpu.guard("tabpfn")
    out = HERE / "confirm5.csv"
    rows = []
    for task, N in CELLS:
        t0 = time.time()
        z = pd.DataFrame(zeroshot(task, N, range(R0, R0 + R)))
        b = pd.DataFrame(c1.baselines(task, N, range(R0, R0 + R))).drop(columns="arm")
        rows.append(z.merge(b, on=["task", "N", "r"], validate="one_to_one"))
        pd.concat(rows).to_csv(out, index=False)
        print(f"{time.strftime('%F %T')} {task} N={N}: {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)
