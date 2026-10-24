"""Run 2: co-expression embedding coverage among V3's input proteins, over every run-2 draw (post hoc record).

A protein is covered if its co-expression embedding is non-zero. V3's input is the reference pipeline's 500
in-fold proteins, restricted to proteins with a sequence, on the run's own draws (repeats 200-299).

    python coverage2.py [emb2 directory]       (default: ./emb2, built by `confirm2.py <repo> emb`)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REPO = str(HERE.resolve().parents[1])
sys.path.insert(0, REPO)
from bench.preprocess import prepare                    # noqa: E402
from bench.run import context                           # noqa: E402
from bench.subsample import draw, seed                  # noqa: E402

EMB = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "emb2"
CELLS = [("B2", 6), ("B2", 12), ("B2", 30), ("B5", 6), ("B5", 12), ("B5", 30), ("B5", 50)]
rows = []
for task, N in CELLS:
    t, ctx, _ = context(task)
    cov = np.abs(np.load(EMB / f"{task}_coexp32.npy")).sum(1) > 0
    for r in range(200, 300):
        S, H = draw(t, N, r)
        p = prepare(ctx, S, H, t.y[S].astype(int), "ref", seed("impute", task, N, r, "true", "ext", 0), restrict_seq=True)
        rows.append(dict(task=task, N=N, r=r, covered=float(cov[p.feat].mean()), all_proteins=float(cov.mean())))
D = pd.DataFrame(rows)
D.to_csv(HERE / "confirm2_coverage.csv", index=False)
print(D.groupby("task").covered.agg(["mean", "min", "max"]).round(3).to_string())
print(D.groupby(["task", "N"]).covered.mean().round(3).to_string())
print(D.groupby("task").all_proteins.first().round(3).to_string())
