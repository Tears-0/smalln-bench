"""Judge the FusionV2 pilot with the paper's paired decision rule, on the benchmark's own draws."""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]).replace("\\", "/")   # repository root
import numpy as np, pandas as pd

pd.set_option("display.width", 220)
P = pd.read_csv("pilot.csv")
D = pd.read_parquet(_REPO + "/results/main_all_runs.parquet",
                    columns=["task", "N", "r", "arm", "config", "auc_ext"])
B = D[D.config.isin(["CCP", "EN", "RF", "TabPFN", "FUS"])].pivot_table(
    index=["task", "N", "r", "arm"], columns="config", values="auc_ext").reset_index()
M = P.merge(B, on=["task", "N", "r", "arm"], validate="one_to_one")
V = ["Ridge", "EIL", "EIL_esm_only", "Ridge_loo", "EIL_loo", "EIL_en"]
BASE = ["FUS", "CCP", "EN", "RF", "TabPFN"]

print("EN re-run vs stored EN: max |diff| =", float((M.EN_rerun - M.EN).abs().max()), "| rows", len(M))
T = M[M.arm == "true"]
print("\nMedian held-out AUC (true labels)")
print(T.groupby(["task", "N"])[BASE + V].median().round(3).to_string())
print("\nPermuted labels, mean held-out AUC (should be ~0.5)")
print(M[M.arm == "perm"].groupby(["task", "N"])[BASE + V].mean().round(3).to_string())


def rule(d):
    return dict(med=d.median(), pos=(d > 0).mean(), win=bool(d.median() > 0.02 and (d > 0).mean() >= 0.75),
                lose=bool(d.median() < -0.02 and (d < 0).mean() >= 0.75))


rows = []
for (task, N), g in T.groupby(["task", "N"]):
    for v in V:
        for b in BASE:
            rows.append(dict(task=task, N=N, variant=v, vs=b, **rule(g[v] - g[b])))
R = pd.DataFrame(rows)
print("\nPaired difference variant - baseline: median (fraction of draws > 0); W = rule met, L = baseline wins")
R["cell"] = R.apply(lambda x: f"{x.med:+.3f} ({x.pos:.2f}){' W' if x.win else (' L' if x.lose else '')}", axis=1)
for v in V:
    print(f"\n--- {v}")
    print(R[R.variant == v].pivot_table(index=["task", "N"], columns="vs", values="cell", aggfunc="first")[BASE].to_string())

print("\nClaims under the paper's rule (holds in BOTH B2 and B5 at the same N):")
for v in V:
    for b in BASE:
        for N in (6, 12, 30):
            g = R[(R.variant == v) & (R.vs == b) & (R.N == N)]
            if len(g) == 2 and g.win.all():
                print(f"  {v} beats {b} at N = {N}")
            if len(g) == 2 and g.lose.all():
                print(f"  {b} beats {v} at N = {N}")
R.drop(columns="cell").to_csv("pilot_paired.csv", index=False)
