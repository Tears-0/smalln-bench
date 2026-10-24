"""Analysis of confirmation run 3, as fixed in confirm3_plan.md (written before the run)."""
import numpy as np, pandas as pd

pd.set_option("display.width", 220)
D = pd.read_csv("confirm3.csv")

print("== E1: chosen tau per cell and embedding (share tau = 0, share tau <= 0.25, median, share tau >= 2)")
E1 = D.groupby(["task", "N", "emb"]).tau.agg(
    zero=lambda v: (v == 0).mean(), small=lambda v: (v <= 0.25).mean(), median="median",
    large=lambda v: (v >= 2).mean()).round(2)
print(E1.unstack("emb").to_string())
print("\nPooled over cells:")
print(D.groupby("emb").tau.agg(zero=lambda v: (v == 0).mean(), small=lambda v: (v <= 0.25).mean(),
                               median="median", large=lambda v: (v >= 2).mean()).round(2).to_string())


def rule(d):
    med, pos, neg = d.median(), (d > 0).mean(), (d < 0).mean()
    return med, pos, neg, ("beats" if med > 0.02 and pos >= 0.75 else "is beaten" if med < -0.02 and neg >= 0.75 else "no")


rows = []
for (task, N, e), g in D.groupby(["task", "N", "emb"]):
    for a, b in (("ext_star", "ext_0"), ("ext_1", "ext_0")):
        med, pos, neg, flag = rule(g[a] - g[b])
        rows.append(dict(task=task, N=N, emb=e, comparison=f"{a} - {b}", med=med, pos=pos, neg=neg, flag=flag))
R = pd.DataFrame(rows)
R.to_csv("confirm3_paired.csv", index=False)
R["cell"] = R.apply(lambda x: f"{x.med:+.3f} (+{x.pos:.2f}/-{x.neg:.2f}) {x.flag}", axis=1)
print("\n== Held-out AUC: tuned tau minus tau = 0 (H6), and tau = 1 (V2) minus tau = 0 (descriptive)")
print(R.pivot_table(index=["task", "N"], columns=["comparison", "emb"], values="cell", aggfunc="first").to_string())
print("\nMedian held-out AUC")
print(D.groupby(["task", "N", "emb"])[["ext_0", "ext_star", "ext_1"]].median().round(3).unstack("emb").to_string())

print("\n== H6 (tuned beats tau = 0, rule in BOTH B2 and B5)")
for e in ("esmpc32", "coexp32", "esmpc32_shuf", "coexp32_shuf"):
    for N in (12, 30):
        g = R[(R.comparison == "ext_star - ext_0") & (R.emb == e) & (R.N == N)]
        v = ("tuned beats ridge" if (g.flag == "beats").all() else
             "ridge beats tuned" if (g.flag == "is beaten").all() else "no difference")
        print(f"  {e:14s} N = {N}: {v}")
