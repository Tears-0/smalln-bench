"""Analysis of confirmation run 5, as fixed in confirm5_plan.md (written before the run)."""
import pandas as pd

pd.set_option("display.width", 200)
D = pd.read_csv("confirm5.csv")
BASE = ["CCP", "EN", "RF", "TabPFN"]
print("== Median held-out AUC (repeats 500-599)")
print(D.groupby(["task", "N"])[["zeroshot"] + BASE].median().round(3).to_string())


def rule(d):
    med, pos, neg = d.median(), (d > 0).mean(), (d < 0).mean()
    return med, pos, neg, ("beats" if med > 0.02 and pos >= 0.75 else "is beaten" if med < -0.02 and neg >= 0.75 else "no")


rows = []
for (task, N), g in D.groupby(["task", "N"]):
    for b in BASE:
        med, pos, neg, flag = rule(g.zeroshot - g[b])
        rows.append(dict(task=task, N=N, vs=b, med=med, pos=pos, neg=neg, flag=flag))
R = pd.DataFrame(rows)
R.to_csv("confirm5_paired.csv", index=False)
R["cell"] = R.apply(lambda x: f"{x.med:+.3f} (+{x.pos:.2f}/-{x.neg:.2f}) {x.flag}", axis=1)
print("\n== Zero-shot minus comparator: median (share above / below zero), rule outcome per task")
print(R.pivot_table(index=["task", "N"], columns="vs", values="cell", aggfunc="first")[BASE].to_string())
print("\n== H9 (rule in BOTH B2 and B5)")
for N in (12, 30):
    out = []
    for b in BASE:
        g = R[(R.vs == b) & (R.N == N)]
        out.append(f"vs {b}: " + ("zero-shot beats" if (g.flag == "beats").all() else
                                  f"{b} beats" if (g.flag == "is beaten").all() else "no difference"))
    print(f"  N = {N}: " + " | ".join(out))
