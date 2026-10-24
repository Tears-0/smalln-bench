"""Analysis of confirmation run 4, as fixed in confirm4_plan.md (written before the run)."""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]).replace("\\", "/")   # repository root
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, _REPO + "")
from bench.figures import INK, INK2, MUTED, GRID, AXIS, NULL_FILL  # noqa: E402

pd.set_option("display.width", 220)
KEY = ["task", "N", "r"]
T = pd.read_csv("confirm4_transfer.csv").set_index(KEY)
B = pd.read_csv("confirm4_baselines.csv").drop(columns="arm").set_index(KEY)
M = T.join(B, how="left").reset_index()
assert M[["CCP", "EN", "RF", "TabPFN"]].notna().all().all(), "merge failed"
C3 = pd.read_csv("confirm3.csv")

print("== E2: share of draws choosing tau = 0 (run 4 transfer vs run 3 embeddings)")
e2 = M.groupby(["task", "N"]).tau.agg(zero=lambda v: (v == 0).mean(), median="median", large=lambda v: (v >= 2).mean())
r3 = C3[C3.emb.isin(["esmpc32", "coexp32"])].groupby(["task", "N"]).tau.apply(lambda v: (v == 0).mean()).rename("run3_zero")
print(e2.join(r3).round(2).to_string())
print(f"pooled: run 4 tau = 0 in {(M.tau == 0).mean():.2f} of draws; run 3 (real embeddings) "
      f"{(C3[C3.emb.isin(['esmpc32', 'coexp32'])].tau == 0).mean():.2f}")


def rule(d):
    med, pos, neg = d.median(), (d > 0).mean(), (d < 0).mean()
    return med, pos, neg, ("beats" if med > 0.02 and pos >= 0.75 else "is beaten" if med < -0.02 and neg >= 0.75 else "no")


rows = []
for (task, N), g in M.groupby(["task", "N"]):
    for a, b in (("transfer", "ridge"), ("transfer", "TabPFN"), ("zeroshot", "ridge"), ("zeroshot", "TabPFN"),
                 ("tau1", "ridge"), ("transfer", "EN")):
        med, pos, neg, flag = rule(g[a] - g[b])
        rows.append(dict(task=task, N=N, comparison=f"{a} - {b}", med=med, pos=pos, neg=neg, flag=flag))
R = pd.DataFrame(rows)
R.to_csv("confirm4_paired.csv", index=False)
R["cell"] = R.apply(lambda x: f"{x.med:+.3f} (+{x.pos:.2f}/-{x.neg:.2f}) {x.flag}", axis=1)
print("\n== Median held-out AUC")
print(M.groupby(["task", "N"])[["zeroshot", "ridge", "transfer", "tau1", "CCP", "EN", "RF", "TabPFN"]].median().round(3).to_string())
print("\n== Paired differences: median (share above / below zero), rule outcome per task")
print(R.pivot_table(index=["task", "N"], columns="comparison", values="cell", aggfunc="first").to_string())


def verdict(a, b, N):
    g = R[(R.comparison == f"{a} - {b}") & (R.N == N)]
    if (g.flag == "beats").all():
        return f"{a} beats {b}"
    if (g.flag == "is beaten").all():
        return f"{b} beats {a}"
    return "no difference"


print("\n== Pre-specified hypotheses (rule in BOTH B2 and B5)")
for N in (12, 30):
    print(f"  N = {N}: H7 {verdict('transfer', 'ridge', N):35s} | H8 {verdict('transfer', 'TabPFN', N)}")
print("  descriptive, zero-shot: " + "; ".join(f"N = {N} vs ridge: {verdict('zeroshot', 'ridge', N)}, "
                                               f"vs TabPFN: {verdict('zeroshot', 'TabPFN', N)}" for N in (12, 30)))

# ------------------------------------------------------------------ figure
COL = {"transfer": "#e34948", "zeroshot": "#1baf7a", "ridge": "#2a78d6", "TabPFN": "#eda100"}   # validated, all pairs
NAME = {"transfer": "Ridge + transferred TP53 signature (tau by inner CV)", "zeroshot": "Transferred signature alone (zero-shot)",
        "ridge": "Ridge (no outside information)", "TabPFN": "TabPFN"}
NS = [6, 12, 30, 50]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
for j, (task, title) in enumerate({"B2": "B2 PDAC TP53", "B5": "B5 haem. lines TP53"}.items()):
    g = M[M.task == task]
    Ns = sorted(g.N.unique()); xs = [NS.index(n) for n in Ns]
    med = g.groupby("N").median(numeric_only=True)
    ax = axes[0, j]
    ax.fill_between(xs, med[["CCP", "EN", "RF", "TabPFN"]].min(1), med[["CCP", "EN", "RF", "TabPFN"]].max(1),
                    color=NULL_FILL, lw=0, zorder=1)
    for m in ["TabPFN", "ridge", "zeroshot", "transfer"]:
        ax.plot(xs, med[m], color=COL[m], lw=2.4 if m == "transfer" else 1.6, ls=(0, (4, 2)) if m == "zeroshot" else "-",
                marker="o", ms=5 if m == "transfer" else 4.2, mec="white", mew=1.0, zorder=4)
    ends = {m: med[m].iloc[-1] for m in COL}
    lab = {"transfer": "transfer", "zeroshot": "zero-shot", "ridge": "ridge", "TabPFN": "TabPFN"}
    placed = []
    for m in sorted(ends, key=ends.get):
        y = ends[m] if not placed else max(ends[m], placed[-1] + 0.016)
        placed.append(y)
        ax.annotate(lab[m], (xs[-1], ends[m]), xytext=(xs[-1] + 0.12, y), textcoords="data", va="center",
                    fontsize=6.8, color=INK2, annotation_clip=False)
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)
    ax.set_ylim(0.45, 0.85); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    if j == 0:
        ax.set_ylabel("Median held-out AUC\n(fresh repeats 400-499)", color=INK2)
    ax = axes[1, j]
    ax.axhspan(-0.02, 0.02, color=NULL_FILL, lw=0, zorder=0); ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for k, b in enumerate(["ridge", "TabPFN"]):
        dx = (k - 0.5) * 0.22
        for n in Ns:
            d = g[g.N == n]["transfer"] - g[g.N == n][b]
            q1, q2, q3 = d.quantile([0.25, 0.5, 0.75])
            met = rule(d)[3] != "no"
            xi = NS.index(n) + dx
            ax.plot([xi, xi], [q1, q3], color=COL[b], lw=1.6, zorder=2)
            ax.plot([xi], [q2], marker="o", ms=6, color=COL[b] if met else "white", mec=COL[b], mew=1.5, zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlabel("N")
    if j == 0:
        ax.set_ylabel("Transfer minus comparator\n(median and IQR)", color=INK2)
    for a_ in axes[:, j]:
        a_.set_xticks(range(len(NS))); a_.set_xticklabels([str(n) for n in NS]); a_.set_xlim(-0.45, len(NS) - 0.45)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
lo = min(a.get_ylim()[0] for a in axes[1]); hi = max(a.get_ylim()[1] for a in axes[1])
for a in axes[1]:
    a.set_ylim(lo, hi)
axes[0, 0].text(3, 0.49, "N = 50\nnot estimable", ha="center", va="center", fontsize=6.5, color=MUTED)
h = [Line2D([], [], color=COL[m], lw=2.4 if m == "transfer" else 1.6, ls=(0, (4, 2)) if m == "zeroshot" else "-",
            marker="o", ms=5, mec="white", label=NAME[m]) for m in ["transfer", "zeroshot", "ridge", "TabPFN"]]
h += [Patch(color=NULL_FILL, label="range of CCP, EN, RF, TabPFN (top) / \u00b10.02 margin (bottom)"),
      Line2D([], [], ls="none", marker="o", ms=5, color=INK2, label="paired rule met (filled)")]
fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=6.8, frameon=False,
           handlelength=2.4, columnspacing=1.2)
fig.subplots_adjust(hspace=0.16, wspace=0.2, right=0.88, top=0.84)
for ext in ("png", "pdf"):
    fig.savefig(f"confirm4.{ext}", dpi=220, bbox_inches="tight")
print("\nfigure written")
