"""Analysis of confirmation run 2, as fixed in confirm2_plan.md (written before the run)."""
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

pd.set_option("display.width", 240)
KEY = ["task", "N", "r", "arm"]
V3 = pd.read_csv("confirm2_v3.csv", dtype={"arm": str}).pivot_table(index=KEY, columns="model", values="auc_ext")
Bs = pd.read_csv("confirm2_baselines.csv", dtype={"arm": str}).set_index(KEY)
V2 = pd.read_csv("confirm2_v2.csv", dtype={"arm": str}).set_index(KEY)
M = V3.join(Bs, how="left").join(V2, how="left").reset_index()
T, P = M[M.arm == "true"], M[M.arm == "perm"]
REAL = ["coexp32", "coexp64", "esmpc32", "esmpc64"]
assert T[["CCP", "TabPFN", "V2"]].notna().all().all(), "merge failed"

print("== Check: permuted-label mean held-out AUC at N = 12 (plan: within 0.5 +/- 0.05)")
pm = P.groupby(["task", "N"])[REAL].mean()
print(pm.round(3).to_string())
print("   ->", "PASS" if ((pm - 0.5).abs() <= 0.05).all().all() else "FAIL")


def rule(d):
    med, pos, neg = d.median(), (d > 0).mean(), (d < 0).mean()
    return med, pos, neg, ("beats" if med > 0.02 and pos >= 0.75 else "is beaten" if med < -0.02 and neg >= 0.75 else "no")


rows = []
for (task, N), g in T.groupby(["task", "N"]):
    for e in REAL:
        for b in (f"{e}_shuf", "TabPFN", "V2"):
            med, pos, neg, flag = rule(g[e] - g[b])
            rows.append(dict(task=task, N=N, comparison=f"{e} - {b}", med=med, pos=pos, neg=neg, flag=flag))
R = pd.DataFrame(rows)
R.to_csv("confirm2_paired.csv", index=False)
R["cell"] = R.apply(lambda x: f"{x.med:+.3f} (+{x.pos:.2f}/-{x.neg:.2f}) {x.flag}", axis=1)
print("\n== Median held-out AUC (true labels)")
print(T.groupby(["task", "N"])[REAL + [f"{e}_shuf" for e in REAL] + ["V2", "CCP", "EN", "RF", "TabPFN"]]
      .median().round(3).to_string())
for e in REAL:
    print(f"\n== {e}: paired differences, median (share above / below zero), rule outcome per task")
    print(R[R.comparison.str.startswith(e + " ")].pivot_table(index=["task", "N"], columns="comparison",
                                                              values="cell", aggfunc="first").to_string())


def verdict(a, b, N):
    g = R[(R.comparison == f"{a} - {b}") & (R.N == N)]
    if (g.flag == "beats").all():
        return f"{a} beats {b}"
    if (g.flag == "is beaten").all():
        return f"{b} beats {a}"
    return "no difference"


print("\n== Pre-specified hypotheses (rule in BOTH B2 and B5)")
for N in (12, 30):
    for e in REAL:
        h = "H3" if e.startswith("coexp") else "H4"
        print(f"  N = {N}  {h} {e}: {verdict(e, e + '_shuf', N):45s} | H5 vs TabPFN: {verdict(e, 'TabPFN', N)}")

# ------------------------------------------------------------------ figure
COL = {"coexp": "#1baf7a", "esmpc": "#6250d6", "TabPFN": "#eda100", "V2": "#008300"}
LS = {"32": "-", "64": (0, (4, 2))}
NS = [6, 12, 30, 50]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
for j, (task, title) in enumerate({"B2": "B2 PDAC TP53", "B5": "B5 haem. lines TP53"}.items()):
    g = T[T.task == task]
    Ns = sorted(g.N.unique()); xs = [NS.index(n) for n in Ns]
    med = g.groupby("N").median(numeric_only=True)
    ax = axes[0, j]
    ax.fill_between(xs, med[["CCP", "EN", "RF", "TabPFN"]].min(1), med[["CCP", "EN", "RF", "TabPFN"]].max(1),
                    color=NULL_FILL, lw=0, zorder=1)
    for m in ["TabPFN", "V2"] + REAL:
        src, dim = (m[:-2], m[-2:]) if m in REAL else (m, "32")
        ax.plot(xs, med[m], color=COL[src], ls=LS[dim], lw=1.9 if m in REAL else 1.5, marker="o", ms=4.2,
                mec="white", mew=0.9, zorder=4)
    ends = {m: med[m].iloc[-1] for m in ["TabPFN", "V2"] + REAL}
    lab = {"TabPFN": "TabPFN", "V2": "V2", "coexp32": "co-exp 32", "coexp64": "co-exp 64",
           "esmpc32": "ESM-PC 32", "esmpc64": "ESM-PC 64"}
    placed = []
    for m in sorted(ends, key=ends.get):
        y = ends[m] if not placed else max(ends[m], placed[-1] + 0.015)
        placed.append(y)
        ax.annotate(lab[m], (xs[-1], ends[m]), xytext=(xs[-1] + 0.12, y), textcoords="data", va="center",
                    fontsize=6.5, color=INK2, annotation_clip=False)
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)
    ax.set_ylim(0.45, 0.82); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    if j == 0:
        ax.set_ylabel("Median held-out AUC\n(fresh repeats 200-299)", color=INK2)
    ax = axes[1, j]
    ax.axhspan(-0.02, 0.02, color=NULL_FILL, lw=0, zorder=0); ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for k, e in enumerate(REAL):
        dx = (k - 1.5) * 0.15
        src = e[:-2]
        for n in Ns:
            d = g[g.N == n][e] - g[g.N == n][e + "_shuf"]
            q1, q2, q3 = d.quantile([0.25, 0.5, 0.75])
            met = rule(d)[3] != "no"
            xi = NS.index(n) + dx
            ax.plot([xi, xi], [q1, q3], color=COL[src], lw=1.5, ls=LS[e[-2:]], zorder=2)
            ax.plot([xi], [q2], marker="o" if e.endswith("32") else "s", ms=5.5, color=COL[src] if met else "white",
                    mec=COL[src], mew=1.4, zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlabel("N")
    if j == 0:
        ax.set_ylabel("Real minus shuffled embedding\n(median and IQR)", color=INK2)
    for a_ in axes[:, j]:
        a_.set_xticks(range(len(NS))); a_.set_xticklabels([str(n) for n in NS]); a_.set_xlim(-0.45, len(NS) - 0.45)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
lo = min(a.get_ylim()[0] for a in axes[1]); hi = max(a.get_ylim()[1] for a in axes[1])
for a in axes[1]:
    a.set_ylim(lo, hi)
axes[0, 0].text(3, 0.49, "N = 50\nnot estimable", ha="center", va="center", fontsize=6.5, color=MUTED)
h = [Line2D([], [], color=COL["coexp"], lw=1.9, marker="o", ms=5, mec="white", label="V3 + co-expression PCs, d = 32"),
     Line2D([], [], color=COL["coexp"], lw=1.9, ls=LS["64"], marker="s", ms=5, mec="white", label="V3 + co-expression PCs, d = 64"),
     Line2D([], [], color=COL["esmpc"], lw=1.9, marker="o", ms=5, mec="white", label="V3 + ESM-2 PCs, d = 32"),
     Line2D([], [], color=COL["esmpc"], lw=1.9, ls=LS["64"], marker="s", ms=5, mec="white", label="V3 + ESM-2 PCs, d = 64"),
     Line2D([], [], color=COL["TabPFN"], lw=1.5, marker="o", ms=5, mec="white", label="TabPFN"),
     Line2D([], [], color=COL["V2"], lw=1.5, marker="o", ms=5, mec="white", label="V2 (ridge + ESM PCs)"),
     Patch(color=NULL_FILL, label="range of CCP, EN, RF, TabPFN (top) / \u00b10.02 margin (bottom)"),
     Line2D([], [], ls="none", marker="o", ms=5, color=INK2, label="paired rule met (filled)")]
fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=6.8, frameon=False,
           handlelength=2.4, columnspacing=1.1)
fig.subplots_adjust(hspace=0.16, wspace=0.2, right=0.88, top=0.83)
for ext in ("png", "pdf"):
    fig.savefig(f"confirm2.{ext}", dpi=220, bbox_inches="tight")
print("\nfigure written")
