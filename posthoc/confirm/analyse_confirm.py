"""Analysis of the confirmation run, exactly as fixed in confirm_plan.md (written before the run)."""
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
KEY = ["task", "N", "r", "arm"]
V3 = pd.read_csv("confirm_v3.csv", dtype={"arm": str}).pivot_table(index=KEY, columns="model", values="auc_ext")
Bs = pd.read_csv("confirm_baselines.csv", dtype={"arm": str}).set_index(KEY)
V2 = pd.read_csv("confirm_v2.csv", dtype={"arm": str}).set_index(KEY)
M = V3.join(Bs, how="left").join(V2, how="left").reset_index()
T, P = M[M.arm == "true"], M[M.arm == "perm"]

print("== Check: permuted-label mean held-out AUC at N = 12 (plan: within 0.5 +/- 0.05)")
pm = P.groupby(["task", "N"])[["V3_esm", "V3_shuffled"]].mean()
print(pm.round(3).to_string())
print("   ->", "PASS" if ((pm - 0.5).abs() <= 0.05).all().all() else "FAIL")


def rule(d):
    med, pos, neg = d.median(), (d > 0).mean(), (d < 0).mean()
    return med, pos, neg, ("beats" if med > 0.02 and pos >= 0.75 else "is beaten" if med < -0.02 and neg >= 0.75 else "no")


rows = []
for (task, N), g in T.groupby(["task", "N"]):
    for a, b in [("V3_esm", "V3_shuffled"), ("V3_esm", "TabPFN"), ("V3_esm", "V2"), ("V3_esm", "CCP"),
                 ("V3_esm", "EN"), ("V3_esm", "RF"), ("V3_shuffled", "TabPFN"), ("V2", "TabPFN")]:
        med, pos, neg, flag = rule(g[a] - g[b])
        rows.append(dict(task=task, N=N, comparison=f"{a} - {b}", med=med, pos=pos, neg=neg, flag=flag))
R = pd.DataFrame(rows)
R.to_csv("confirm_paired.csv", index=False)
R["cell"] = R.apply(lambda x: f"{x.med:+.3f} (+{x.pos:.2f}/-{x.neg:.2f}) {x.flag}", axis=1)
print("\n== Median held-out AUC (true labels)")
print(T.groupby(["task", "N"])[["V3_esm", "V3_shuffled", "V2", "CCP", "EN", "RF", "TabPFN"]].median().round(3).to_string())
print("\n== Paired differences: median (share above / below zero), rule outcome per task")
print(R.pivot_table(index=["task", "N"], columns="comparison", values="cell", aggfunc="first").to_string())


def verdict(a, b, N):
    g = R[(R.comparison == f"{a} - {b}") & (R.N == N)]
    if len(g) < 2:
        return "n/a"
    if (g.flag == "beats").all():
        return f"{a} beats {b}"
    if (g.flag == "is beaten").all():
        return f"{b} beats {a}"
    return "no difference (rule not met in both tasks)"


print("\n== Pre-specified hypotheses (rule in BOTH B2 and B5)")
for N in (12, 30):
    print(f"  N = {N}:  H1 {verdict('V3_esm', 'V3_shuffled', N)} | H2a {verdict('V3_esm', 'TabPFN', N)} | "
          f"H2b {verdict('V3_esm', 'V2', N)}")

# ------------------------------------------------------------------ figure
COL = {"V3_esm": "#e34948", "V3_shuffled": "#6250d6", "V2": "#008300", "TabPFN": "#eda100"}
NAME = {"V3_esm": "V3 with ESM-2", "V3_shuffled": "V3 with shuffled ESM-2 (identity only)",
        "V2": "V2 (ridge + ESM PCs)", "TabPFN": "TabPFN"}
NS = [6, 12, 30, 50]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
for j, (task, title) in enumerate({"B2": "B2 PDAC TP53", "B5": "B5 haem. lines TP53"}.items()):
    g = T[T.task == task]
    Ns = sorted(g.N.unique()); xs = [NS.index(n) for n in Ns]
    med = g.groupby("N").median(numeric_only=True)
    ax = axes[0, j]
    ax.fill_between(xs, med[["CCP", "EN", "RF", "TabPFN"]].min(1), med[["CCP", "EN", "RF", "TabPFN"]].max(1),
                    color=NULL_FILL, lw=0, zorder=1)
    for m in ["TabPFN", "V2", "V3_shuffled", "V3_esm"]:
        ax.plot(xs, med[m], color=COL[m], lw=2.4 if m == "V3_esm" else 1.6, ls=(0, (4, 2)) if m == "V3_shuffled" else "-",
                marker="o", ms=5 if m == "V3_esm" else 4.2, mec="white", mew=1.0, zorder=4)
    ends = {m: med[m].iloc[-1] for m in COL}
    placed = []
    for m in sorted(ends, key=ends.get):
        y = ends[m] if not placed else max(ends[m], placed[-1] + 0.016)
        placed.append(y)
        ax.annotate({"V3_esm": "V3 ESM", "V3_shuffled": "V3 shuffled", "V2": "V2", "TabPFN": "TabPFN"}[m],
                    (xs[-1], ends[m]), xytext=(xs[-1] + 0.12, y), textcoords="data", va="center", fontsize=6.8,
                    color=INK2, annotation_clip=False)
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)
    ax.set_ylim(0.45, 0.82); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    if j == 0:
        ax.set_ylabel("Median held-out AUC\n(fresh repeats 100-199)", color=INK2)
    ax = axes[1, j]
    ax.axhspan(-0.02, 0.02, color=NULL_FILL, lw=0, zorder=0); ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for k, b in enumerate(["V3_shuffled", "TabPFN", "V2"]):
        dx = (k - 1) * 0.2
        for n in Ns:
            d = g[g.N == n]["V3_esm"] - g[g.N == n][b]
            q1, q2, q3 = d.quantile([0.25, 0.5, 0.75])
            met = rule(d)[3] != "no"
            xi = NS.index(n) + dx
            ax.plot([xi, xi], [q1, q3], color=COL[b], lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot([xi], [q2], marker="o", ms=6, color=COL[b] if met else "white", mec=COL[b], mew=1.5, zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlabel("N")
    if j == 0:
        ax.set_ylabel("V3 with ESM-2 minus comparator\n(median and IQR)", color=INK2)
    for a_ in axes[:, j]:
        a_.set_xticks(range(len(NS))); a_.set_xticklabels([str(n) for n in NS]); a_.set_xlim(-0.45, len(NS) - 0.45)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
lo = min(a.get_ylim()[0] for a in axes[1]); hi = max(a.get_ylim()[1] for a in axes[1])
for a in axes[1]:
    a.set_ylim(lo, hi)
axes[0, 0].text(3, 0.49, "N = 50\nnot estimable", ha="center", va="center", fontsize=6.5, color=MUTED)
h = [Line2D([], [], color=COL[m], lw=2.4 if m == "V3_esm" else 1.6, ls=(0, (4, 2)) if m == "V3_shuffled" else "-",
            marker="o", ms=5, mec="white", label=NAME[m]) for m in ["V3_esm", "V3_shuffled", "V2", "TabPFN"]]
h += [Patch(color=NULL_FILL, label="range of CCP, EN, RF, TabPFN (top) / \u00b10.02 rule margin (bottom)"),
      Line2D([], [], ls="none", marker="o", ms=5, color=INK2, label="paired rule met"),
      Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=INK2, mew=1.4, label="rule not met")]
fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=7, frameon=False,
           handlelength=2.2, columnspacing=1.2)
fig.subplots_adjust(hspace=0.16, wspace=0.18, right=0.9, top=0.84)
for ext in ("png", "pdf"):
    fig.savefig(f"confirm.{ext}", dpi=220, bbox_inches="tight")
print("\nfigure written")
