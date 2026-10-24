"""Supplementary figure for run 5: zero-shot transferred TP53 signature vs the benchmark's models."""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]).replace("\\", "/")   # repository root
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, _REPO)
from bench.figures import MODEL_COLOR, INK, INK2, MUTED, GRID, AXIS, NULL_FILL  # noqa: E402

D = pd.read_csv("confirm5.csv")
COL = {"zeroshot": "#e34948", "TabPFN": "#eda100"}   # validated with --pairs all; CCP/EN/RF as a grey band
NS = [6, 12, 30, 50]
BASE = ["CCP", "EN", "RF", "TabPFN"]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
for j, (task, title) in enumerate({"B2": "B2 PDAC TP53", "B5": "B5 haem. lines TP53"}.items()):
    g = D[D.task == task]
    Ns = sorted(g.N.unique()); xs = [NS.index(n) for n in Ns]
    med = g.groupby("N").median(numeric_only=True)
    ax = axes[0, j]
    ax.fill_between(xs, med[["CCP", "EN", "RF"]].min(1), med[["CCP", "EN", "RF"]].max(1), color=NULL_FILL, lw=0, zorder=1)
    for m in ["TabPFN", "zeroshot"]:
        hero = m == "zeroshot"
        ax.plot(xs, med[m], color=COL[m], lw=2.4 if hero else 1.6, marker="o", ms=5 if hero else 4.2, mec="white",
                mew=0.9, zorder=5 if hero else 3)
    ends = {m: med[m].iloc[-1] for m in ["TabPFN", "zeroshot"]}
    lab = {"zeroshot": "zero-shot", "TabPFN": "TabPFN"}
    placed = []
    for m in sorted(ends, key=ends.get):
        y = ends[m] if not placed else max(ends[m], placed[-1] + 0.014)
        placed.append(y)
        ax.annotate(lab[m], (xs[-1], ends[m]), xytext=(xs[-1] + 0.12, y), textcoords="data", va="center",
                    fontsize=6.8, color=INK2, annotation_clip=False)
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)
    ax.set_ylim(0.5, 0.82); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    if j == 0:
        ax.set_ylabel("Median held-out AUC\n(repeats 500-599)", color=INK2)
    ax = axes[1, j]
    ax.axhspan(-0.02, 0.02, color=NULL_FILL, lw=0, zorder=0); ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for n in Ns:
        d = g[g.N == n].zeroshot - g[g.N == n].TabPFN
        q1, q2, q3 = d.quantile([0.25, 0.5, 0.75])
        met = q2 > 0.02 and (d > 0).mean() >= 0.75
        ax.plot([NS.index(n)] * 2, [q1, q3], color=COL["TabPFN"], lw=1.8, zorder=2)
        ax.plot([NS.index(n)], [q2], marker="o", ms=6.5, color=COL["TabPFN"] if met else "white",
                mec=COL["TabPFN"], mew=1.5, zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlabel("N")
    if j == 0:
        ax.set_ylabel("Zero-shot minus TabPFN\n(median and IQR)", color=INK2)
    for a_ in axes[:, j]:
        a_.set_xticks(range(len(NS))); a_.set_xticklabels([str(n) for n in NS]); a_.set_xlim(-0.45, len(NS) - 0.45)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
lo = min(a.get_ylim()[0] for a in axes[1]); hi = max(a.get_ylim()[1] for a in axes[1])
for a in axes[1]:
    a.set_ylim(lo, hi)
axes[0, 0].text(3, 0.54, "N = 50\nnot estimable", ha="center", va="center", fontsize=6.5, color=MUTED)
h = [Line2D([], [], color=COL["zeroshot"], lw=2.4, marker="o", ms=5, mec="white",
            label="Transferred TP53 signature, zero-shot (no task labels)")]
h += [Line2D([], [], color=COL["TabPFN"], lw=1.6, marker="o", ms=4, mec="white", label="TabPFN")]
h += [Patch(color=NULL_FILL, label="range of CCP, EN, RF (top) / ±0.02 rule margin (bottom)"),
      Line2D([], [], ls="none", marker="o", ms=5, color=INK2, label="paired rule met (filled)")]
fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985), fontsize=6.8, frameon=False,
           handlelength=2.2, columnspacing=1.1)
fig.subplots_adjust(hspace=0.16, wspace=0.2, right=0.88, top=0.905)
for ext in ("png", "pdf"):
    fig.savefig(f"confirm5.{ext}", dpi=220, bbox_inches="tight")
print("figure written")
