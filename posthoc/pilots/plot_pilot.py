"""FusionV2 pilot: held-out AUC by N, and FusionV2's paired gain over each comparator."""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]).replace("\\", "/")   # repository root
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, _REPO + "")
from bench.figures import MODEL_COLOR, INK, INK2, MUTED, GRID, AXIS, NULL_FILL  # noqa: E402  (also sets rcParams)

COL = dict(MODEL_COLOR, FusionV2="#008300", Ridge="#6250d6")       # slots 6, 7 of the validated palette
NAME = {"CCP": "CCP", "EN": "Elastic net", "RF": "Random forest", "TabPFN": "TabPFN",
        "FUS": "Fusion (original)", "FusionV2": "FusionV2 (EIL)", "Ridge": "Ridge (FusionV2 without ESM)"}
NS = [6, 12, 30, 50]
TASKS = {"B2": "B2 PDAC TP53", "B5": "B5 haem. lines TP53"}

P = pd.read_csv("pilot.csv").rename(columns={"EIL": "FusionV2"})
D = pd.read_parquet(_REPO + "/results/main_all_runs.parquet",
                    columns=["task", "N", "r", "arm", "config", "auc_ext"])
B = D[D.config.isin(MODEL_COLOR)].pivot_table(index=["task", "N", "r", "arm"], columns="config",
                                               values="auc_ext").reset_index()
M = P.merge(B, on=["task", "N", "r", "arm"], validate="one_to_one")
T = M[M.arm == "true"]
x = lambda Ns: [NS.index(n) for n in Ns]

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
for j, (task, title) in enumerate(TASKS.items()):
    g = T[T.task == task]
    Ns = sorted(g.N.unique())
    med = g.groupby("N")[list(MODEL_COLOR) + ["Ridge", "FusionV2"]].median()

    # --- top: median held-out AUC
    ax = axes[0, j]
    for m in ["CCP", "EN", "RF", "TabPFN", "FUS", "Ridge", "FusionV2"]:
        hero = m == "FusionV2"
        ax.plot(x(Ns), med[m], color=COL[m], lw=2.6 if hero else 1.4, ls=(0, (4, 2)) if m == "Ridge" else "-",
                marker="o", ms=6 if hero else 4.2, mec="white", mew=1.0, zorder=5 if hero else 3)
    for m, dy in (("FusionV2", 0), ("FUS", 0)):
        ax.annotate(NAME[m].split(" (")[0], (x(Ns)[-1], med[m].iloc[-1]), xytext=(7, dy), textcoords="offset points",
                    va="center", fontsize=7, color=INK2, annotation_clip=False)
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)
    ax.set_ylim(0.45, 0.82)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    if j == 0:
        ax.set_ylabel("Median held-out AUC", color=INK2)

    # --- bottom: paired gain FusionV2 - comparator, median and IQR
    ax = axes[1, j]
    ax.axhspan(-0.02, 0.02, color=NULL_FILL, lw=0, zorder=0)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    comps = ["CCP", "EN", "RF", "TabPFN", "Ridge"]
    for k, c in enumerate(comps):
        dx = (k - (len(comps) - 1) / 2) * 0.13
        for n in Ns:
            d = g[g.N == n].FusionV2 - g[g.N == n][c]
            q1, q2, q3 = d.quantile([0.25, 0.5, 0.75])
            met = (q2 > 0.02 and (d > 0).mean() >= 0.75) or (q2 < -0.02 and (d < 0).mean() >= 0.75)
            xi = NS.index(n) + dx
            ax.plot([xi, xi], [q1, q3], color=COL[c], lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot([xi], [q2], marker="o", ms=5, color=COL[c] if met else "white", mec=COL[c], mew=1.4, zorder=3)
    ax.set_ylim(-0.075, 0.095)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("N")
    if j == 0:
        ax.set_ylabel("FusionV2 minus comparator\n(held-out AUC, median and IQR)", color=INK2)
    for a in axes[:, j]:
        a.set_xticks(range(len(NS))); a.set_xticklabels([str(n) for n in NS])
        a.set_xlim(-0.45, len(NS) - 0.45)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
axes[1, 0].text(3, 0.06, "N = 50\nnot estimable", ha="center", va="center", fontsize=6.5, color=MUTED)
axes[0, 0].text(3, 0.49, "N = 50\nnot estimable", ha="center", va="center", fontsize=6.5, color=MUTED)

h = [Line2D([], [], color=COL[m], lw=2.6 if m == "FusionV2" else 1.4, ls=(0, (4, 2)) if m == "Ridge" else "-",
            marker="o", ms=5, mec="white", label=NAME[m]) for m in ["FusionV2", "Ridge", "FUS", "CCP", "EN", "RF", "TabPFN"]]
h += [Line2D([], [], ls="none", marker="o", ms=5, color=INK2, label="paired rule met"),
      Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=INK2, mew=1.4, label="rule not met"),
      Patch(color=NULL_FILL, label="within \u00b10.02 (rule margin)")]
fig.legend(handles=h, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=7, handlelength=2.2,
           columnspacing=1.2, frameon=False)
axes[1, 1].text(0.0, 1.02, "markers left to right in each N: CCP, elastic net, random forest, TabPFN, Ridge",
                transform=axes[1, 1].transAxes, fontsize=6.6, color=MUTED, ha="left", va="bottom")
fig.subplots_adjust(hspace=0.16, wspace=0.18, right=0.9, top=0.86)
for ext in ("png", "pdf"):
    fig.savefig(f"fusionv2_pilot.{ext}", dpi=220, bbox_inches="tight")
print("ok")
