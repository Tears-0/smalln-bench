"""FusionV3 (denoiser + V1 encoder; add / multiply) against V1 on the benchmark's draws."""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[2]).replace("\\", "/")   # repository root
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, _REPO + "")
from bench.figures import MODEL_COLOR, INK, INK2, MUTED, GRID, AXIS, NULL_FILL  # noqa: E402

pd.set_option("display.width", 220)
V3 = pd.read_csv("pilot_v3.csv").pivot_table(index=["task", "N", "r", "arm"], columns="mode", values="auc_ext")
V3.columns = [f"V3_{c}" for c in V3.columns]
V2 = pd.read_csv("pilot.csv").set_index(["task", "N", "r", "arm"])[["EIL"]].rename(columns={"EIL": "V2"})
D = pd.read_parquet(_REPO + "/results/main_all_runs.parquet",
                    columns=["task", "N", "r", "arm", "config", "auc_ext"])
B = D[D.config.isin(MODEL_COLOR)].pivot_table(index=["task", "N", "r", "arm"], columns="config", values="auc_ext")
B = B.rename(columns={"FUS": "V1"})
M = V3.join(V2, how="left").join(B, how="left").reset_index()
NEW = ["V3_add", "V3_multiply"]
COMP = ["V1", "V2", "CCP", "EN", "RF", "TabPFN"]

T, Pm = M[M.arm == "true"], M[M.arm == "perm"]
print("Permuted labels, mean held-out AUC (should be ~0.5)")
print(Pm.groupby(["task", "N"])[NEW + ["V1"]].mean().round(3).to_string())
print("\nMedian held-out AUC, true labels")
print(T.groupby(["task", "N"])[NEW + COMP].median().round(3).to_string())


def rule(d):
    med, pos, neg = d.median(), (d > 0).mean(), (d < 0).mean()
    return med, pos, ("W" if med > 0.02 and pos >= 0.75 else "L" if med < -0.02 and neg >= 0.75 else "")


rows = []
for (task, N), g in T.groupby(["task", "N"]):
    for a in NEW:
        for b in COMP + [x for x in NEW if x != a]:
            med, pos, flag = rule(g[a] - g[b])
            rows.append(dict(task=task, N=N, model=a, vs=b, med=med, pos=pos, flag=flag))
R = pd.DataFrame(rows)
R["cell"] = R.apply(lambda x: f"{x.med:+.3f} ({x.pos:.2f}) {x.flag}".rstrip(), axis=1)
for a in NEW:
    print(f"\n--- {a} minus comparator: median (fraction > 0); W = rule met, L = comparator wins")
    print(R[R.model == a].pivot_table(index=["task", "N"], columns="vs", values="cell", aggfunc="first").to_string())
print("\nClaims (rule holds in BOTH B2 and B5 at the same N):")
for a in NEW:
    for b in R[R.model == a].vs.unique():
        for N in (6, 12, 30):
            g = R[(R.model == a) & (R.vs == b) & (R.N == N)]
            if len(g) == 2 and (g.flag == "W").all():
                print(f"  {a} beats {b} at N = {N}")
            if len(g) == 2 and (g.flag == "L").all():
                print(f"  {b} beats {a} at N = {N}")
R.drop(columns="cell").to_csv("pilot_v3_paired.csv", index=False)

# ------------------------------------------------------------------ figure
COL = dict(MODEL_COLOR, V1=MODEL_COLOR["FUS"], V2="#008300", V3_add="#6250d6", V3_multiply="#e34948")
NAME = {"V1": "V1 (original fusion)", "V2": "V2 (FusionV2, EIL)", "V3_add": "V3 add (denoiser, e + gate)",
        "V3_multiply": "V3 multiply (denoiser, e \u00d7 gate)", "CCP": "CCP", "EN": "Elastic net",
        "RF": "Random forest", "TabPFN": "TabPFN"}
NS = [6, 12, 30, 50]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
for j, (task, title) in enumerate({"B2": "B2 PDAC TP53", "B5": "B5 haem. lines TP53"}.items()):
    g = T[T.task == task]
    Ns = sorted(g.N.unique()); xs = [NS.index(n) for n in Ns]
    med = g.groupby("N").median(numeric_only=True)
    ax = axes[0, j]
    for m in ["CCP", "EN", "RF", "TabPFN", "V2", "V1", "V3_add", "V3_multiply"]:
        hero = m.startswith("V3")
        base = m in ("CCP", "EN", "RF", "TabPFN")
        ax.plot(xs, med[m], color=COL[m], lw=2.4 if hero else (1.0 if base else 1.6), alpha=0.55 if base else 1,
                marker="o", ms=5.5 if hero else 3.8, mec="white", mew=1.0, zorder=5 if hero else 3)
    for m in ("V3_add", "V3_multiply", "V1"):
        ax.annotate({"V3_add": "V3 add", "V3_multiply": "V3 mult.", "V1": "V1"}[m], (xs[-1], med[m].iloc[-1]),
                    xytext=(7, 0), textcoords="offset points", va="center", fontsize=7, color=INK2, annotation_clip=False)
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)
    ax.set_ylim(0.45, 0.82); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    if j == 0:
        ax.set_ylabel("Median held-out AUC", color=INK2)
    ax = axes[1, j]
    ax.axhspan(-0.02, 0.02, color=NULL_FILL, lw=0, zorder=0); ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for k, a in enumerate(NEW):
        dx = (k - 0.5) * 0.22
        for n in Ns:
            d = g[g.N == n][a] - g[g.N == n]["V1"]
            q1, q2, q3 = d.quantile([0.25, 0.5, 0.75])
            met = rule(d)[2] != ""
            xi = NS.index(n) + dx
            ax.plot([xi, xi], [q1, q3], color=COL[a], lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot([xi], [q2], marker="o", ms=6, color=COL[a] if met else "white", mec=COL[a], mew=1.5, zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlabel("N")
    if j == 0:
        ax.set_ylabel("V3 minus V1\n(held-out AUC, median and IQR)", color=INK2)
    for a_ in axes[:, j]:
        a_.set_xticks(range(len(NS))); a_.set_xticklabels([str(n) for n in NS]); a_.set_xlim(-0.45, len(NS) - 0.45)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
lo = min(a.get_ylim()[0] for a in axes[1]); hi = max(a.get_ylim()[1] for a in axes[1])
for a in axes[1]:
    a.set_ylim(lo, hi)
axes[0, 0].text(3, 0.49, "N = 50\nnot estimable", ha="center", va="center", fontsize=6.5, color=MUTED)
h = [Line2D([], [], color=COL[m], lw=2.4 if m.startswith("V3") else 1.4, marker="o", ms=5, mec="white",
            alpha=0.55 if m in ("CCP", "EN", "RF", "TabPFN") else 1, label=NAME[m])
     for m in ["V3_add", "V3_multiply", "V1", "V2", "CCP", "EN", "RF", "TabPFN"]]
h += [Line2D([], [], ls="none", marker="o", ms=5, color=INK2, label="paired rule met"),
      Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=INK2, mew=1.4, label="rule not met"),
      Patch(color=NULL_FILL, label="within \u00b10.02 (rule margin)")]
fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=7, frameon=False,
           handlelength=2.2, columnspacing=1.2)
fig.subplots_adjust(hspace=0.16, wspace=0.18, right=0.9, top=0.84)
for ext in ("png", "pdf"):
    fig.savefig(f"fusionv3_pilot.{ext}", dpi=220, bbox_inches="tight")
print("figure written")
