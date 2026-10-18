"""Figures (spec §7, F31). Static print figures for a 10-page Letter paper.

    python -m bench.figures            -> figures/fig{1..5}_*.{pdf,png}

Colour: one fixed hue per model across every figure (validated categorical slots
1-5, light mode: all hard checks pass; slots 3-5 are below 3:1 contrast, so every
figure carries a legend and direct labels, and results/analysis/*.csv is the table
view). Estimator variants are line styles, never extra hues. Fig. 5 colours by N
with a validated single-hue ordinal ramp.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .paths import RESULTS, ROOT

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
A = RESULTS / "analysis"

MODEL_COLOR = {"CCP": "#2a78d6", "EN": "#eb6834", "RF": "#1baf7a", "TabPFN": "#eda100", "FUS": "#e87ba4"}
MODEL_LABEL = {"CCP": "CCP", "EN": "Elastic net", "RF": "Random forest", "TabPFN": "TabPFN", "FUS": "Fusion (case study)"}
N_RAMP = {6: "#86b6ef", 9: "#5598e7", 12: "#2a78d6", 20: "#1c5cab", 30: "#104281"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NULL_FILL = "#e1e0d9"
GRID_N = [6, 9, 12, 20, 30, 50]
TASK_TITLE = {"B1": "B1  LUAD tumour vs normal (large effect)",
              "B2": "B2  PDAC TP53 (small effect)",
              "B5": "B5  Haem. cell lines TP53 (moderate)"}
TASK_SHORT = {"B1": "B1 LUAD tumour vs normal", "B2": "B2 PDAC TP53", "B5": "B5 haem. lines TP53"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
    "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 6.5, "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
LW, MS = 1.4, 4.2


def _xpos(Ns):
    return [GRID_N.index(N) for N in Ns]


def _style(ax, ylabel=None, Ns=GRID_N):
    ax.set_xticks(range(len(GRID_N)))
    ax.set_xticklabels([str(n) for n in GRID_N])
    ax.set_xlim(-0.35, len(GRID_N) - 0.65)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def _line(ax, Ns, ys, color, ls="-", marker="o", lw=LW, filled=True, label=None, z=3):
    x = _xpos(Ns)
    ax.plot(x, ys, ls=ls, color=color, lw=lw, solid_capstyle="round", dash_capstyle="round", zorder=z, label=label)
    if marker:
        ax.plot(x, ys, ls="none", marker=marker, ms=MS, color=color, mfc=color if filled else "white",
                mec="white" if filled else color, mew=0.8 if filled else 0.9, zorder=z + 0.1)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return FIG / f"{name}.png"


# ------------------------------------------------------------------ Fig. 1
def fig1_ranking():
    mr = pd.read_csv(A / "rank_mean_set_i.csv")
    kw = pd.read_csv(A / "rank_kendall_w.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.25), sharey=True)
    for ax, task in zip(axes, ("B1", "B2", "B5")):
        g = mr[mr.task == task].sort_values("N")
        for m in MODEL_COLOR:
            _line(ax, g.N, g[m], MODEL_COLOR[m], label=MODEL_LABEL[m])
        _style(ax, "Mean rank (1 = best of 5)" if task == "B1" else None)
        ax.set_title(TASK_SHORT[task], loc="left", color=INK)
        ax.set_xlabel("N (samples drawn)")
        ax.set_ylim(5.25, 0.75)
        ax.set_yticks([1, 2, 3, 4, 5])
    # direct labels where lines separate (B5, right end); legend carries identity everywhere
    g = mr[(mr.task == "B5")].sort_values("N").iloc[-1]
    for m in MODEL_COLOR:
        axes[2].annotate(MODEL_LABEL[m].split(" (")[0], (len(GRID_N) - 1, g[m]), xytext=(5, 0), textcoords="offset points",
                         va="center", fontsize=6.3, color=INK2, annotation_clip=False)
    axes[0].text(2.5, 3.55, "the four baselines tie\n(external AUC 1.00)", ha="center", va="center", fontsize=6.2,
                 color=MUTED)
    axes[1].text(5, 3.0, "N = 50\nnot\nestimable", ha="center", va="center", fontsize=5.8, color=MUTED)
    axes[0].text(2.0, 4.18, "fusion loses ties:\nAUC deficit < 0.01", ha="center", va="center", fontsize=6.2, color=MUTED)
    w = kw[kw.tasks == "B2+B5 (secondary)"].dropna().set_index("N").W
    fig.text(0.5, -0.06, "Agreement of the B2 and B5 orderings (Kendall's W):  " +
             "   ".join(f"N = {int(n)}: {v:.2f}" for n, v in w.items()), ha="center", fontsize=6.3, color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.09),
               handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.12)
    return _save(fig, "fig1_ranking")


# ------------------------------------------------------------------ Fig. 2
def fig2_optimism():
    T = pd.read_csv(A / "optimism.csv")
    models = list(MODEL_COLOR)
    fig, axes = plt.subplots(2, 5, figsize=(6.6, 3.15), sharex=True, sharey=True)
    for i, task in enumerate(("B2", "B5")):
        for j, m in enumerate(models):
            ax = axes[i, j]
            g = T[(T.task == task) & (T.config == m)].sort_values("N")
            x = _xpos(g.N)
            ax.fill_between(x, g.null_q25, g.null_q75, color=NULL_FILL, lw=0, zorder=1)
            ax.axhline(0, color=AXIS, lw=0.6, zorder=2)
            ax.axvline(2.5, color=AXIS, lw=0.5, zorder=2)
            c = MODEL_COLOR[m]
            _line(ax, g.N, g.honest_median, c)
            lp = g.dropna(subset=["lpo_median"])
            if len(lp):
                _line(ax, lp.N, lp.lpo_median, c, ls=(0, (1, 1.6)), filled=False, lw=1.1)
            lk = g.dropna(subset=["leaky_median"])
            if len(lk):
                _line(ax, lk.N, lk.leaky_median, c, ls=(0, (4, 2)), marker=None, lw=1.1)
            _style(ax)
            if j == len(models) - 1:
                ax.text(1.06, 0.5, TASK_SHORT[task], transform=ax.transAxes, rotation=270, ha="left", va="center",
                        fontsize=7, color=INK)
            if i == 0:
                ax.set_title(MODEL_LABEL[m].split(" (")[0] if m != "FUS" else "Fusion", color=INK)
            if i == 1:
                ax.set_xlabel("N")
            ax.set_ylim(-0.72, 0.52)
            ax.set_xticklabels([str(n) for n in GRID_N], fontsize=5.8)
    axes[0, 0].annotate("LOOCV", (2.5, 0.48), xytext=(-2, 0), textcoords="offset points", ha="right", va="top",
                        fontsize=5.6, color=MUTED)
    axes[0, 0].annotate("k-fold", (2.5, 0.48), xytext=(2, 0), textcoords="offset points", ha="left", va="top",
                        fontsize=5.6, color=MUTED)
    handles = [Line2D([], [], color=INK2, lw=LW, marker="o", ms=MS, mec="white", label="honest (primary estimator)"),
               Line2D([], [], color=INK2, lw=1.1, ls=(0, (1, 1.6)), marker="o", ms=MS, mfc="white", label="leave-pair-out (N <= 12)"),
               Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 2)), label="filter fitted before CV (leaky)"),
               Patch(color=NULL_FILL, label="permuted labels, honest estimator (IQR)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.035), handlelength=2.2)
    fig.supylabel("Optimism = internal AUC - external AUC", fontsize=7, color=INK2, x=0.055)
    fig.subplots_adjust(wspace=0.1, hspace=0.18)
    return _save(fig, "fig2_optimism")


# ------------------------------------------------------------------ Fig. 3 (appendix)
def fig3_learning():
    df = pd.read_parquet(RESULTS / "main_all_runs.parquet")
    df = df[df.config.isin(MODEL_COLOR)]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2), sharey=True)
    for ax, task in zip(axes, ("B1", "B2", "B5")):
        d = df[df.task == task]
        null = d[d.arm == "perm"].groupby("N").auc_ext.quantile([0.25, 0.75]).unstack()
        x = _xpos(null.index)
        ax.fill_between(x, null[0.25], null[0.75], color=NULL_FILL, lw=0, zorder=1)
        for m in MODEL_COLOR:
            g = d[(d.arm == "true") & (d.config == m)].groupby("N").auc_ext.median()
            _line(ax, g.index, g.values, MODEL_COLOR[m], label=MODEL_LABEL[m])
        _style(ax, "Median external AUC" if task == "B1" else None)
        ax.set_title(TASK_SHORT[task], loc="left")
        ax.set_xlabel("N")
        ax.set_ylim(0.0, 1.03)
    h, l = axes[0].get_legend_handles_labels()
    h.append(Patch(color=NULL_FILL)); l.append("permuted labels (IQR)")
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.1), handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.12)
    return _save(fig, "fig3_learning_curves")


# ------------------------------------------------------------------ Fig. 4
def fig4_recovery():
    R = pd.read_csv(A / "recovery.csv")
    models = ["CCP", "EN", "RF", "FUS"]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.2), sharex=True)
    for j, task in enumerate(("B1", "B2", "B5")):
        for i, (col, ylab) in enumerate((("p50", "precision@50"), ("rbo", "RBO vs full-data ranking"))):
            ax = axes[i, j]
            null = R[(R.task == task) & (R.arm == "perm") & R.config.isin(models)].groupby("N")[col].max()
            ax.fill_between(_xpos(null.index), 0, null.values, color=NULL_FILL, lw=0, zorder=1)
            for m in models:
                g = R[(R.task == task) & (R.arm == "true") & (R.config == m)].sort_values("N")
                _line(ax, g.N, g[col], MODEL_COLOR[m], label=MODEL_LABEL[m])
            _style(ax, ylab if j == 0 else None)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(TASK_SHORT[task] + ("" if task == "B1" else "  (own y-scale)"), loc="left")
            if i == 1:
                ax.set_xlabel("N")
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(Patch(color=NULL_FILL)); l.append("permuted labels (max over methods)")
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.22, hspace=0.12)
    return _save(fig, "fig4_recovery")


# ------------------------------------------------------------------ Fig. 5 (appendix)
def fig5_stability_recovery():
    S = pd.read_parquet(RESULTS / "stability.parquet")
    f21 = json.loads((RESULTS / "stability_f21.json").read_text())
    S = S[(S.arm == "true") & (S.N <= 30)]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.25))
    for ax, task in zip(axes, ("B1", "B2", "B5")):
        g = S[S.task == task]
        for N in sorted(g.N.unique()):
            h = g[g.N == N]
            ax.scatter(h.kuncheva, h.p50_mean, s=16, color=N_RAMP[N], edgecolor="white", linewidth=0.6, zorder=3,
                       label=f"N = {N}")
        ax.set_title(TASK_SHORT[task], loc="left")
        ax.set_xlabel("Selection stability (Kuncheva, top-50)")
        if task == "B1":
            ax.set_ylabel("Recovery (precision@50)")
        ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        reg = f21["registered"][task]["rho"]; fx = f21["fixed_n_diagnostic"][task]["rho_fixed_N"]
        # empty corner per panel, chosen by looking at the rendered data
        x, y, ha, va = {"B1": (0.03, 0.97, "left", "top"), "B2": (0.97, 0.97, "right", "top"),
                        "B5": (0.97, 0.04, "right", "bottom")}[task]
        ax.text(x, y, f"ρ pooled over N = {reg:.2f}\nρ at fixed N = {fx:.2f}",
                transform=ax.transAxes, ha=ha, va=va, fontsize=6.0, color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08),
               handletextpad=0.2, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.28)
    return _save(fig, "fig5_stability_vs_recovery")


if __name__ == "__main__":
    for f in (fig1_ranking, fig2_optimism, fig3_learning, fig4_recovery, fig5_stability_recovery):
        print("wrote", f())
