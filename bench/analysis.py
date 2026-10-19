"""Registered analyses (spec §6, §16 A8). Outputs go to results/analysis/.

    python -m bench.analysis rankings | optimism | usable_n | recovery | decision

All analyses use the true-label arm unless stated. No p-value is computed
across repeats (§6.0, F17): repeats are summarised descriptively.

Paired decision rule (§6.0): A outranks B at (task, N) iff the paired difference
d = AUC_ext(A) - AUC_ext(B) on the same draws has median > 0.02 and P(d > 0) >= 0.75.
A claim generalises only if it holds in every task eligible at that N; under A8
no ranking claim can generalise across all three tasks while B1 is at ceiling,
so a secondary analysis over B2 and B5 is reported and labelled as such.
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

from .paths import RESULTS

OUT = RESULTS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
MARGIN, PROB = 0.02, 0.75
SET_I = ["CCP", "EN", "RF", "TabPFN", "FUS"]
SET_II = {"EN": ["EN-var", "EN-drop", "EN-path", "EN-smote"],
          "RF": ["RF-var", "RF-drop", "RF-path", "RF-smote"]}
SECONDARY = ("B2", "B5")


def load(arm="true"):
    df = pd.read_parquet(RESULTS / "main_all_runs.parquet")
    return df[df.arm == arm] if arm else df


def wide(df, values="auc_ext"):
    return df.pivot_table(index=["task", "N", "r"], columns="config", values=values)


# --------------------------------------------------------------- 1. rankings
def paired(W, a, b):
    d = (W[a] - W[b]).dropna()
    return dict(median_d=float(d.median()), p_pos=float((d > 0).mean()), p_neg=float((d < 0).mean()), n=len(d),
                outranks=bool(d.median() > MARGIN and (d > 0).mean() >= PROB),
                outranked=bool(d.median() < -MARGIN and (d < 0).mean() >= PROB))


def kendall_w(R):
    """Kendall's W for an [m raters x n objects] matrix of ranks (ties corrected)."""
    R = np.asarray(R, float)
    m, n = R.shape
    if m < 2:
        return np.nan
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum()
    T = 0.0
    for row in R:
        _, c = np.unique(row, return_counts=True)
        T += (c ** 3 - c).sum()
    denom = m ** 2 * (n ** 3 - n) - m * T
    return float(12 * S / denom) if denom > 0 else np.nan


def rankings():
    df = load("true")
    W = wide(df)
    tasks_at = df.groupby("N").task.unique().apply(sorted).to_dict()

    # set (i): pairwise table and mean ranks
    pairs = []
    for (task, N), g in W.groupby(level=["task", "N"]):
        for a, b in itertools.permutations(SET_I, 2):
            pairs.append(dict(set="i", task=task, N=N, A=a, B=b, **paired(g, a, b)))
    # set (ii): each variant vs its own reference
    for (task, N), g in W.groupby(level=["task", "N"]):
        for ref, variants in SET_II.items():
            for v in variants:
                pairs.append(dict(set="ii", task=task, N=N, A=v, B=ref, **paired(g, v, ref)))
    P = pd.DataFrame(pairs)

    # per-repeat ranks within set (i): 1 = best; ties take the average rank
    ranks = W[SET_I].rank(axis=1, ascending=False, method="average")
    mr = ranks.groupby(level=["task", "N"]).mean()
    iqr = ranks.groupby(level=["task", "N"]).quantile(0.75) - ranks.groupby(level=["task", "N"]).quantile(0.25)

    # Kendall's W of mean ranks across tasks at each N: all eligible tasks, and B2+B5 (secondary)
    kw = []
    for N in sorted(mr.index.get_level_values("N").unique()):
        for label, tasks in (("all eligible", tasks_at[N]), ("B2+B5 (secondary)", [t for t in SECONDARY if t in tasks_at[N]])):
            M = mr.loc[[(t, N) for t in tasks]]
            kw.append(dict(N=N, tasks=label, n_tasks=len(tasks), W=kendall_w(M.rank(axis=1).to_numpy())))
    KW = pd.DataFrame(kw)

    # claims: "A outranks B" must hold in every eligible task (registered) or in BOTH
    # B2 and B5 (secondary, A8). A single eligible task (B5 alone at N = 50) is not a claim.
    # Set (ii) is tested in both directions (variant over reference, reference over variant).
    rev = P[P.set == "ii"].rename(columns={"A": "B", "B": "A"}).assign(
        median_d=lambda d: -d.median_d, outranks=lambda d: d.outranked)
    both = pd.concat([P, rev], ignore_index=True)
    claims = []
    for (s, N, a, b), g in both.groupby(["set", "N", "A", "B"]):
        reg = sorted(g.task) == tasks_at[N] and bool(g.outranks.all())
        sec_g = g[g.task.isin(SECONDARY)]
        sec = len(sec_g) == 2 and bool(sec_g.outranks.all())
        holds = ",".join(g.task[g.outranks])
        if holds:
            claims.append(dict(set=s, N=N, A=a, B=b, registered_generalises=reg, secondary_B2_and_B5=sec,
                               holds_in=holds, median_d={t: round(v, 3) for t, v in zip(g.task, g.median_d)}))
    C = pd.DataFrame(claims)

    P.to_csv(OUT / "rank_pairs.csv", index=False)
    mr.round(3).to_csv(OUT / "rank_mean_set_i.csv"); iqr.to_csv(OUT / "rank_iqr_set_i.csv")
    KW.to_csv(OUT / "rank_kendall_w.csv", index=False); C.to_csv(OUT / "rank_claims.csv", index=False)
    return P, mr, iqr, KW, C


# ------------------------------------------------------- 2. usable-N (F22)
USABLE = 0.05


def optimism_frame(df):
    d = df.copy()
    d["O"] = d.auc_int - d.auc_ext
    d["O_leaky"] = d.auc_int_leaky - d.auc_ext
    d["O_lpo"] = d.auc_int_lpo - d.auc_ext
    return d


def _first_stable(meds, ok):
    """Smallest N such that ok(median) holds at that N and every larger N run."""
    Ns = sorted(meds)
    for i, N in enumerate(Ns):
        if all(ok(meds[M]) for M in Ns[i:]):
            return N
    return None


def usable_n():
    d = optimism_frame(load("true"))
    med = d.groupby(["task", "config", "N"]).O.median()
    rows = []
    for (task, cfg), s in med.groupby(level=["task", "config"]):
        meds = {N: v for (_, _, N), v in s.items()}
        top = max(meds)
        reg = _first_stable(meds, lambda m: m < USABLE)                 # registered (one-sided)
        diag = _first_stable(meds, lambda m: abs(m) < USABLE)           # diagnostic (two-sided)
        rows.append(dict(task=task, config=cfg, registered_usable_N=reg if reg else f"> {top}",
                         diagnostic_two_sided=diag if diag else f"> {top}",
                         median_O_at_N6=round(meds[min(meds)], 3), median_O_at_max=round(meds[top], 3)))
    U = pd.DataFrame(rows)
    U.to_csv(OUT / "usable_n.csv", index=False)
    return U


# ----------------------------------------------------- 3. optimism (§6.1)
def optimism():
    """Median, IQR and P(O > 0.1) per (task, config, N) for four variants:
    honest (primary estimator), leaky (filter on all of S_r), LPO (N <= 12), and the
    permuted-label null of the honest estimator. Reported per task = effect-size
    stratum (F16); task effect sizes from bench.references."""
    d = optimism_frame(load(None))
    rows = []
    for (task, cfg, N), g in d.groupby(["task", "config", "N"]):
        t, p = g[g.arm == "true"], g[g.arm == "perm"]
        r = dict(task=task, config=cfg, N=N)
        for name, v in (("honest", t.O), ("leaky", t.O_leaky), ("lpo", t.O_lpo), ("null", p.O)):
            v = v.dropna()
            if v.empty:
                continue
            r.update({f"{name}_median": v.median(), f"{name}_q25": v.quantile(0.25), f"{name}_q75": v.quantile(0.75),
                      f"{name}_p_gt_0.1": (v > 0.1).mean()})
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "optimism.csv", index=False)
    return T


# ----------------------------------------------------- 4. recovery (§6.2)
REC = ["p10", "p20", "p50", "p100", "p50_wil", "rbo", "rbo_wil", "dir50"]


def recovery():
    """Descriptive only (A8): B2's reference has no FDR-significant protein and B1 is a
    single task, so no recovery claim is made. Permuted-label recovery is the empirical
    null; the analytic chance level of precision@k is k / n_proteins."""
    from . import loaders
    d = load(None).dropna(subset=["p50"])
    n_prot = {t: loaders.load(t).p for t in d.task.unique()}
    g = d.groupby(["task", "config", "N", "arm"])[REC].mean().reset_index()
    g["chance_p50"] = g.task.map(lambda t: 50 / n_prot[t])
    g.to_csv(OUT / "recovery.csv", index=False)
    return g


# --------------------------------------------------- 5. decision table (§7)
def decision():
    """The paper's decision table (Table 3), rebuilt from the saved analyses. Rows come from
    the two weak-signal tasks (B2, B5) except the last (B1). Evidence: secondary claim =
    paired rule holds in both B2 and B5 (a post hoc relaxation of the pre-specified rule);
    descriptive = measured and consistent in both, no pre-specified decision rule; none =
    no basis to choose. Needs `python -m bench.review all` and `python -m bench.stability`."""
    small = ["B2", "B5"]
    base = ["CCP", "EN", "RF", "TabPFN"]
    T = pd.read_csv(OUT / "optimism.csv")
    T = T[T.task.isin(small)]
    C = pd.read_csv(OUT / "rank_claims.csv")
    sec = C[C.secondary_B2_and_B5]
    F = pd.read_csv(OUT / "review_fus_lpo.csv")
    K = pd.read_csv(OUT / "review_kfold_tail_pooled.csv")
    SM = pd.read_csv(OUT / "review_smote_split.csv")
    E = pd.read_csv(OUT / "review_effect_null.csv")
    S = pd.read_parquet(RESULTS / "stability.parquet")
    f2 = lambda x: f"{x:.2f}".replace("-0.00", "0.00")
    sg = lambda x: f"{x:+.2f}"

    loo = T[T.config.isin(base + ["FUS"]) & (T.N <= 12)].honest_median
    lpo = pd.concat([T[T.config.isin(base) & (T.N <= 12)].lpo_median,
                     F[(F.arm == "true") & F.task.isin(small)].O_lpo_median])
    kp = K[K.N <= 30].p_gt_01
    lk = T[T.config.isin(base)].leaky_median
    sm = SM[(SM.arm == "true") & SM.task.isin(small) & (SM.N >= 12)].smote_extra
    fus = sec[sec.B == "FUS"]
    first = fus.groupby("A").N.min()
    by_n = {}
    for a, n in first.items():
        by_n.setdefault(n, []).append(a)
    order = ", ".join(" and ".join(sorted(by_n[n])) + f" from {'N = ' if i == 0 else ''}{n}"
                      for i, n in enumerate(sorted(by_n)))
    marg = [v for d in fus.median_d.map(eval) for t, v in d.items() if t in small]
    pa = sec[sec.B.str.endswith("-path")].groupby("B").N.apply(sorted)
    st = S[(S.arm == "true") & S.task.isin(small) & (S.N <= 30) & ~S.config.str.contains("-var")
           & (S.config != "EN-smote-leaky") & (S.config != "RF-smote-leaky")].kuncheva
    d_b1 = E[(E.task == "B1") & (E.scope == "full") & (E.arm == "true")].median_abs_d.iloc[0]

    rows = [
        ("<= 12", "Internal estimate",
         f"Report leave-pair-out AUC, not pooled LOOCV (errors {sg(loo.min())} to {sg(loo.max())} "
         f"vs. +/-{f2(lpo.abs().max())})", "descriptive"),
        (">= 20", "Internal estimate",
         f"Repeated k-fold, AUC per fold; expect >0.1 overstatement in about one study in five "
         f"({100 * kp.min():.0f}-{100 * kp.max():.0f}% at N = 20-30)", "descriptive"),
        ("all", "Own null",
         "Re-run the whole pipeline on permuted labels with leave-pair-out or per-fold k-fold; a mean well "
         "above 0.5 exposes a leak (at 3 vs 3 only 20 labellings exist, so a permutation p cannot fall below "
         "0.05, or 0.1 for label-symmetric pipelines)", "descriptive"),
        ("all", "Feature selection",
         f"Inside the folds only (outside: {sg(lk.min())} to {sg(lk.max())} optimism)", "descriptive"),
        ("all", "Augmentation",
         f"Never before the split (SMOTE adds {f2(sm.min())}-{f2(sm.max())} from N = 12 on top of other leaks); "
         "inside the folds it does not help", "descriptive"),
        (f"{sorted(fus.N.unique())[0]}-{sorted(fus.N.unique())[-1]}", "Model",
         f"Avoid the fusion network ({order}; by {f2(min(marg))}-{f2(max(marg))} AUC)", "secondary claim"),
        ("6-30", "Model", "No basis to choose among CCP, elastic net, random forest, TabPFN", "none"),
        (f"{pa['RF-path'][0]}-{pa['RF-path'][-1]}", "Features",
         f"With RF, prefer proteins to Reactome pathway scores (EN: N = {', '.join(map(str, pa['EN-path']))} only)",
         "secondary claim"),
        ("<= 30", "Biomarker list",
         f"Do not read the top 50 as a biomarker list (Kuncheva {f2(st.min())}-{f2(st.max())})", "descriptive"),
        ("<= 30", "Stability", "Not a proxy for recovery at a fixed N, and high even with no signal", "none"),
        ("any", "Large effect",
         f"Any method reaches AUC 1.00 from N = 6 (B1, full-cohort |d| ~ {d_b1:.0f}); no basis to choose", "none"),
    ]
    D = pd.DataFrame(rows, columns=["N", "Question", "Recommendation", "Evidence"])
    D.to_csv(OUT / "decision_table.csv", index=False)
    md = ["Decision table (paper Table 3). Rows from the weak-signal tasks B2 and B5, except the last (B1).", "",
          "| " + " | ".join(D.columns) + " |", "|" + "---|" * len(D.columns)]
    md += ["| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |" for r in D.itertuples(index=False)]
    (OUT / "decision_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return D

if __name__ == "__main__":
    pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    what = sys.argv[1] if len(sys.argv) > 1 else "rankings"
    if what == "decision":
        pd.set_option("display.max_colwidth", 170)
        D = decision()
        print(D.to_string(index=False))
    if what == "recovery":
        g = recovery()
        show = ["CCP", "EN", "RF", "FUS", "EN-var", "RF-var"]
        t, p = g[g.arm == "true"], g[g.arm == "perm"]
        print("chance level of precision@50 (50 / proteins):", g.groupby("task").chance_p50.first().round(4).to_dict())
        for col, title in (("p50", "precision@50 vs limma reference, true labels (mean over repeats)"),
                           ("p50_wil", "precision@50 vs Wilcoxon reference, true labels"),
                           ("rbo", "RBO vs limma ranking (depth 500, p = 0.98), true labels"),
                           ("dir50", "directional agreement of recovered proteins, true labels")):
            print(f"\n{title}")
            print(t[t.config.isin(show)].pivot_table(index=["task", "config"], columns="N", values=col).round(3).to_string())
        print("\nprecision@50, PERMUTED labels (empirical null)")
        print(p[p.config.isin(show)].pivot_table(index=["task", "config"], columns="N", values="p50").round(3).to_string())
    if what == "optimism":
        T = optimism()
        show = ["CCP", "EN", "RF", "TabPFN", "FUS"]
        for col, title in (("honest_median", "honest, primary estimator (pooled LOOCV N<=12, k-fold N>=20): median O"),
                           ("honest_p_gt_0.1", "honest: P(O > 0.1)"),
                           ("lpo_median", "leave-pair-out (N<=12): median O"),
                           ("leaky_median", "leaky filter: median O"),
                           ("null_median", "permuted-label null, honest estimator: median O")):
            print(f"\n{title}")
            print(T[T.config.isin(show)].pivot_table(index=["task", "config"], columns="N", values=col).round(2).to_string())
        sm = T[T.config.str.contains("smote")].pivot_table(index=["task", "config"], columns="N", values="honest_median").round(2)
        print("\nSMOTE in-fold vs before the split: median honest-internal O"); print(sm.to_string())
    if what == "usable_n":
        U = usable_n()
        print("Usable-N: smallest N at which median optimism < 0.05 and stays so at every larger N (true labels,")
        print("primary internal estimator). 'registered' = one-sided as pre-registered (F22); 'diagnostic' = |median| < 0.05.")
        print(U.pivot_table(index="config", columns="task", values="registered_usable_N", aggfunc="first").to_string())
        print("\ndiagnostic (two-sided):")
        print(U.pivot_table(index="config", columns="task", values="diagnostic_two_sided", aggfunc="first").to_string())
        print("\nmedian optimism at N = 6 (primary estimator):")
        print(U.pivot_table(index="config", columns="task", values="median_O_at_N6", aggfunc="first").to_string())
    if what == "rankings":
        P, mr, iqr, KW, C = rankings()
        print("Mean rank within set (i), true labels (1 = best of 5; ties averaged)")
        print(mr.round(2).to_string())
        print("\nIQR of per-repeat rank")
        print(iqr.round(1).to_string())
        print("\nKendall's W across tasks (1 = same ordering in every task)")
        print(KW.round(3).to_string(index=False))
        print("\nPairs meeting the outranking rule in at least one task (set i):")
        hit = P[(P.set == "i") & P.outranks]
        print(hit[["task", "N", "A", "B", "median_d", "p_pos"]].round(3).to_string(index=False))
        print("\nSet (ii): variant vs its reference -- outranks (+) / outranked (-) per task and N")
        q = P[P.set == "ii"].assign(v=lambda d: np.where(d.outranks, "+", np.where(d.outranked, "-", ".")))
        print(q.pivot_table(index=["A", "B"], columns=["task", "N"], values="v", aggfunc="first").to_string())
        print("\nRegistered claims (hold in every eligible task):", int(C.registered_generalises.sum()))
        sec = C[C.secondary_B2_and_B5]
        print("Secondary claims (hold in BOTH B2 and B5; A8):")
        print(sec[["set", "N", "A", "B", "median_d"]].to_string(index=False) if len(sec) else "  none")
        one = C[~C.secondary_B2_and_B5 & ~C.registered_generalises]
        print(f"\nSingle-task results (NOT claims): {len(one)} rows")
        print(one.groupby(["set", "A", "B", "holds_in"]).N.apply(lambda x: ",".join(map(str, sorted(x)))).to_string())
