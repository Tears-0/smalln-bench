"""Task selection under the pre-specified rules, recomputed with the statistic of Table 1.

  B1  the CPTAC cohort with the largest min(#tumour, #NAT) over distinct patients;
      ties go to the larger total number of samples.
  B2  within each CPTAC cohort, tumours only, the candidates {TP53 non-silent mutant vs
      wild-type; AJCC pathological stage I vs III-IV}; eligible if every class has at
      least 35 units (N = 30 draws 15 per class and leaves >= 20 in H_r); the eligible
      contrast with the smallest effect is chosen.
  B5  ProCan-DepMapSanger lines, within each tissue, the candidates {KRAS, TP53
      cancer-driver mutant vs wild-type}, wild-type = sequenced without a driver call;
      units are donors; the same eligibility and choice.

The effect statistic is the one in Table 1 (bench.references): the median |Cohen's d|
over the top 100 proteins ranked by R limma |t| on the full contrast. The original
selection (made once, before any model was fitted) used an earlier implementation of
the statistic; this recomputation checks that the same tasks are chosen.

    python -m bench.selection      -> results/analysis/task_selection.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .loaders import NONSILENT
from .paths import DERIVED, RAW
from .references import _limma_r
from .review import OUT, _med_d_top100

CPTAC = ["Brca", "Ccrcc", "Coad", "Gbm", "Hnscc", "Lscc", "Luad", "Ov", "Pdac", "Ucec"]
SOURCE = {"Coad": "bcm"}          # umich COAD fails to download in cptac 1.5.14
MIN_CLASS = 35                    # N = 30: 15 per class drawn + 20 per class held out


def _stat(X, y):
    return _med_d_top100(X, y, _limma_r(X, y)[0])


def cptac_candidates():
    import cptac
    warnings.filterwarnings("ignore")
    rows = []
    for name in CPTAC:
        try:
            ds = getattr(cptac, name)()
            P = ds.get_proteomics(SOURCE.get(name, "umich"))
        except Exception as e:                          # record, do not drop silently
            rows.append(dict(task="B1", cohort=name, contrast=f"not loadable: {type(e).__name__}"))
            continue
        idx = P.index.astype(str)
        nat = idx.str.endswith(".N")
        tum_ids, nat_ids = set(idx[~nat]), set(idx[nat].str.replace(r"\.N$", "", regex=True))
        rows.append(dict(task="B1", cohort=name, contrast="tumour vs NAT", n1=len(tum_ids), n0=len(nat_ids),
                         min_class=min(len(tum_ids), len(nat_ids)), total=len(idx)))
        T = P[~nat]
        tid = T.index.astype(str)
        sm = ds.get_somatic_mutation("harmonized")
        msrc = "harmonized"
        if sm.empty:
            sm, msrc = ds.get_somatic_mutation("washu"), "washu"
        seqd = set(sm.index.astype(str))
        mut = set(sm.index[(sm["Gene"] == "TP53") & sm["Mutation"].astype(str).str.contains(NONSILENT)].astype(str))
        keep = np.array([t in seqd for t in tid])
        y = np.array([t in mut for t in tid], dtype=int)[keep]
        cands = [(f"TP53 mutant vs wild-type ({msrc} calls)", T.to_numpy(float)[keep], y)]
        try:
            cl = ds.get_clinical("mssm")
        except Exception:
            cl = pd.DataFrame()
        if "tumor_stage_pathological" in cl.columns:
            st = cl["tumor_stage_pathological"].astype(str)
            stage = {i: ("I" if s == "Stage I" else "III-IV" if s in ("Stage III", "Stage IV") else None)
                     for i, s in st.items()}
            sel = np.array([stage.get(t) is not None for t in tid])
            ys = np.array([stage.get(t) == "I" for t in tid], dtype=int)[sel]
            cands.append(("stage I vs III-IV", T.to_numpy(float)[sel], ys))
        for contrast, X, yy in cands:
            n1, n0 = int((yy == 1).sum()), int((yy == 0).sum())
            r = dict(task="B2", cohort=name, contrast=contrast, n1=n1, n0=n0, min_class=min(n1, n0), total=len(yy))
            if min(n1, n0) >= MIN_CLASS:
                r["median_abs_d"] = _stat(X, yy)
            rows.append(r)
        print(f"{name} done", flush=True)
    return rows


def procan_candidates():
    m = pd.read_csv(RAW / "ProCan-DepMapSanger_protein_matrix_8498_averaged.txt", sep="\t", low_memory=False)
    m.index = m.pop("Project_Identifier").str.split(";").str[0]
    ml = pd.read_csv(RAW / "model_list_20261010.csv", low_memory=False).set_index("model_id")
    calls = pd.read_csv(DERIVED / "kras_tp53_calls.csv", dtype=str)
    seqd = set(pd.read_csv(DERIVED / "models_with_mutation_data.csv")["model_id"])
    lines = [i for i in m.index if i in ml.index and i in seqd]
    rows = []
    for tissue, g in ml.loc[lines].groupby("tissue"):
        ids = list(g.index)
        for gene in ("KRAS", "TP53"):
            mut = set(calls.loc[(calls.gene_symbol == gene) & (calls.cancer_driver == "t"), "model_id"])
            y = np.array([i in mut for i in ids], dtype=int)
            donors = g.patient_id.astype(str).to_numpy()
            n1, n0 = len(set(donors[y == 1])), len(set(donors[y == 0]))
            r = dict(task="B5", cohort=tissue, contrast=f"{gene} driver mutant vs wild-type", n1=n1, n0=n0,
                     min_class=min(n1, n0), total=len(ids))
            if min(n1, n0) >= MIN_CLASS:
                r["median_abs_d"] = _stat(m.loc[ids].to_numpy(float), y)
            rows.append(r)
    return rows


def run():
    S = pd.DataFrame(cptac_candidates() + procan_candidates())
    S["eligible"] = np.where(S.task == "B1", True, S.min_class >= MIN_CLASS)
    chosen = []
    for task, g in S.groupby("task"):
        if task == "B1":
            best = g.sort_values(["min_class", "total"], ascending=False).index[0]
        else:
            best = g[g.eligible].median_abs_d.idxmin()
        chosen.append(best)
    S["chosen"] = S.index.isin(chosen)
    S.to_csv(OUT / "task_selection.csv", index=False)
    return S


if __name__ == "__main__":
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    S = run()
    print(S.round(3).to_string(index=False))
