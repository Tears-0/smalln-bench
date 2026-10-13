"""Task loaders (spec §5.0, §8 loader contract; tasks fixed in §16 A3).

Every loader returns a Task: X (samples x proteins, log2, NaN for missing), y
(binary), unit (the draw unit: patient or donor), protein identifiers, and a
provenance dict. Built tasks are cached in data/derived/task_<name>.npz so that
worker processes never touch the network or the cptac package.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import reference as ref
from .paths import DERIVED, RAW

TASKS = ("B1", "B2", "B5")
NONSILENT = re.compile(r"Missense|Nonsense|Frame_Shift|In_Frame|Splice|Nonstop|Translation_Start", re.I)


@dataclass
class Task:
    name: str
    X: np.ndarray            # float32 [n_samples, n_proteins], log2, NaN = missing
    y: np.ndarray            # int8 [n_samples], 1 = positive class
    unit: np.ndarray         # str [n_samples], patient / donor id
    sample_ids: np.ndarray   # str [n_samples]
    protein_ids: np.ndarray  # str [n_proteins], ENSP (CPTAC) or UniProt accession (ProCan)
    gene: np.ndarray         # str [n_proteins]
    uniprot: np.ndarray      # str [n_proteins], reviewed accession or ''
    sequence: np.ndarray | None  # object [n_proteins] or None unless load(..., sequences=True)
    provenance: dict = field(default_factory=dict)
    has_seq: np.ndarray | None = None  # bool [n_proteins]

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def p(self):
        return self.X.shape[1]

    def class_counts(self):
        return int((self.y == 1).sum()), int((self.y == 0).sum())


# --------------------------------------------------------------------- builders
def _cptac(name, source):
    import cptac
    warnings.filterwarnings("ignore")
    ds = getattr(cptac, name)()
    return ds, ds.get_proteomics(source)


def _cptac_proteins(P):
    genes = P.columns.get_level_values(0).astype(str).to_numpy()
    ensp = P.columns.get_level_values(1).astype(str).to_numpy()
    g2u, up, pep = ref.gene_to_uniprot(), ref.uniprot(), ref.ensembl_peptides()
    acc = np.array([g2u.get(g, "") for g in genes], dtype=object)
    seq = []
    n_ens = 0
    for e, a in zip(ensp, acc):
        s = pep.get(e.split(".")[0], "")
        if s:
            n_ens += 1
        elif a:
            s = up[a][2]
        seq.append(s)
    return ensp, genes, acc, np.array(seq, dtype=object), n_ens


def build_B1():
    """CPTAC LUAD, tumour (1) vs NAT (0), unit = patient. umich proteomics."""
    ds, P = _cptac("Luad", "umich")
    idx = P.index.astype(str)
    is_nat = idx.str.endswith(".N")
    unit = idx.str.replace(r"\.N$", "", regex=True).to_numpy()
    ensp, genes, acc, seq, n_ens = _cptac_proteins(P)
    return Task(
        name="B1", X=P.to_numpy(dtype=np.float32), y=(~is_nat).astype(np.int8), unit=unit,
        sample_ids=idx.to_numpy(), protein_ids=ensp, gene=genes, uniprot=acc, sequence=seq,
        provenance=dict(source="cptac 1.5.14, Luad, proteomics source 'umich'",
                        contrast="tumour (1) vs NAT (0)", unit="patient (sample id without '.N')",
                        scale="log2 TMT ratio, as distributed", sequences_from_ensembl=int(n_ens)))


def build_B2():
    """CPTAC PDAC tumours, TP53 non-silent mutant (1) vs wild-type (0); washu calls."""
    ds, P = _cptac("Pdac", "umich")
    idx = P.index.astype(str)
    T = P[~idx.str.endswith(".N")]
    sm = ds.get_somatic_mutation("washu")
    sequenced = set(sm.index.astype(str))
    mutant = set(sm.index[(sm["Gene"] == "TP53") & sm["Mutation"].astype(str).str.contains(NONSILENT)].astype(str))
    T = T[T.index.astype(str).isin(sequenced)]
    tid = T.index.astype(str)
    ensp, genes, acc, seq, n_ens = _cptac_proteins(T)
    return Task(
        name="B2", X=T.to_numpy(dtype=np.float32), y=np.array([s in mutant for s in tid], dtype=np.int8),
        unit=tid.to_numpy(), sample_ids=tid.to_numpy(), protein_ids=ensp, gene=genes, uniprot=acc, sequence=seq,
        provenance=dict(source="cptac 1.5.14, Pdac, proteomics 'umich', somatic_mutation 'washu' "
                               "(the 'harmonized' table is empty for Pdac in 1.5.14)",
                        contrast="TP53 non-silent mutant (1) vs wild-type (0), sequenced tumours only",
                        unit="patient", scale="log2 TMT ratio, as distributed",
                        sequences_from_ensembl=int(n_ens)))


def build_B5():
    """ProCan-DepMapSanger, haematopoietic & lymphoid lines, TP53 driver mutant (1) vs wt (0)."""
    m = pd.read_csv(RAW / "ProCan-DepMapSanger_protein_matrix_8498_averaged.txt", sep="\t", low_memory=False)
    m.index = m.pop("Project_Identifier").str.split(";").str[0]
    ml = pd.read_csv(RAW / "model_list_20261010.csv", low_memory=False).set_index("model_id")
    calls = pd.read_csv(DERIVED / "kras_tp53_calls.csv", dtype=str)
    sequenced = set(pd.read_csv(DERIVED / "models_with_mutation_data.csv")["model_id"])
    tp53 = set(calls.loc[(calls.gene_symbol == "TP53") & (calls.cancer_driver == "t"), "model_id"])
    keep = [i for i in m.index if i in ml.index and ml.loc[i, "tissue"] == "Haematopoietic and Lymphoid"
            and i in sequenced]
    T = m.loc[keep]
    cols = T.columns.astype(str)
    raw_acc = cols.str.split(";").str[0].to_numpy()
    entry = cols.str.split(";").str[1].to_numpy()
    up, e2u = ref.uniprot(), ref.entry_to_uniprot()
    acc, genes, seq = [], [], []
    for a, e in zip(raw_acc, entry):
        a0 = a.split("-")[0]
        a_ok = a0 if a0 in up else e2u.get(e, "")
        acc.append(a_ok)
        genes.append(up[a_ok][0] if a_ok else e.replace("_HUMAN", ""))
        seq.append(up[a_ok][2] if a_ok else "")
    units = ml.loc[keep, "patient_id"].astype(str).to_numpy()
    return Task(
        name="B5", X=T.to_numpy(dtype=np.float32), y=np.array([i in tp53 for i in keep], dtype=np.int8),
        unit=units, sample_ids=np.array(keep), protein_ids=raw_acc, gene=np.array(genes, dtype=object),
        uniprot=np.array(acc, dtype=object), sequence=np.array(seq, dtype=object),
        provenance=dict(source="ProCan-DepMapSanger_protein_matrix_8498_averaged.txt (figshare 19345397); "
                               "model_list_20261010.csv; mutations_all_20260724 (Cell Model Passports)",
                        contrast="TP53 cancer-driver mutant (1) vs wild-type (0), tissue = Haematopoietic and Lymphoid",
                        unit="Cell Model Passports patient_id (donor)", scale="log2, as distributed"))


BUILDERS = {"B1": build_B1, "B2": build_B2, "B5": build_B5}


# ------------------------------------------------------------------ cache I/O
def _path(name):
    return DERIVED / f"task_{name}.npz"


def _seq_path(name):
    return DERIVED / f"task_{name}_sequences.json"


def save(t: Task):
    # Sequences live in their own file: as a fixed-width numpy string array they
    # cost ~1.9 GB per process for B1 (width of the longest protein, 35,991 aa),
    # and only bench.embeddings needs them.
    seq = [str(s) for s in t.sequence]
    _seq_path(t.name).write_text(json.dumps(seq))
    np.savez_compressed(
        _path(t.name), X=t.X, y=t.y, unit=t.unit.astype(str), sample_ids=t.sample_ids.astype(str),
        protein_ids=t.protein_ids.astype(str), gene=np.asarray(t.gene).astype(str),
        uniprot=np.asarray(t.uniprot).astype(str), has_seq=np.array([bool(s) for s in seq]))
    (DERIVED / f"task_{t.name}.json").write_text(json.dumps(
        dict(t.provenance, references=ref.versions(), n=t.n, p=t.p, class_counts=t.class_counts(),
             missing=float(np.isnan(t.X).mean()), with_sequence=int((np.asarray(t.sequence) != "").sum()),
             with_uniprot=int((np.asarray(t.uniprot) != "").sum())), indent=2))


def _migrate(name):
    """Older caches stored sequences inside the npz; split them out once."""
    z = np.load(_path(name), allow_pickle=False)
    if "sequence" in z.files:
        prov = json.loads((DERIVED / f"task_{name}.json").read_text())
        t = Task(name=name, X=z["X"], y=z["y"], unit=z["unit"], sample_ids=z["sample_ids"],
                 protein_ids=z["protein_ids"], gene=z["gene"], uniprot=z["uniprot"],
                 sequence=z["sequence"], provenance=prov)
        z.close()
        seq = [str(s) for s in t.sequence]
        _seq_path(name).write_text(json.dumps(seq))
        np.savez_compressed(_path(name), X=t.X, y=t.y, unit=t.unit, sample_ids=t.sample_ids,
                            protein_ids=t.protein_ids, gene=t.gene, uniprot=t.uniprot,
                            has_seq=np.array([bool(s) for s in seq]))


def load(name: str, sequences: bool = False) -> Task:
    if not _path(name).exists():
        save(BUILDERS[name]())
    _migrate(name)
    z = np.load(_path(name), allow_pickle=False)
    prov = json.loads((DERIVED / f"task_{name}.json").read_text())
    seq = np.array(json.loads(_seq_path(name).read_text()), dtype=object) if sequences else None
    return Task(name=name, X=z["X"], y=z["y"], unit=z["unit"], sample_ids=z["sample_ids"],
                protein_ids=z["protein_ids"], gene=z["gene"], uniprot=z["uniprot"],
                sequence=seq, provenance=prov, has_seq=z["has_seq"])
