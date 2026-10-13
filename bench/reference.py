"""External reference resources: protein sequences and Reactome membership.

Files live in data/raw/reference/ (downloaded 2026-10-13; see downloaded_at_utc.txt):
  uniprot_human_reviewed.fasta.gz        UniProtKB/Swiss-Prot, organism 9606
  Homo_sapiens.GRCh38.pep.all.fa.gz      Ensembl release 116 peptides
  UniProt2Reactome.txt                   Reactome release 97, lowest-level pathways
"""
from __future__ import annotations

import gzip
import re
from functools import lru_cache

import numpy as np

from .paths import REFERENCE


def _read_fasta_gz(path):
    header, seq = None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header, seq = line[1:], []
            else:
                seq.append(line)
    if header is not None:
        yield header, "".join(seq)


@lru_cache(maxsize=1)
def uniprot():
    """accession -> (gene symbol or '', entry name, sequence), reviewed human only."""
    out = {}
    for h, s in _read_fasta_gz(REFERENCE / "uniprot_human_reviewed.fasta.gz"):
        # sp|P21333|FLNA_HUMAN Filamin-A OS=Homo sapiens OX=9606 GN=FLNA PE=1 SV=4
        _, acc, entry = h.split(" ")[0].split("|")
        m = re.search(r"\bGN=(\S+)", h)
        out[acc] = (m.group(1) if m else "", entry, s)
    return out


@lru_cache(maxsize=1)
def gene_to_uniprot():
    """gene symbol -> reviewed accession (first in file order; one canonical per gene)."""
    g2a = {}
    for acc, (gene, _, _) in uniprot().items():
        if gene and gene not in g2a:
            g2a[gene] = acc
    return g2a


@lru_cache(maxsize=1)
def entry_to_uniprot():
    return {entry: acc for acc, (_, entry, _) in uniprot().items()}


@lru_cache(maxsize=1)
def ensembl_peptides():
    """Ensembl protein stable ID (no version) -> sequence."""
    out = {}
    for h, s in _read_fasta_gz(REFERENCE / "Homo_sapiens.GRCh38.pep.all.fa.gz"):
        pid = h.split(" ")[0].split(".")[0]
        out[pid] = s.rstrip("*")
    return out


@lru_cache(maxsize=1)
def reactome_membership():
    """UniProt accession -> set of human lowest-level Reactome pathway IDs."""
    m = {}
    with open(REFERENCE / "UniProt2Reactome.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6 or parts[5] != "Homo sapiens":
                continue
            m.setdefault(parts[0], set()).add(parts[1])
    return m


def membership_matrix(uniprot_acc: np.ndarray):
    """Boolean [n_proteins, n_pathways] matrix for a task's protein list.

    Proteins with no accession or no human pathway get an all-false row.
    Returns (matrix, pathway_ids).
    """
    rm = reactome_membership()
    paths = sorted({p for a in uniprot_acc if a for p in rm.get(a, ())})
    col = {p: j for j, p in enumerate(paths)}
    M = np.zeros((len(uniprot_acc), len(paths)), dtype=bool)
    for i, a in enumerate(uniprot_acc):
        for p in rm.get(a, ()) if a else ():
            M[i, col[p]] = True
    return M, np.array(paths)


def versions():
    v = {}
    for name in ("reactome_version.txt", "ensembl_release.json", "downloaded_at_utc.txt"):
        p = REFERENCE / name
        v[name] = p.read_text().strip() if p.exists() else None
    return v
