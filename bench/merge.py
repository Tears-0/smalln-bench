"""Merge the per-block run files into one table.

Each runner writes one parquet file per block under results/runs/<out>/<runner>/
(resumable; see bench.run, bench.fusion). This concatenates them, checks that
every (task, N, r, arm, configuration) appears exactly once, and writes
results/main_all_runs.parquet.

    python -m bench.merge --out main
"""
from __future__ import annotations

import argparse

import pandas as pd

from .paths import RESULTS, RUNS

RUNNERS = ("cpu", "tabpfn", "fus")
KEY = ["task", "N", "r", "arm", "config_id"]


def merge(out="main"):
    parts = [pd.read_parquet(f) for runner in RUNNERS for f in sorted((RUNS / out / runner).glob("*.parquet"))]
    D = pd.concat(parts, ignore_index=True)
    # configurations without a protein ranking (TabPFN) store NaN in some blocks: use null throughout
    D["top100"] = [v if hasattr(v, "__len__") else None for v in D.top100]
    dup = D.duplicated(KEY)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated runs, e.g. {D.loc[dup, KEY].iloc[0].to_dict()}")
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="main")
    a = ap.parse_args()
    D = merge(a.out)
    D.to_parquet(RESULTS / "main_all_runs.parquet")
    print(f"{len(D)} runs -> {RESULTS / 'main_all_runs.parquet'}")
