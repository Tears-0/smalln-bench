"""Re-run sample blocks of every runner from scratch and compare with the stored rows.

Blocks are re-run exactly as in the main run (same units per block, so the fusion
network's batched trainer sees the same batches) into results/runs/repro/, then
compared value by value with results/main_all_runs.parquet. One block per runner and
code version: the CPU rows all carry c8068f7; TabPFN and the fusion network have
blocks under both c8068f7 and 748dcfe.

    python -m bench.reproduce      -> results/analysis/reproduce.csv, results/logs/reproduce.log
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import fusion, gpu
from .configs import CONFIGS
from .paths import RESULTS, RUNS
from .run import do_block

OUT = RUNS / "repro"
BLOCKS = [  # (runner, task, N, arm, first repeat, block size) -- block sizes as in the main run
    ("cpu", "B2", 12, "true", 0, 10),
    ("cpu", "B5", 30, "perm", 50, 10),
    ("tabpfn", "B1", 12, "perm", 0, 10),    # c8068f7 (leakage cells)
    ("tabpfn", "B2", 6, "perm", 0, 10),     # 748dcfe
    ("fus", "B1", 6, "perm", 0, 50),        # c8068f7
    ("fus", "B2", 6, "perm", 0, 50),        # 748dcfe
]
VALUES = ["auc_int", "auc_int_lpo", "auc_int_leaky", "auc_ext", "n_feat", "p10", "p20", "p50", "p100",
          "p10_wil", "p20_wil", "p50_wil", "p100_wil", "rbo", "rbo_wil", "dir50"]
KEY = ["task", "N", "r", "arm", "config_id"]


def rerun(runner, task, N, arm, r0, size):
    d = OUT / runner
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{task}_N{N}_{arm}_r{r0:03d}.parquet"
    if not f.exists():
        if runner == "fus":
            gpu.guard("fusion")
            fusion.run_cell(task, N, [(r, arm) for r in range(r0, r0 + size)], f)
        else:
            if runner == "tabpfn":
                gpu.guard("tabpfn")
            do_block(task, N, arm, r0, r0 + size, f, [c.id for c in CONFIGS if c.runner == runner])
    return pd.read_parquet(f)


def compare(new, ref):
    m = new.merge(ref, on=KEY, suffixes=("", "_ref"), validate="one_to_one")
    diffs = {}
    for c in VALUES:
        if c not in new.columns:                 # TabPFN rows carry no recovery columns
            assert ref.loc[ref.config.isin(new.config.unique()), c].isna().all()
            continue
        a, b = m[c].to_numpy(float), m[c + "_ref"].to_numpy(float)
        same_nan = np.isnan(a) == np.isnan(b)
        diffs[c] = np.inf if not same_nan.all() else float(np.nanmax(np.abs(a - b), initial=0.0))
    if "top100" in new.columns:
        top = all((x is None and y is None) or (x is not None and y is not None and np.array_equal(x, y))
                  for x, y in zip(m.top100, m.top100_ref))
    else:                                        # no protein ranking: stored rows must have none either
        top = bool(m.top100.isna().all()) if "top100" in m.columns else True
    return len(m), max(diffs.values()), top, sorted(m.code_ref.unique())


def run():
    ref = pd.read_parquet(RESULTS / "main_all_runs.parquet")
    rows, log = [], []
    for runner, task, N, arm, r0, size in BLOCKS:
        t0 = time.time()
        new = rerun(runner, task, N, arm, r0, size)
        n, maxdiff, top, codes = compare(new, ref)
        rows.append(dict(runner=runner, task=task, N=N, arm=arm, repeats=f"{r0}-{r0 + size - 1}", rows=n,
                         stored_code=",".join(codes), max_abs_diff=maxdiff, top100_identical=top,
                         seconds=round(time.time() - t0)))
        log.append(f"{time.strftime('%F %T')} {runner} {task} N={N} {arm} r{r0}-{r0 + size - 1}: {n} rows, "
                   f"stored code {','.join(codes)}, max |diff| {maxdiff:.3g}, top100 identical {top}")
        print(log[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "analysis" / "reproduce.csv", index=False)
    (RESULTS / "logs" / "reproduce.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    return R


if __name__ == "__main__":
    print(run().to_string(index=False))
