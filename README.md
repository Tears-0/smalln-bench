# smalln-bench

Code and complete results for the paper *When can a small proteomics study trust its
cross-validated AUC? A null-calibration condition and a pre-specified benchmark*
(submitted to RECOMB 2027).

Large public proteomic cohorts are subsampled to small N (6 to 50), every method is
scored by its own internal cross-validation on the subsample and on the held-out
remainder, and the gap between the two — the *optimism* of the internal estimate — is
measured. Three tasks, 15 configurations, 100 repeats, true and permuted labels:
51,000 runs.

The design (tasks, configurations, metrics, leakage tests, decision rules) was fixed before
the first benchmark run; the paper reports every change made to the design or analysis
after that.

## Layout

| Path | Contents |
|---|---|
| `bench/` | The benchmark: loaders, subsampling, in-fold preprocessing, models, CV estimators, runners, analyses, figures |
| `tests/` | `test_limma.py` (moderated t against R limma); `test_no_leakage.py` (leakage suite; the original tests 1 and 1b are kept as expected failures next to their replacements) |
| `results/` | Complete results (below) |
| `figures/` | The paper's figures, produced by `python -m bench.figures` |
| `vendor/fusion/` | The case-study fusion network and trainer, frozen, with MD5 checksums |
| `posthoc/` | Follow-up runs after the benchmark, each with its plan, code, logs and results (paper §4.6, Supplementary Section S5; see `posthoc/README.md`) |
| `prereg/measurements/` | Timing measurements and the batched-trainer equivalence check made before the runs |
| `env/` | Python and R environments, pinned versions |
| `data/`, `models/` | Local only, not committed (see *Reproducing*) |

## Results

- `results/main_all_runs.parquet` — one row per run (51,000). Key: `task`, `N`, `r`
  (repeat), `arm` (`true` / `perm`), `config_id` / `config`. Estimates: `auc_int`
  (primary internal estimate: pooled LOOCV for N ≤ 12, 5 × 5 stratified k-fold with AUC
  per fold for N ≥ 20), `auc_int_lpo` (leave-pair-out, N ≤ 12), `auc_int_leaky`
  (supervised filter fitted on the whole subsample), `auc_ext` (held-out AUC). Recovery:
  `p10`–`p100`, `rbo`, `dir50` (and `_wil` against the Wilcoxon reference); `top100` (the
  run's top-ranked protein indices). `code` is an MD5 over `bench/*.py` at run time; the
  two values are explained in Supplementary Section S3 and can be reproduced from commits
  `8e50bbe` and `17be563`.
- `results/runs/main/plan_*.json` — the run plans. The per-block files behind
  `main_all_runs.parquet` are not committed; `python -m bench.merge` rebuilds the table
  from them.
- `results/runs/fus_lpo/` — post hoc leave-pair-out for the fusion network.
- `results/runs/smoke/` — the two-draw smoke test of the fusion runner (13 October 2026).
- `results/runs/repro/` — one block of every runner under each code version, re-run from
  scratch by `bench.reproduce`; `results/logs/reproduce.log` records that all 380 runs match
  the stored values exactly.
- `results/analysis/task_selection.csv` — the pre-specified task selection, recomputed with
  the effect statistic of Table 1 (`bench.selection`; Supplementary Table S1).
- `results/analysis/` — ranking, optimism, usable-N, recovery and decision tables
  (`bench.analysis`), and the post hoc analyses (`review_*`, `bench.review`).
- `results/stability.parquet`, `results/stability_f21.json` — selection stability.
- `results/case_study/` — internal AUCs on the motivating nine-sample study. The study's
  data are not public; they are available from the authors on reasonable request.
- `results/logs/` — run, leakage-test and analysis logs, including the failed original
  leakage gate (`LEAKAGE_FAILED_registered_gate_2026-10-13.txt`).

## Reproducing

**Environment.** See `env/README.md` (Python 3.13, PyTorch with CUDA, `tabpfn` 9.0.0,
`cptac` 1.5.14; R 4.4 with limma for the reference rankings). The runs used Windows 11, 16
CPU worker processes and one RTX 5070 Ti (16 GB). Run every command from the repository root with
`PYTHONPATH=.` and `OMP_NUM_THREADS=1`.

**Data** (into `data/raw/`):

- CPTAC LUAD and PDAC (tasks B1, B2): fetched by the `cptac` package on first use.
- ProCan-DepMapSanger (task B5): `ProCan-DepMapSanger_protein_matrix_8498_averaged.txt`
  (figshare, doi:10.6084/m9.figshare.19345397), with the Cell Model Passports
  `model_list_20261010.csv` and `mutations_all_20260724.zip`.
- `data/raw/reference/`: UniProtKB/Swiss-Prot human, Ensembl 116 peptides, Reactome 97
  `UniProt2Reactome`.
- `models/tabpfn/tabpfn-v3.5-fast-20260909.safetensors` (MD5
  `9340b9f6927ab257228158cbe446f01f`), downloaded from Prior Labs under its licence.

**Steps.**

```
python -m bench.derive          # B5 mutation tables -> data/derived/
python -m bench.selection       # task selection under the pre-specified rules (needs R)
python -m bench.references      # full-cohort R limma references (task tables are built on first load)
python -c "from bench import embeddings; print(embeddings.build_all())"   # ESM-2 embeddings
#   (python -m bench.embeddings also checks against the case-study embeddings, which are not public)

# leakage cells (B1, permuted labels), then the gate, then the full run
python -m bench.run --runner cpu    --out main --tasks B1 --N 12,20 --arms perm --workers 16
python -m bench.run --runner tabpfn --out main --tasks B1 --N 12,20 --arms perm
python -m bench.fusion --out main --tasks B1 --N 20 --arms perm --block 50
python -m bench.orchestrate     # runs tests/, launches the full run only if they pass

python -m bench.merge --out main                 # -> results/main_all_runs.parquet
python -m bench.reproduce                        # re-run one block per runner and code version, compare
python -m bench.stability
python -m bench.analysis rankings                # also: optimism, usable_n, recovery, decision

# post hoc
python -m bench.fusion_lpo --tasks B2,B5 --arms perm,true
python -m bench.fusion_lpo --tasks B1 --N 12 --arms perm
python -m bench.review all
python -m bench.figures
python -m bench.case_study      # needs the (non-public) case-study data
```

Runners are resumable: each writes one file per block and skips blocks already present.
GPU runs can be wrapped in `python -m bench.supervise --tries 8 --stall 60 -- <args>`,
which restarts a crashed or stalled run. CPU arms took about 6 h; the GPU arms about 45 h.

## Licence

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The results in
`results/` and the figures in `figures/` are released under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The public cohorts and the TabPFN and ESM-2 weights are not redistributed here and keep
their own terms; the motivating nine-sample dataset is not public.
