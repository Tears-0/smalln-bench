# data/

Nothing in this folder except this file is committed (see `.gitignore`).

## case_study/ — the motivating nine-sample study (local only)

Copied 2026-10-03 from the CSH project. Not public; whether to release it is
the data owner's decision.

| File | Source in CSH | MD5 |
|---|---|---|
| `not_imputed.csv` | `data/not_imputed.csv` — 2,841 accessions × 9 samples (F61–F69), log2, NaN for missing; columns Accession, Description, F61…F69, Sequence | 35aad02c620ecf14ec8ee7ae5d3f1073 |
| `esm2_t12_35M_cls_embeddings.pt` | `src/static_protein_embs(NI).pt` — 2,841 × 480 ESM-2 (esm2_t12_35M_UR50D) embeddings, row-aligned to `not_imputed.csv`, CLS-token readout (`representations[12][:, 0, :]`) | 2f641a73e0ac57ed46de798dedde04b7 |
| `Control_Proteins.xlsx`, `8hours_Proteins.xlsx`, `24hours_Proteins.xlsx` | `data/` — per-condition Proteome Discoverer exports, kept as provenance for `not_imputed.csv` | — |

Sample groups: Control = F61–F63, 8 h = F64–F66, 24 h = F67–F69.

## raw/ — public benchmark datasets (fetched, not committed)

| Task | What to place here | Source |
|---|---|---|
| B1, B2 | nothing — fetched by the `cptac` package into its own cache | `cptac` 1.5.14 |
| B3 | *dropped* — `cohort_1_processed_matrix.xlsx` has no patient IDs | — |
| B5 | `ProCan-DepMapSanger_protein_matrix_8498_averaged.txt` | figshare, doi:10.6084/m9.figshare.19345397 |
| B5 | `model_list_20261010.csv`, `mutations_all_20260724.zip` | Cell Model Passports downloads (release dates in the file names) |

## derived/ — small tables computed from raw/ (2026-10-13)

| File | Contents |
|---|---|
| `kras_tp53_calls.csv` | Every KRAS and TP53 call in `mutations_all_20260724.csv` (2,480 rows) |
| `models_with_mutation_data.csv` | The 1,569 Cell Model Passports models with any mutation call (wild-type requires being sequenced) |
| `cptac_cohort_counts.csv` | Tumour / NAT / patient counts and missingness for the 10 CPTAC cohorts (B1 rule) |
| `cptac_b2_candidates.csv` | B2 candidate contrasts with class counts and effect sizes (`harmonized` mutations; PDAC and COAD were re-evaluated with `washu`, as their `harmonized` tables are empty in `cptac` 1.5.14) |
| `procan_cohort1_annotation.csv` | Sample annotation of the dropped B3 matrix (no patient IDs) |
