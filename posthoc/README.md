# Follow-up runs

Post hoc with respect to the benchmark: everything here was done after the benchmark's runs
and analysis were complete. Paper: Section 5.1; Supplementary Section S5.

## Pilots (`pilots/`): exploratory, not evidence

Fusion-network variants tried on the benchmark's own draws (repeats 0–99), with the stored
benchmark values as comparators. Nine variants were tried on the same draws, so these results
only motivated the plans below.

| Script | Model |
|---|---|
| `pilot_fusion_v2.py` | V2 (EIL): ridge logistic regression on `[Z, Z @ ESM-PCs]` |
| `pilot_fusion_v3.py` | V3: denoising autoencoder (500–1024–1024–500, 25% masking) in front of the fusion network's per-protein encoder; fusion by adding or multiplying; `noesm` drops the ESM-2 embedding |
| `analyse_pilot.py`, `analyse_v3.py`, `analyse_noesm.py`, `plot_pilot.py` | Tables and figures |

## Follow-up runs (`confirm/`)

Each run has a plan (`confirm*_plan.md`), written before the run and not changed afterwards,
fixing the models, data, hypotheses, decision rule and how each outcome would be read. Every
run uses fresh repeats of the benchmark's seeded draws, true labels, B2 at N = 6, 12, 30 and
B5 at N = 6, 12, 30, 50. A hypothesis holds only if the paper's paired rule (median difference
> 0.02, positive in ≥ 75% of draws) is met in both B2 and B5 at N = 12 or 30.

| Run | Repeats | Plan fixed | Plan SHA-256 | Question | Outcome |
|---|---|---|---|---|---|
| 1 | 100–199 | 2026-10-23 00:24:03 | `f5beb49ee35df112aa33fc0ed1301bcc780a5f0062affec22880d7a5cd66e332` | Does ESM-2 content help V3 (vs the same embeddings shuffled across proteins)? Does V3 beat TabPFN or V2? | No |
| 2 | 200–299 | 2026-10-23 17:09:48 | `5de9321ccaa591edd480d953cdc3445a52ab7b2293b6eb4e8569281f0d77e652` | Do 32- or 64-dimensional ESM-2 PCs or co-expression embeddings help V3 (vs shuffled)? | No |
| 3 | 300–399 | 2026-10-24 10:10:49 | `6dc484cfba923664a8c4e95733ec2c90d342be3d7e733270845f815cb75bcf86` | Does an embedding strength τ, tuned by nested CV, beat τ = 0? | No (median gain 0.000 in every cell) |
| 4 | 400–499 | 2026-10-24 10:48:33 | `ad995aff673040997e0564c40d8978c08bc0fb60d500fa95152beef63e87f5ed` | Does a transferred TP53 signature as a τ prior beat ridge, or TabPFN? | No (met in B5 only) |
| 5 | 500–599 | 2026-10-24 13:39:11 | `f4e05aa9f5aa8a7ec2e4794bf170952e6931c0d76866d002c94385c28b2578d0` | Does the transferred signature, applied zero-shot, beat TabPFN (and CCP, EN, RF)? | **Yes**, at N = 12 and 30 in both tasks |

The plans of runs 1–3 describe those runs as outside the paper. The decision to report all
five was taken after run 5.

Signatures used in runs 4 and 5 (`confirm/sig4/`, built once by `confirm4.py sig` before any
run-4 draw and used unchanged in run 5):

- `B2_beta.npy`: SHA-256 `7406156880754c46ff4a3c5b0231c59ef288f8e5429aa49c15f4064e8d8a9447`
  (831 tumours of eight other CPTAC cohorts)
- `B5_beta.npy`: SHA-256 `0c9f62d2229535eb07b0b0c718e21415833c6e5f7ac0e7fec668b72874c9aa5c`
  (775 ProCan-DepMapSanger lines of 23 other tissues)

Each run has a runner (`confirm*.py`), a log (`confirm*.log`), result tables (`confirm*_*.csv`),
an analysis (`analyse_confirm*.py`, which writes `confirm*_paired.csv`) and, for runs 1, 2, 4
and 5, a figure. Run 5's figure (`confirm5.pdf`, Supplementary Figure S3) is drawn by
`plot_confirm5.py`.

### Recorded deviations

- **Run 1, check stage.** The reproduction check (the baseline and V2 runners reproduce the
  stored values on repeats 0–9) passed before the run's main stages, but its output was not
  written to a file. It was re-run on 2026-10-24 with the same result (`confirm_check.log`).
  Run 2 repeats the same check in its own log.
- **Run 1, analysis.** In files holding only true-label rows, pandas read the `arm` column
  as boolean, so the first merge produced missing values. The analysis reads `arm` as text;
  no stored value changed.
- **Run 4, COAD.** The plan lists COAD among the B2 signature's cohorts, but its proteomics
  (`bcm`) are indexed by Ensembl gene IDs, so no protein matched under the plan's rule
  (Ensembl protein ID). COAD contributed nothing (`confirm4_sig.log`).

## Reproducing

Use the benchmark environment (`env/README.md`) and run from `posthoc/confirm/`. The repository
root is taken from the first argument or from `SMALLN_REPO` (default: two levels up). Runs 3
and 4 use joblib workers, which import the runner as a module, so `posthoc/confirm` must be on
`PYTHONPATH`.

```
cd posthoc/confirm
python confirm.py ../..                      # run 1 (check, baselines, v2, v3)
python confirm2.py ../.. emb                 # builds emb2/ (not committed; needed by runs 2 and 3)
python confirm2.py ../..                     # run 2
PYTHONPATH=. python confirm3.py              # run 3
PYTHONPATH=. python confirm4.py ../.. sig baselines transfer   # run 4 (sig rebuilds sig4/)
python confirm5.py ../..                     # run 5
python analyse_confirm5.py && python plot_confirm5.py
```

The fusion-network runs (runs 1 and 2) need a GPU and took about 3 and 10 hours; runs 3–5
took under 15 minutes each. Rebuilding `sig4/` gives the same signatures up to the solver's
tolerance, not byte-identical files: on 2026-10-24 the largest coefficient difference was
4 × 10⁻⁴ in B2 (largest coefficient 0.22) and 3 × 10⁻¹³ in B5, with identical
leave-one-group-out AUCs. The committed files, with the hashes above, are the ones runs 4 and
5 used.
