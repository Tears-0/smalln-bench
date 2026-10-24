# Confirmation run 4: does outside information about the label (a transferred TP53 signature) help?

Written 2026-10-24, before any run-4 result exists. Nothing below changes after the run.
Follow-up to runs 1-3 (embeddings carried no usable content). Post hoc with respect to the benchmark.

## External TP53 signature beta_ext (built once, before any task draw is touched)
- **B2** (CPTAC PDAC tumours): tumours of the other CPTAC cohorts with proteomics (`umich`; COAD `bcm`) and
  TP53 calls (non-silent; `harmonized`, else `washu`, as in the task selection): BRCA, CCRCC, COAD, GBM,
  HNSCC, LSCC, LUAD, OV, UCEC. Proteins matched by Ensembl protein ID.
- **B5** (ProCan haematopoietic and lymphoid lines): ProCan-DepMapSanger lines of every other tissue with
  mutation data; TP53 driver mutant vs wild-type (Cell Model Passports calls, as in the task).
- Within each cohort (CPTAC) or tissue (ProCan), each protein is z-scored across samples (missing where
  observed in < 50%), then missing = 0. A ridge logistic regression (C = 1) on the pooled external samples
  gives beta_ext over the task's proteins (0 where a protein is absent).
- Quality check (descriptive): leave-one-cohort/tissue-out AUC within the external data, for groups with at
  least 10 samples per class.

## Models on the task (in-fold, reference pipeline input: 500 proteins by moderated t, z-scored)
- **Ridge** (tau = 0): ridge logistic regression, C = 1.
- **Transfer-tau**: ridge on [Z, tau * Z b], b = beta_ext on the selected proteins scaled to unit norm, so the
  effective coefficients are beta = delta + tau * u * b. tau in {0, 0.125, 0.25, 0.5, 1, 2, 4, 8}, chosen by
  inner CV on S_r with preprocessing refitted per inner fold (leave-pair-out AUC for N <= 12, 5 x 5 per-fold
  k-fold for N >= 20); ties to the smallest tau.
- **Zero-shot** (no task training): each held-out sample's score is beta_ext . z, with z standardised by S_r's
  protein means and SDs over proteins observed in >= 50% of S_r (missing = 0). Labels of S_r are not used.
- **Baselines** CCP, EN, RF, TabPFN as in the benchmark (final fit on S_r).

## Data
Fresh repeats **r = 400-499**, true labels; B2 N = 6, 12, 30; B5 N = 6, 12, 30, 50. Outcome: held-out AUC on
H_r.

## Hypotheses (paper's paired rule: median difference > 0.02 and positive in >= 75% of draws, in BOTH B2 and
B5; evaluated at N = 12 and 30; N = 6 and B5 N = 50 descriptive)
- **H7**: Transfer-tau beats Ridge.
- **H8**: Transfer-tau vs TabPFN: "beats", "is beaten", or "no difference".
- **E2** (descriptive): the share of draws choosing tau = 0 is lower than with the embeddings in run 3.
- Descriptive: zero-shot held-out AUC against Ridge and TabPFN.

## Interpretation fixed in advance
H7 holds -> outside label information improves small-N prediction where protein embeddings did not. H7 fails
-> no evidence that this transferred signature helps; report the zero-shot AUC to show whether the signature
transfers at all. One run; results reported whatever they are.
