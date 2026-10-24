# Confirmation run 3: embedding strength tau chosen by nested cross-validation

Written 2026-10-24, before any run-3 result exists. Nothing below changes after the run.
Exploratory follow-up to runs 1-2; not part of the RECOMB paper.

## Model
Ridge logistic regression (C = 1, lbfgs) on the in-fold input of the fusion arm (reference pipeline, 500
proteins with a sequence, z-scored): features [Z, tau * Z A], with A the protein embedding restricted to the
selected proteins and divided by sqrt(500). The effective protein coefficients are beta = delta + tau * A u:
tau = 0 is plain ridge, tau = 1 is V2.
- Embeddings (k = 32): **ESMPC** (top 32 PCs of ESM-2 CLS embeddings, unit SD) and **COEXP** (co-expression
  PCs from independent cohorts, run 2's files), each with its **shuffled** control (run 2's files).
- tau grid: {0, 0.125, 0.25, 0.5, 1, 2, 4, 8}.
- tau chosen on the training subsample S_r only, by inner cross-validation with preprocessing refitted
  inside every inner fold: leave-pair-out AUC for N <= 12, 5 x 5 stratified k-fold with AUC per fold averaged
  for N >= 20 (the pair-symmetric estimators of Proposition 1). Ties go to the smallest tau.
- The final model is refitted on all of S_r with the chosen tau and scored on H_r.

## Data
Fresh repeats **r = 300-399**, true labels; B2 N = 6, 12, 30; B5 N = 6, 12, 30, 50.

## Expectations and decision rule (stated before the run)
- **E1**: chosen tau concentrates at or near 0, and its distribution with the real embedding does not differ
  materially from that with the shuffled embedding. Reported: share of draws choosing tau = 0, median tau,
  per cell and embedding.
- **H6**: the tau-tuned model beats plain ridge (tau = 0) on held-out AUC (paper's paired rule: median
  difference > 0.02 and positive in >= 75% of draws, in BOTH B2 and B5; evaluated at N = 12 and 30).
  Expected outcome: no difference.
- Descriptive: held-out AUC at tau = 1 (V2) on the same draws.

## Interpretation fixed in advance
tau near 0 and no held-out gain -> one more line of evidence that these embeddings carry no usable content
for these tasks. A clear preference for tau > 0 with the real but not the shuffled embedding, together with a
held-out gain, would contradict runs 1-2 and be reported as such.
