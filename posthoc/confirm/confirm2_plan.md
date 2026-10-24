# Confirmation run 2: do PCA-derived protein embeddings carry useful content for the fusion network?

Written 2026-10-23, before any run-2 result exists. Nothing below changes after the run.
Exploratory follow-up to confirmation run 1 (confirm_plan.md); not part of the RECOMB paper.

## Why
Run 1 (repeats 100-199, pre-specified): V3 with real ESM-2 embeddings did no better than with the same
embeddings shuffled across proteins, so ESM-2's benefit in V3 was protein identity, not content. Run 2 asks
whether a data-derived embedding of protein biology (co-expression in independent cohorts), or a compressed
ESM-2 embedding, carries content that helps.

## Embeddings (all label-free and sample-disjoint from the task; fixed before running)
- **COEXP-d** (external co-expression, d = 32, 64). B2: tumour samples of the eight other CPTAC cohorts with
  `umich` proteomics (LUAD, LSCC, CCRCC, HNSCC, UCEC, GBM, OV, BRCA; 859 tumours). B5: every
  ProCan-DepMapSanger line outside the haematopoietic and lymphoid tissue. Per cohort, each protein is
  z-scored across that cohort's samples (left missing where it is observed in < 50% of them); missing = 0;
  the protein x sample matrix is centred per sample over proteins; SVD; a protein's embedding is its
  coordinates on the top d components (U_d S_d), each dimension scaled to unit SD over covered proteins;
  uncovered proteins get 0. Proteins are matched by Ensembl protein ID (B2) or matrix column (B5).
- **ESMPC-d** (d = 32, 64): the top d principal components of the ESM-2 CLS embeddings, each scaled to
  unit SD (as in V2).
- **Shuffled controls**: for each embedding, rows permuted across covered proteins once, with a fixed seed
  per (task, source, d). Identity kept, content broken.

## Model
V3 multiply with embedding width d: denoiser 500-1024-1024-500 (25% masking, reconstruction weight 1), V1
per-protein encoder Linear(1,128)-ReLU-Linear(128,d)-sigmoid, f = e x gate, attention Linear(d,128)-tanh-
Linear(128,1), classifier Linear(d,256)-ReLU-Dropout(0.5)-Linear(256,2); RMSprop lr 1e-4, wd 1e-4,
1,000 epochs, batch 1. A real embedding and its shuffled control share initial weights, masks and sample
order on every draw, so only the embedding's content differs.

## Data
Fresh repeats **r = 200-299**, true labels; cells B2 N = 6, 12, 30 and B5 N = 6, 12, 30, 50. Permuted-label
check at N = 12 (B2, B5) for the four real-embedding models. Baselines (CCP, EN, RF, TabPFN) and V2 on the
same draws. Outcome: held-out AUC on H_r.

## Hypotheses (paper's paired rule: median difference > 0.02 and positive in >= 75% of draws, in BOTH
B2 and B5; evaluated at N = 12 and 30; N = 6 and B5 N = 50 descriptive)
- **H3 (d = 32, 64)**: V3[COEXP-d] beats V3[shuffled COEXP-d].
- **H4 (d = 32, 64)**: V3[ESMPC-d] beats V3[shuffled ESMPC-d].
- **H5**: each real-embedding V3 vs TabPFN: "beats", "is beaten" (reverse rule in both tasks), or
  "no difference".
Eight content tests (4 embeddings x 2 N); the both-task rule is the safeguard against chance passes.

## Checks (otherwise stop and report)
- The baseline runner reproduces the stored benchmark on repeats 0-9 (B2 N = 12, B5 N = 30).
- Embedding coverage is reported (share of task proteins with a non-zero embedding).
- Permuted-label mean held-out AUC of each real-embedding model at N = 12 lies within 0.5 +/- 0.05.

## Interpretation fixed in advance
- H3 holds for some d -> co-expression in independent cohorts carries information the fusion network can
  use beyond protein identity.
- H4 holds -> compressed ESM-2 carries usable content (contrary to run 1 for full ESM-2).
- Neither -> no evidence that these embeddings' content helps; the identity explanation stands.
- H5 "beats" -> competitive as a method on these tasks.
One run; results reported whatever they are.
