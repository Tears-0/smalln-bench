# Confirmation run: does ESM-2 content help the fusion network, and does fusion beat the baselines?

Written 2026-10-23, before any confirmation result exists. Nothing below changes after the run.
Exploratory follow-up to the post hoc pilots (V2, V3); not part of the RECOMB paper.

## Why
The pilots (benchmark repeats 0-99) showed: V3 (denoiser + V1 encoder) beats V1; removing ESM-2 from V3
costs 0.01-0.07 AUC at N >= 12; but in the linear V2, ESM-2 adds nothing, and no fusion variant beats the
baselines. The V3 effect may come from protein *identity* (attention pooling is permutation-invariant, so an
embedding is the only per-protein tag), not from ESM-2's biological *content*. Nine variants were tried on
the same draws, so pilot results are not evidence.

## Models (all fixed; no tuning)
- **V3-ESM**: V3 multiply (best V3 variant in the pilot): denoiser 500-1024-1024-500 (25% input masking,
  reconstruction weight 1), V1 per-protein encoder, f = e x gate, V1 pooling/classifier/training.
- **V3-shuffled**: identical, but the ESM-2 embedding rows are permuted across proteins once per task
  (fixed seed `seed("shuffle-esm", task)`, among proteins with a sequence). Same distribution and geometry,
  identity preserved, biological content broken. Same initial weights, masks and sample order as V3-ESM.
- **V2** (EIL): ridge logistic regression on [z, z @ ESM-PCs], C = 1, k = 32.
- **Baselines**: CCP, EN, RF, TabPFN exactly as in the benchmark (final fit on S_r, held-out AUC on H_r).

## Data
Fresh draws: repeats **r = 100-199** of the benchmark's seeded draw procedure (never used before), true
labels. Cells: B2 at N = 6, 12, 30; B5 at N = 6, 12, 30, 50. Permuted-label check: B2 and B5 at N = 12 for
both V3 models. Same in-fold reference preprocessing as the benchmark (FUS input restricted to proteins with
a sequence). Outcome: held-out AUC on H_r.

## Hypotheses and decision rule (the paper's paired rule)
A beats B at (task, N) if the paired difference A - B has median > 0.02 and is positive in >= 75% of draws.
A claim requires the rule in **both B2 and B5** at the same N. Evaluated at **N = 12 and N = 30**
(primary); N = 6 and B5 N = 50 reported descriptively.

- **H1 (ESM-2 content)**: V3-ESM beats V3-shuffled.
- **H2 (method)**: V3-ESM beats TabPFN; V3-ESM beats V2. For each, the outcome is "beats", "is beaten"
  (reverse rule in both tasks) or "no difference".
- Everything else (other baselines, medians) is descriptive.

## Checks that must pass first (otherwise stop and report)
- The external-only baseline runner reproduces the stored benchmark values exactly on repeats 0-9
  (B2 N = 12, B5 N = 30; CCP, EN, RF, TabPFN).
- The V2 runner reproduces the pilot's EIL values exactly on repeats 0-9 (same cells).
- Permuted-label mean held-out AUC of both V3 models at N = 12 lies within 0.5 +/- 0.05.

## Interpretation fixed in advance
- H1 holds -> evidence that ESM-2 carries information beyond protein identity in this network.
- H1 fails -> the V3-with/without-ESM gap is explained by identity; no evidence for ESM-2 content.
- H2 "beats" -> fusion is competitive as a method on these tasks; otherwise it is not shown to be.
One run; results reported whatever they are.
