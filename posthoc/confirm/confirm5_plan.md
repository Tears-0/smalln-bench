# Confirmation run 5: does a transferred TP53 signature, applied zero-shot, beat models trained on the study?

Written 2026-10-24, before any run-5 result exists. Nothing below changes after the run.
Run 4 (repeats 400-499) found this descriptively; run 5 tests it as a hypothesis on fresh draws.

## Models
- **Zero-shot**: run 4's external TP53 signatures (`sig4/B2_beta.npy`, `sig4/B5_beta.npy`, unchanged; built
  from other CPTAC cohorts and other ProCan tissues). A held-out sample's score is beta_ext . z, with z
  standardised by S_r's protein means and SDs over proteins observed in >= 50% of S_r (missing = 0). No
  label of S_r is used.
- **Comparators**: CCP, EN, RF, TabPFN exactly as in the benchmark (final fit on S_r, scored on H_r).

## Data
Fresh repeats **r = 500-599**, true labels; B2 N = 6, 12, 30; B5 N = 6, 12, 30, 50. Outcome: held-out AUC.

## Hypothesis (paper's paired rule: median difference > 0.02 and positive in >= 75% of draws, in BOTH B2
and B5; evaluated at N = 12 and 30; N = 6 and B5 N = 50 descriptive)
- **H9**: zero-shot beats TabPFN. Secondary, same rule: zero-shot beats each of CCP, EN, RF.

## Interpretation fixed in advance
H9 holds at an N -> at that N, scoring with a signature transferred from external labelled data predicts
better than any of the benchmark's models trained on the study's own samples, in both tasks. H9 fails ->
report the run-4 result as descriptive only. One run; results reported whatever they are.
