# Measurements behind the §5.3 run budget

Machine: Windows 11, NVIDIA GeForce RTX 5070 Ti (16 GB), 24 logical CPU cores,
torch 2.11.0+cu128, Python 3.13.12. Measured 2026-10-01/02, before any
benchmark run. The scripts import the case-study model from
`CSH/src/rebuttal/` by absolute path; the files they imported are vendored in
`vendor/fusion/` with checksums.

## 1. Case-study trainer, one process — `time_fusion.py`

Exact `trainer.py` inner loop (RMSprop wd 1e-4, batch size 1, AMP + GradScaler,
deterministic algorithms, `loss.item()` sync every step), binary head.

| Proteins P | ms per optimizer step |
|---|---|
| 500 | 3.47 |
| 2,000 | 3.55 |
| 5,000 | 3.45 |
| 10,000 | 3.39 |

Flat in P: the loop is kernel-launch-bound.

## 2. Concurrent processes on one GPU — `conc_fusion.py K`

P = 2,000, 20 s window per worker after warm-up.

| Workers K | ms/step per worker | Aggregate steps/s | Speed-up |
|---|---|---|---|
| 1 | 3.25 | 308 | 1.0× |
| 4 | 5.8–5.9 | 685 | 2.2× |
| 8 | 11.7–11.9 | 678 | 2.2× |
| 12 | 17.8–17.9 | 673 | 2.2× |

Saturates at 2.2× from four workers.

## 3. Batched lockstep trainer — `batched_fusion.py`, `batched_timing2.py`

Equivalence (part 1 of `batched_fusion.py`): four models, 200 steps, fp32,
dropout off, identical initialisation and sample order, one model with a
masked protein subset. Against independently trained
`MultiModalProteomicsModel` instances: max |logit difference| = 7.15e-07,
max |W5 difference| = 1.65e-06.

Timing, dropout on:

| P | Models M | Precision | ms per step | ms per model-step | vs 3.3 ms |
|---|---|---|---|---|---|
| 500 | 100 | AMP | 7.25 | 0.0725 | 46× |
| 500 | 100 | fp32 | 8.42 | 0.0842 | 39× |
| 500 | 400 | AMP | 27.64 | 0.0691 | 48× |
| 500 | 400 | fp32 | 33.96 | 0.0849 | 39× |
| 500 | 800 | AMP | 56.53 | 0.0707 | 47× |
| 500 | 800 | fp32 | 67.62 | 0.0845 | 39× |
| 2,000 | 25 | AMP | 4.59 | 0.184 | 18× |
| 2,000 | 100 | AMP | 17.55 | 0.176 | 19× |
| 2,000 | 100 | fp32 | 25.58 | 0.256 | 13× |
| 2,000 | 400 | AMP | 70.72 | 0.177 | 19× |
| 2,000 | 400 | fp32 | 101.84 | 0.255 | 13× |
| 2,000 | 800 | either | 2,500–3,500 | > 3 | spills out of 16 GB VRAM |
| 10,000 | 25 | AMP | 20.81 | 0.832 | 4.0× |
| 10,000 | 100 | AMP | 77.08 | 0.771 | 4.3× |
| 10,000 | 200 | AMP | — | — | out of memory |

## 4. Budget arithmetic used in §5.3

Model-steps per (task, repeat, label arm) for FUS, 1,000 epochs × summed
training-set sizes, grid {6, 9, 12, 20, 30, 50}:

| N | Internal fits × training size | External | Sum |
|---|---|---|---|
| 6 | 6 × 5 | 6 | 36 |
| 9 | 9 × 8 | 9 | 81 |
| 12 | 12 × 11 | 12 | 144 |
| 20 | 25 × 16 | 20 | 420 |
| 30 | 25 × 24 | 30 | 630 |
| 50 | 25 × 40 | 50 | 1,050 |
| | | **Total** | **2,361** |

2,361 × 1,000 × 100 repeats × 2 arms × 4 tasks = 1.89 × 10⁹ model-steps;
× 0.085 ms = 1.6 × 10⁵ s ≈ **45 GPU-h**.

As first drafted (case-study trainer, 7 N including N = 100 at
25 × 80 + 100 = 2,100, R = 100, 3 tasks): 4,461 × 1,000 × 100 × 2 × 3 =
2.68 × 10⁹ model-steps × 3.5 ms ≈ **2,600 GPU-h**.

## 5. Other checks run at the freeze

- `imblearn` SMOTE with `sampling_strategy = {c: 2·n_c}` and
  `k_neighbors = min(5, n_min − 1)` runs on class counts (3, 2), (2, 3), (5, 4)
  and (3, 3), giving (6, 4), (4, 6), (10, 8) and (6, 6).
- `cptac` 1.5.14 serves proteomics for 10 cohorts. LUAD (`umich` source):
  213 samples × 13,302 proteins, 111 tumours, 102 NATs (`.N` suffix), 111
  distinct patients, 102 with both, 18.9% missing.
