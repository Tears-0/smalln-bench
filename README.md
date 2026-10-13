# smalln-bench

A subsampling benchmark for proteomic biomarker discovery at very small sample
sizes (N = 6 to 50). Large public cohorts are subsampled to small N, every
method is scored against the held-out remainder, and the gap between a method's
own internal cross-validation estimate and its held-out performance — its
*optimism* — is measured as a function of N.

**Status: pre-registered design, no results yet.** The design in
[`specification.md`](specification.md) (§5 experimental design, §6 metrics) was
committed before any benchmark run, so that the analysis cannot be tuned to the
outcome. §15 records every decision taken at the freeze, with its reason.

## Layout

| Path | Contents |
|---|---|
| `specification.md` | Research question, frozen design, metrics, leakage tests, freeze record |
| `prereg/measurements/` | Timing scripts and results behind the run budget (§5.3) |
| `vendor/fusion/` | The frozen case-study fusion model and trainer, with MD5 checksums |
| `env/` | Python and R environment setup and pinned versions |
| `data/` | Local only, git-ignored: case-study data and fetched public datasets; the README records provenance |
| `models/` | Local only, git-ignored: TabPFN weights; the README records the checksum |
| `future_work.md` | Ideas that arrive after the freeze go here, not into the grid |

Loaders, the benchmark harness and the leakage test suite follow (spec §3,
D2–D3). No data is committed to this repository.
