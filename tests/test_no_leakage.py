"""Leakage test suite (spec §8, amended F24). MUST PASS before the full run.

  test 1   permuted labels: median honest-internal AUC within 0.5 +/- 0.05 for every
           config (leave-pair-out at N <= 12, k-fold at N >= 20; pooled LOOCV is
           excluded because it is biased below 0.5 under the null)
  test 1b  positive control: the leaky-internal arm must FAIL test 1 (median > 0.6 at N <= 12)
  test 2   index tracking: no preprocessing object fitted on a training fold saw a test index
  test 3   S_r and H_r are disjoint at the patient / donor level, for every draw
  test 4   augmentation never runs before the split, except in the labelled leaky arm

Tests 1 and 1b read the permuted-label cells B1 N = 12 and N = 20 produced by
`python -m bench.leakage` (identical seeds to the main run, so they are reused).
"""
import glob

import numpy as np
import pandas as pd
import pytest

from bench import loaders, preprocess, run
from bench.configs import BY_ID, CONFIGS
from bench.paths import RUNS
from bench.subsample import GRID, R_MAIN, draw, eligible

TOL, POS_CTRL = 0.05, 0.6


# ------------------------------------------------------------------ test 3
@pytest.mark.parametrize("name", loaders.TASKS)
def test_draws_unit_disjoint(name):
    t = loaders.load(name)
    for N in GRID:
        if not eligible(t, N, R_MAIN):
            continue
        for r in range(R_MAIN):
            S, H = draw(t, N, r)
            assert len(S) == N and len(set(S)) == N
            uS, uH = set(t.unit[S]), set(t.unit[H])
            assert len(uS) == N, "S_r must hold one sample per unit"
            assert not (uS & uH), f"{name} N={N} r={r}: a unit is on both sides"
            assert not (set(S) & set(H))
            assert abs(int((t.y[S] == 1).sum()) - int((t.y[S] == 0).sum())) <= 1


# ------------------------------------------------------------------ test 2
@pytest.mark.parametrize("name,N,r,arm", [("B2", 12, 0, "perm"), ("B5", 20, 1, "true"), ("B1", 6, 2, "true")])
def test_index_tracking(name, N, r, arm):
    preprocess.AUDIT = []
    try:
        cpu = [c.id for c in CONFIGS if c.runner == "cpu"]
        run.run_unit(name, N, r, arm, cpu)
        calls = list(preprocess.AUDIT)
    finally:
        preprocess.AUDIT = None
    t = loaders.load(name)
    S, H = draw(t, N, r)
    honest = [c for c in calls if c[2].endswith(":honest")]
    leaky = [c for c in calls if "leaky" in c[2]]
    assert honest and leaky, "audit must see both honest and leaky preprocessing"
    for fit, test, tag in honest:
        assert not (set(fit) & set(test)), f"{tag}: fitted on a test index"
        assert set(fit) <= set(S), f"{tag}: fitted outside S_r"
    for fit, test, tag in calls:
        assert not (set(fit) & set(H)), f"{tag}: a preprocessing object saw H_r"


# ------------------------------------------------------------------ test 4
def test_augmentation_after_split(monkeypatch):
    from bench import models
    seen_infold, seen_before_split = [], []
    real = models.smote
    monkeypatch.setattr(models, "smote", lambda Z, y, sd: (seen_infold.append(len(y)), real(Z, y, sd))[1])
    monkeypatch.setattr(run, "smote", lambda Z, y, sd: (seen_before_split.append(len(y)), real(Z, y, sd))[1])
    N = 12
    run.run_unit("B2", N, 3, "true", [c.id for c in CONFIGS if c.runner == "cpu" and c.smote != "leaky"])
    assert not seen_before_split, "SMOTE ran on all of S_r outside the leaky arm"
    assert seen_infold and set(seen_infold) <= {N - 2, N - 1, N}
    assert seen_infold.count(N) == 2, "only the two external fits (EN, RF) may augment all of S_r"
    run.run_unit("B2", N, 3, "true", [17])
    assert seen_before_split == [N], "the leaky arm must augment all of S_r exactly once"


# ------------------------------------------------------------------ tests 1, 1b
def _perm_rows():
    files = []
    for runner in ("cpu", "tabpfn"):
        for N in (12, 20):
            files += glob.glob(str(RUNS / "main" / runner / f"B1_N{N}_perm_r*.parquet"))
    files += glob.glob(str(RUNS / "main" / "fus" / "B1_N20_perm_r*.parquet"))
    if not files:
        pytest.skip("run `python -m bench.leakage` first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df.arm == "perm"]


@pytest.mark.xfail(strict=False, reason="registered criterion miscalibrated: 0.91 familywise false-alarm "
                                        "probability on a clean pipeline (spec §16 A7); kept as a record")
def test_permuted_null_honest_internal():
    df = _perm_rows()
    bad = []
    for (cid, N), g in df.groupby(["config_id", "N"]):
        col = "auc_int_lpo" if N <= 12 else "auc_int"
        v = g[col].dropna()
        if v.empty:
            continue
        if BY_ID[cid].smote == "leaky":
            continue  # the leaky SMOTE arm is leaky by design (tested in 1b)
        med = float(v.median())
        if abs(med - 0.5) > TOL:
            bad.append((BY_ID[cid].name, N, col, round(med, 3), len(v)))
    assert not bad, f"median permuted-label honest-internal AUC outside 0.5 +/- {TOL}: {bad}"


@pytest.mark.xfail(strict=False, reason="registered on pooled LOOCV, whose method-dependent bias (A5) "
                                        "masks the small pathway-filter leak (spec §16 A7); kept as a record")
def test_positive_control_leaky_fails():
    df = _perm_rows()
    g = df[(df.N == 12)]
    checked = 0
    for cid, gg in g.groupby("config_id"):
        cfg = BY_ID[cid]
        col = "auc_int_leaky" if cfg.leaky_filter else ("auc_int" if cfg.smote == "leaky" else None)
        if col is None or gg[col].dropna().empty:
            continue
        med = float(gg[col].median())
        assert med > POS_CTRL, f"{cfg.name}: leaky arm median {med:.3f} -- the test cannot detect a known leak"
        checked += 1
    assert checked >= 8


# ------------------------------------------------------------------ amended gate (spec §16 A7)
def test_A7_permuted_null_one_sided_mean():
    """A7 test 1: leakage only inflates, so the check is one-sided, on the mean of the
    honest-internal AUC (leave-pair-out at N <= 12, k-fold at N >= 20), threshold
    0.5 + TOL. Familywise false-alarm probability on a clean pipeline: 0.08."""
    df = _perm_rows()
    df = df[~df.config_id.map(lambda c: BY_ID[c].smote == "leaky")].copy()
    df["honest"] = np.where(df.N <= 12, df.auc_int_lpo, df.auc_int)
    m = df.dropna(subset=["honest"]).groupby(["config", "N"])["honest"].mean()
    bad = m[m > 0.5 + TOL]
    assert bad.empty, f"mean permuted-label honest-internal AUC above {0.5 + TOL}: {bad.round(3).to_dict()}"
    assert len(m) >= 25


def test_A7_positive_control_paired():
    """A7 test 1b: the leaky arm must be detectably higher than the honest arm on the same
    draw and estimator -- positive in >= 90% of repeats at N = 20 (k-fold), for every
    supervised-filter config; the leaky SMOTE arm must exceed 0.9 at N = 12."""
    df = _perm_rows()
    d = df.dropna(subset=["auc_int_leaky"]).copy()
    d["diff"] = d.auc_int_leaky - d.auc_int
    frac = d[d.N == 20].groupby("config")["diff"].apply(lambda x: (x > 0).mean())
    assert len(frac) >= 10
    weak = frac[frac < 0.9]
    assert weak.empty, f"leak not detected at N = 20: {weak.round(2).to_dict()}"
    sm = df[(df.N == 12) & df.config_id.map(lambda c: BY_ID[c].smote == "leaky")]
    assert sm.groupby("config")["auc_int"].median().min() > 0.9
