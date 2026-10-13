"""In-fold preprocessing (spec §5.1 reference pipeline and axis variants).

Every statistic is estimated on the training indices only and then applied to
the test indices. Each call is recorded in AUDIT when auditing is switched on,
so tests/test_no_leakage.py can assert by index tracking (spec §8 test 2) that
no fitted object has seen a test index.

variant  presence filter        missing values                    filter (top 500)
ref      >= 50% observed        row-wise left-censored draw        |moderated t|
var      >= 50% observed        row-wise left-censored draw        training variance
drop     100% observed          none (test NaN -> training mean)   |moderated t|
path     >= 50% observed        row-wise left-censored draw        |moderated t| on Reactome
                                                                    pathway features
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .limma import moderated_t_complete

P_FILTER = 500
DOWNSHIFT, WIDTH = 1.5, 0.3
PATH_MIN, PATH_MAX = 10, 200

AUDIT = None  # set to a list to record (fit_indices, test_indices, tag)


@dataclass
class Prep:
    Ztr: np.ndarray        # [n_train, f]
    Zte: np.ndarray        # [n_test, f]
    feat: np.ndarray       # task protein indices (or pathway column indices for 'path')
    t_feat: np.ndarray     # in-fold moderated t of the selected features
    rest: np.ndarray       # task protein indices present but not selected, by |t| desc
    kind: str              # 'protein' or 'pathway'


@dataclass
class TaskCtx:
    X: np.ndarray                 # float32 [n, p]
    has_seq: np.ndarray           # bool [p]
    M: np.ndarray | None = None   # bool [p, q] Reactome membership


def _order_by_abs(t, idx):
    o = np.argsort(-np.nan_to_num(np.abs(t), nan=-1.0), kind="stable")
    return idx[o], t[o]


def prepare(ctx: TaskCtx, tr, te, ytr, variant, sd, fixed=None, restrict_seq=False, tag=""):
    if AUDIT is not None:
        AUDIT.append((np.asarray(tr).copy(), np.asarray(te).copy(), tag))
    Xtr, Xte = ctx.X[tr], ctx.X[te]
    frac = (~np.isnan(Xtr)).mean(0)
    present = np.flatnonzero(frac == 1.0) if variant == "drop" else np.flatnonzero(frac >= 0.5)
    A = Xtr[:, present].astype(np.float64)
    B = Xte[:, present].astype(np.float64)

    if variant == "drop":
        mu = A.mean(0)
        B = np.where(np.isnan(B), mu, B)
    else:
        mu_o = np.nanmean(A, 0)
        sd_o = np.nan_to_num(np.nanstd(A, 0, ddof=1), nan=0.0)
        loc, scale = mu_o - DOWNSHIFT * sd_o, WIDTH * sd_o
        rng = np.random.default_rng(sd)
        for M in (A, B):
            miss = np.isnan(M)
            if miss.any():
                cols = np.nonzero(miss)[1]
                M[miss] = rng.normal(loc[cols], scale[cols])

    t_all = moderated_t_complete(A, ytr)
    m, s = A.mean(0), A.std(0, ddof=1)
    s[~(s > 0)] = 1.0
    ZA, ZB = (A - m) / s, (B - m) / s

    if variant == "path":
        Mp = ctx.M[present]
        cnt = Mp.sum(0)
        keep = np.flatnonzero((cnt >= PATH_MIN) & (cnt <= PATH_MAX))
        W = Mp[:, keep].astype(np.float64) / cnt[keep]
        FA, FB = ZA @ W, ZB @ W
        tf = moderated_t_complete(FA, ytr)
        if fixed is not None:
            sel = np.flatnonzero(np.isin(keep, fixed))
        else:
            sel = np.argsort(-np.abs(tf), kind="stable")[:P_FILTER]
        sel = sel[np.argsort(-np.abs(tf[sel]), kind="stable")]
        rest_idx, _ = _order_by_abs(t_all, present)
        return Prep(FA[:, sel], FB[:, sel], keep[sel], tf[sel], rest_idx, "pathway")

    cand = np.arange(present.size)
    if restrict_seq:
        cand = cand[ctx.has_seq[present[cand]]]
    if fixed is not None:
        sel = cand[np.isin(present[cand], fixed)]
    elif variant == "var":
        v = A[:, cand].var(0, ddof=1)
        sel = cand[np.argsort(-v, kind="stable")[:P_FILTER]]
    else:
        sel = cand[np.argsort(-np.abs(t_all[cand]), kind="stable")[:P_FILTER]]
    sel = sel[np.argsort(-np.abs(t_all[sel]), kind="stable")]
    mask = np.ones(present.size, bool); mask[sel] = False
    rest_idx, _ = _order_by_abs(t_all[mask], present[mask])
    return Prep(ZA[:, sel], ZB[:, sel], present[sel], t_all[sel], rest_idx, "protein")


def leaky_features(ctx, S, yS, variant, sd, restrict_seq=False):
    """The supervised filter fitted once on all of S_r (leaky-internal variant, spec §5.2)."""
    p = prepare(ctx, S, S[:1], yS, variant, sd, restrict_seq=restrict_seq, tag="leaky-filter-on-S")
    return p.feat
