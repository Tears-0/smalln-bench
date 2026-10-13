"""Fold plans and AUC estimators (spec §5.2, F14).

Positions are indices into S_r (0..N-1).
  N <= 12  primary   LOOCV, AUC over the N pooled held-out scores
           secondary leave-pair-out: every (positive, negative) pair held out once
  N >= 20  primary   5 x repeated stratified 5-fold, AUC per fold, averaged
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .subsample import seed

LOOCV_MAX_N = 12
KFOLD_REPEATS, KFOLD_K = 5, 5


def primary_kind(N):
    return "loocv" if N <= LOOCV_MAX_N else "kfold"


def primary_folds(yS, N, coords):
    """List of (train_pos, test_pos)."""
    pos = np.arange(N)
    if primary_kind(N) == "loocv":
        return [(np.delete(pos, i), np.array([i])) for i in range(N)]
    folds = []
    for rep in range(KFOLD_REPEATS):
        skf = StratifiedKFold(KFOLD_K, shuffle=True, random_state=seed("kfold", *coords, rep) % (2**32))
        for tr, te in skf.split(np.zeros(N), yS):
            folds.append((tr, te))
    return folds


def lpo_folds(yS, N):
    if N > LOOCV_MAX_N:
        return []
    P, Q = np.flatnonzero(yS == 1), np.flatnonzero(yS == 0)
    pos = np.arange(N)
    return [(np.setdiff1d(pos, [i, j]), np.array([i, j])) for i in P for j in Q]


def _auc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, s)


def primary_auc(yS, N, folds, scores):
    """scores: list of arrays aligned with each fold's test positions."""
    if primary_kind(N) == "loocv":
        pooled = np.empty(N)
        for (_, te), s in zip(folds, scores):
            pooled[te] = s
        return _auc(yS, pooled)
    return float(np.nanmean([_auc(yS[te], s) for (_, te), s in zip(folds, scores)]))


def lpo_auc(yS, folds, scores):
    """Fraction of held-out (pos, neg) pairs ordered correctly; ties count 1/2."""
    if not folds:
        return np.nan
    v = []
    for (_, te), s in zip(folds, scores):
        i, j = te  # te = [positive, negative]
        assert yS[i] == 1 and yS[j] == 0
        v.append(1.0 if s[0] > s[1] else 0.5 if s[0] == s[1] else 0.0)
    return float(np.mean(v))


def external_auc(yH, s):
    return _auc(yH, s)
