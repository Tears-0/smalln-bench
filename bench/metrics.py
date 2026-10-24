"""Scored metrics (paper §3.5-3.6). Must never import bench.enrichment."""
from __future__ import annotations

import numpy as np

KS = (10, 20, 50, 100)
RBO_DEPTH, RBO_P = 500, 0.98


def full_ranking(prep, importance, depth=RBO_DEPTH):
    """Task protein indices: selected features by importance, then the rest by |t|."""
    o = np.argsort(-importance, kind="stable")
    return np.concatenate([prep.feat[o], prep.rest])[:depth]


def precision_at(rank, ref_rank, k):
    return len(set(rank[:k]) & set(ref_rank[:k])) / k


def rbo_ext(a, b, p=RBO_P, depth=RBO_DEPTH):
    """Extrapolated rank-biased overlap (Webber et al. 2010) at a fixed depth."""
    a, b = list(a[:depth]), list(b[:depth])
    k = min(len(a), len(b))
    sa, sb, X = set(), set(), 0
    acc = 0.0
    for d in range(1, k + 1):
        x, y = a[d - 1], b[d - 1]
        if x == y:
            X += 1
        else:
            X += (x in sb) + (y in sa)
        sa.add(x); sb.add(y)
        acc += (X / d) * p ** d
    return (X / k) * p ** k + (1 - p) / p * acc


def directional(rank, ref_rank, sign_S, sign_ref, k=50):
    rec = np.array(sorted(set(rank[:k]) & set(ref_rank[:k])), dtype=int)
    if rec.size == 0:
        return np.nan
    return float(np.mean(sign_S[rec] == sign_ref[rec]))


def recovery(rank, ref, sign_S):
    out = {}
    for k in KS:
        out[f"p{k}"] = precision_at(rank, ref["rank_limma"], k)
        out[f"p{k}_wil"] = precision_at(rank, ref["rank_wilcoxon"], k)
    out["rbo"] = rbo_ext(rank, ref["rank_limma"])
    out["rbo_wil"] = rbo_ext(rank, ref["rank_wilcoxon"])
    out["dir50"] = directional(rank, ref["rank_limma"], sign_S, np.sign(ref["t_limma"]))
    return out


def kuncheva(lists, n_total, k=50):
    """Mean pairwise Kuncheva index of top-k lists (paper §3.6)."""
    sets = [set(l[:k]) for l in lists]
    m = len(sets)
    if m < 2 or k >= n_total:
        return np.nan
    v = []
    for i in range(m):
        for j in range(i + 1, m):
            r = len(sets[i] & sets[j])
            v.append((r * n_total - k * k) / (k * (n_total - k)))
    return float(np.mean(v))
