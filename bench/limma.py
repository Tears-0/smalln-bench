"""Two-group moderated t (limma lmFit + eBayes, Smyth 2004), in numpy.

R is too slow to call per fold at 10^5-10^6 fits, so limma's empirical-Bayes
step is reimplemented here and checked against R limma 3.62.2 by
tests/test_limma.py. Missing values are handled per protein, as lmFit does with
NAs: each protein uses its observed samples, so its residual df varies.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, polygamma


def _trigamma(x):
    return polygamma(1, x)


def trigamma_inverse(x):
    """Inverse of trigamma, Newton iteration as in limma::trigammaInverse."""
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    big, small = x > 1e7, x < 1e-6
    y[big] = 1.0 / np.sqrt(x[big])
    y[small] = 1.0 / x[small]
    mid = ~(big | small)
    if mid.any():
        xm = x[mid]
        ym = 0.5 + 1.0 / xm
        for _ in range(50):
            tri = _trigamma(ym)
            dif = tri * (1 - tri / xm) / polygamma(2, ym)
            ym = ym + dif
            if np.max(-dif / ym) < 1e-8:
                break
        y[mid] = ym
    return y


def fit_f_dist(s2, df):
    """Moment estimation of the scaled-F prior (limma::fitFDist, no covariate)."""
    ok = np.isfinite(s2) & (df > 0) & np.isfinite(df)
    x, d = s2[ok], df[ok]
    if x.size < 2:
        return np.inf, float(np.median(x)) if x.size else np.nan
    x = np.maximum(x, 1e-5 * np.median(x))
    z = np.log(x)
    e = z - digamma(d / 2) + np.log(d / 2)
    emean = e.mean()
    evar = ((e - emean) ** 2).sum() / (e.size - 1) - _trigamma(d / 2).mean()
    if evar > 0:
        df0 = 2 * float(trigamma_inverse(np.array([evar]))[0])
        s20 = float(np.exp(emean + digamma(df0 / 2) - np.log(df0 / 2)))
    else:
        df0, s20 = np.inf, float(np.exp(emean))
    return df0, s20


def moderated_t(X, y):
    """Moderated t for group 1 minus group 0, per column of X [n, p].

    NaN entries are ignored per column. Columns without at least one observation
    in each group and one residual df get t = NaN.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(bool)
    obs = ~np.isnan(X)
    Xz = np.where(obs, X, 0.0)
    n1 = (obs & y[:, None]).sum(0)
    n0 = (obs & ~y[:, None]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (Xz * y[:, None]).sum(0) / n1
        m0 = (Xz * ~y[:, None]).sum(0) / n0
        mu = np.where(y[:, None], m1, m0)
        rss = (np.where(obs, X - mu, 0.0) ** 2).sum(0)
        df = (n1 + n0 - 2).astype(float)
        s2 = rss / df
    valid = (n1 >= 1) & (n0 >= 1) & (df >= 1)
    s2 = np.where(valid, s2, np.nan)
    df = np.where(valid, df, np.nan)
    df0, s20 = fit_f_dist(s2, df)
    if np.isinf(df0):
        s2_post = np.full_like(s2, s20)
    else:
        s2_post = (df0 * s20 + df * s2) / (df0 + df)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (m1 - m0) / np.sqrt(s2_post * (1.0 / n1 + 1.0 / n0))
    t[~valid] = np.nan
    return t


def moderated_t_complete(X, y):
    """Fast path for complete matrices (no NaN): identical result, fewer passes."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    m1, m0 = X[y].mean(0), X[~y].mean(0)
    rss = ((X[y] - m1) ** 2).sum(0) + ((X[~y] - m0) ** 2).sum(0)
    d = float(n1 + n0 - 2)
    s2 = rss / d
    df0, s20 = fit_f_dist(s2, np.full_like(s2, d))
    s2_post = np.full_like(s2, s20) if np.isinf(df0) else (df0 * s20 + d * s2) / (df0 + d)
    return (m1 - m0) / np.sqrt(s2_post * (1.0 / n1 + 1.0 / n0))
