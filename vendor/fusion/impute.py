"""Fold-local imputation strategies.

Three modes are provided so that the imputation choice can be ablated
(rebuttal task 2A.2) and its parameters swept (task 2D):

  'asrun'      Bit-faithful reproduction of fit_sfi_hybrid/transform_sfi_hybrid
               from Transferlearning(dl).ipynb -- the code that produced the
               submitted results.  NOTE: with 2-3 replicates per group the
               QRILC/MNAR branch provably never writes a value (see
               audit_imputation.py), so this mode is group-wise KNN in practice.

  'sfi_paper'  SFI-Hybrid exactly as described in the manuscript (Eqs. 1-5):
               a protein-wise (row-wise) left-censored MNAR draw
                   x_hat ~ N(mu_obs - d*sigma_obs, (w*sigma_obs)^2)
               with mu_obs / sigma_obs estimated from the training fold only.

  'columnwise' The sample-wise strategy the manuscript argues against: the
               downshifted Gaussian is parameterised from the distribution of
               all protein intensities within a given sample (Perseus default).

Every mode is leakage-aware: distribution parameters and the missingness
filter are estimated on training columns only and then applied to the
held-out columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

MODES = ("asrun", "sfi_paper", "columnwise")


# --------------------------------------------------------------------------
# 'asrun' -- faithful reproduction of the submitted pipeline
# --------------------------------------------------------------------------
def _qrilc_series(series: pd.Series, downshift: float, width: float, random_state: int) -> pd.Series:
    """Original qrilc_impute_series. Returns the series unchanged when the
    group has no observed value to estimate mu/sigma from."""
    out = series.copy()
    valid = out.dropna()
    if len(valid) == 0:
        return series
    mu = valid.mean() - downshift * valid.std()
    sd = valid.std() * width
    mask = out.isna()
    n = int(mask.sum())
    if n > 0:
        np.random.seed(random_state)
        out[mask] = np.random.normal(loc=mu, scale=sd, size=n)
    return out


def _asrun_impute(df: pd.DataFrame, group_dict: dict, downshift: float, width: float,
                  knn_neighbors: int, seed_base: int) -> pd.DataFrame:
    """Group-local QRILC-then-KNN, as in the notebook."""
    out = df.copy()
    for cols in group_dict.values():
        if not cols:
            continue
        block = out[cols]
        n_missing = block.isna().sum(axis=1).to_numpy()
        total = len(cols)
        trigger = (n_missing == total) | (n_missing / total >= 0.7)
        # Rows that trigger MNAR. For group sizes <= 3 the trigger implies
        # n_missing == total, i.e. no observed value, so _qrilc_series is a
        # no-op; the loop is kept so the branch stays correct for larger groups.
        for pos in np.flatnonzero(trigger):
            idx = out.index[pos]
            if n_missing[pos] == total:
                continue  # provably a no-op: nothing observed to estimate from
            out.loc[idx, cols] = _qrilc_series(
                out.loc[idx, cols], downshift, width, seed_base + idx
            )
        if out[cols].isna().to_numpy().sum() > 0:
            imp = KNNImputer(n_neighbors=min(knn_neighbors, len(cols)), weights="distance")
            out[cols] = imp.fit_transform(out[cols].to_numpy())
    return out


# --------------------------------------------------------------------------
# 'sfi_paper' -- SFI-Hybrid as written in the manuscript
# --------------------------------------------------------------------------
def _protein_params(df: pd.DataFrame, train_cols: list, downshift: float, width: float):
    """mu_imp / sigma_imp per protein, estimated on training columns only."""
    obs = df[train_cols].to_numpy(dtype=float)
    mu = np.nanmean(obs, axis=1)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(obs, axis=1, ddof=1)
    sd = np.where(np.isfinite(sd), sd, 0.0)
    return mu - downshift * sd, np.maximum(width * sd, 1e-8)


def _draw(values: np.ndarray, mu_imp: np.ndarray, sd_imp: np.ndarray, rng: np.random.Generator):
    out = values.copy()
    miss = np.isnan(out)
    if miss.any():
        rows = np.nonzero(miss)[0] if out.ndim == 1 else np.nonzero(miss)[0]
        out[miss] = rng.normal(loc=mu_imp[rows], scale=sd_imp[rows])
    return out


# --------------------------------------------------------------------------
# 'columnwise' -- per-sample downshifted Gaussian (Perseus default)
# --------------------------------------------------------------------------
def _column_params(col: np.ndarray, downshift: float, width: float):
    valid = col[~np.isnan(col)]
    if valid.size < 2:
        return 0.0, 1e-8
    mu, sd = valid.mean(), valid.std(ddof=1)
    return mu - downshift * sd, max(width * sd, 1e-8)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def prepare_fold(df: pd.DataFrame, train_groups: dict, val_groups: dict, *,
                 mode: str = "asrun", downshift: float = 1.5, width: float = 0.3,
                 knn_neighbors: int = 3, seed: int = 42, protein_col: str = "Accession"):
    """Filter + impute one cross-validation fold.

    Returns (train_df, val_df, kept_accessions). Both frames carry the same
    rows in the same order, so 'Original_Index' aligns them to the ESM-2
    embedding tensor.
    """
    if mode not in MODES:
        raise ValueError(f"unknown imputation mode {mode!r}; expected one of {MODES}")

    train_cols = [c for cols in train_groups.values() for c in cols]
    val_cols = [c for cols in val_groups.values() for c in cols]

    # Missingness filter -- training columns only (leakage-aware).
    keep = df[train_cols].notna().sum(axis=1) >= 2
    base = df[keep].reset_index(drop=True)
    kept = base[protein_col].tolist()

    if mode == "asrun":
        train_df = _asrun_impute(base, train_groups, downshift, width, knn_neighbors, seed)
        val_df = _asrun_impute(base, val_groups, downshift, width, knn_neighbors, seed + 57)
        return train_df, val_df, kept

    train_df = base.copy()
    val_df = base.copy()
    rng = np.random.default_rng(seed)

    if mode == "sfi_paper":
        mu_imp, sd_imp = _protein_params(base, train_cols, downshift, width)
        for col in train_cols:
            train_df[col] = _draw(base[col].to_numpy(dtype=float), mu_imp, sd_imp, rng)
        # Held-out columns imputed with the *training* parameters.
        for col in val_cols:
            val_df[col] = _draw(base[col].to_numpy(dtype=float), mu_imp, sd_imp, rng)
        return train_df, val_df, kept

    # columnwise: parameters from each sample's own protein-intensity distribution.
    for col in train_cols:
        vals = base[col].to_numpy(dtype=float)
        mu, sd = _column_params(vals, downshift, width)
        miss = np.isnan(vals)
        vals = vals.copy()
        vals[miss] = rng.normal(mu, sd, int(miss.sum()))
        train_df[col] = vals
    for col in val_cols:
        vals = base[col].to_numpy(dtype=float)
        mu, sd = _column_params(vals, downshift, width)
        miss = np.isnan(vals)
        vals = vals.copy()
        vals[miss] = rng.normal(mu, sd, int(miss.sum()))
        val_df[col] = vals
    return train_df, val_df, kept
