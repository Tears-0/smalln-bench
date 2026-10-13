"""Model arms (spec §5.1). No hyperparameter is tuned (F6).

fit_predict returns (scores for the test rows, importance over the selected
features). Importance drives the protein ranking used for recovery (§6.2);
TabPFN returns None (excluded from recovery, F8).
"""
from __future__ import annotations

import warnings

import numpy as np

CCP_K = 50
_TABPFN = {}


def smote(Z, y, sd):
    """Oversample both classes to 2x (F12); k = min(5, n_min - 1)."""
    from imblearn.over_sampling import SMOTE
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    k = min(5, min(n1, n0) - 1)
    if k < 1:
        return Z, y
    return SMOTE(sampling_strategy={0: 2 * n0, 1: 2 * n1}, k_neighbors=k,
                 random_state=sd % (2**32)).fit_resample(Z, y)


def _ccp(Ztr, ytr, Zte, t):
    k = min(CCP_K, t.size)
    top = np.argsort(-np.abs(t), kind="stable")[:k]
    w = np.sign(t[top])
    return Zte[:, top] @ w, np.abs(t)


def _en(Ztr, ytr, Zte, t, sd):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LogisticRegression(l1_ratio=0.5, C=1.0, solver="saga", max_iter=10000,
                               random_state=sd % (2**32)).fit(Ztr, ytr)
    coef = np.abs(m.coef_.ravel())
    # zero coefficients rank after every non-zero one, ordered among themselves by |t|
    imp = np.where(coef > 0, 1.0 + coef, 1e-3 * np.abs(t) / (np.abs(t).max() + 1e-12))
    return m.decision_function(Zte), imp


def _rf(Ztr, ytr, Zte, sd):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=1,
                               random_state=sd % (2**32)).fit(Ztr, ytr)
    return m.predict_proba(Zte)[:, 1], m.feature_importances_


def _tabpfn(Ztr, ytr, Zte, sd):
    from tabpfn import TabPFNClassifier
    from .paths import TABPFN_CKPT
    m = TabPFNClassifier(model_path=str(TABPFN_CKPT), device="cuda", random_state=sd % (2**32))
    m.fit(Ztr, ytr)
    return m.predict_proba(Zte)[:, 1], None


def fit_predict(model, Ztr, ytr, Zte, t, sd, smote_in_fold=False):
    if smote_in_fold:
        Ztr, ytr = smote(Ztr, ytr, sd)
    if model == "ccp":
        return _ccp(Ztr, ytr, Zte, t)
    if model == "en":
        return _en(Ztr, ytr, Zte, t, sd)
    if model == "rf":
        return _rf(Ztr, ytr, Zte, sd)
    if model == "tabpfn":
        return _tabpfn(Ztr, ytr, Zte, sd)
    raise ValueError(model)
