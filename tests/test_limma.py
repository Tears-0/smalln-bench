"""Validate bench.limma against R limma (a one-off check before the benchmark runs).

Requires the conda env 'smalln-r' (R 4.4.3, limma 3.62.2); skipped otherwise.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bench.limma import moderated_t, moderated_t_complete

R_SCRIPT = r"""
suppressMessages(library(limma)); library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(args[1], header = FALSE)); y <- scan(args[2], quiet = TRUE)
design <- cbind(1, y)
# legacy = TRUE: moment estimation of the prior even when residual df differ
# between proteins (missing values). limma's default then switches to an ML
# estimator; bench.limma implements the moment method, and the full-data
# reference ranking (paper §3.6) is computed by R limma itself (bench.references).
fit <- eBayes(lmFit(t(X), design), legacy = TRUE)
write(toJSON(list(t = unname(fit$t[, 2]), df0 = fit$df.prior, s20 = fit$s2.prior), digits = NA, na = "null"), args[3])
"""


def _run_r(X, y):
    conda = shutil.which("conda")
    if conda is None:
        pytest.skip("conda not available")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        np.savetxt(d / "X.csv", X, delimiter=",", fmt="%.10g")
        np.savetxt(d / "y.txt", y, fmt="%d")
        (d / "s.R").write_text(R_SCRIPT)
        r = subprocess.run([conda, "run", "-n", "smalln-r", "Rscript", str(d / "s.R"),
                            str(d / "X.csv"), str(d / "y.txt"), str(d / "o.json")],
                           capture_output=True, text=True, shell=False)
        if r.returncode != 0:
            pytest.skip("R/limma unavailable: " + r.stderr[-300:])
        out = json.loads((d / "o.json").read_text())
    return np.array([np.nan if v is None else v for v in out["t"]], dtype=float)


def _fixture(seed, n1, n0, p, miss):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n1), np.zeros(n0)].astype(int)
    X = rng.normal(size=(n1 + n0, p)) * rng.uniform(0.3, 2.0, size=p)
    X[y == 1, : p // 10] += 1.0
    if miss:
        X[rng.random(X.shape) < miss] = np.nan
    return X, y


@pytest.mark.parametrize("seed,n1,n0,p,miss", [(0, 6, 6, 400, 0.0), (1, 5, 4, 300, 0.0),
                                               (2, 25, 25, 500, 0.0), (3, 40, 30, 600, 0.25)])
def test_against_r(seed, n1, n0, p, miss):
    X, y = _fixture(seed, n1, n0, p, miss)
    t_r = _run_r(X, y)
    t_py = moderated_t(X, y)
    ok = np.isfinite(t_r)
    assert np.array_equal(ok, np.isfinite(t_py)), "NaN pattern differs from limma"
    np.testing.assert_allclose(t_py[ok], t_r[ok], rtol=1e-6, atol=1e-8)
    if miss == 0:
        np.testing.assert_allclose(moderated_t_complete(X, y), t_r, rtol=1e-6, atol=1e-8)
