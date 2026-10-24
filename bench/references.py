"""Full-data reference rankings per task (paper §3.6) and effect-size summary (Table 1).

  limma     R limma 3.62.2 lmFit + eBayes (default settings), unpaired, on the full
            task with missing values; ranked by |t|. BH-adjusted p kept so the
            number of proteins reaching q < 0.05 can be reported.
  wilcoxon  Mann-Whitney U per protein on observed values (sensitivity
            reference), ranked by p, ties broken by |rank-biserial r|.
  effect    Cohen's d per protein; the task summary is the median |d| over the
            top 100 proteins by limma |t|.

Computed once per task with R itself, so the reference is limma by construction.
Saved to data/derived/reference_<task>.npz.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from . import loaders
from .paths import DERIVED

R_SCRIPT = r"""
suppressMessages(library(limma)); library(jsonlite)
a <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(a[1], header = FALSE, na.strings = "nan")); y <- scan(a[2], quiet = TRUE)
fit <- eBayes(lmFit(t(X), cbind(1, y)))
p <- fit$p.value[, 2]
write(toJSON(list(t = unname(fit$t[, 2]), p = unname(p), q = unname(p.adjust(p, "BH")),
                  df_prior = fit$df.prior, limma = as.character(packageVersion("limma"))),
             digits = NA, na = "null"), a[3])
"""


def _limma_r(X, y):
    conda = shutil.which("conda") or "conda"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        np.savetxt(d / "X.csv", X, delimiter=",", fmt="%.8g")
        np.savetxt(d / "y.txt", y, fmt="%d")
        (d / "s.R").write_text(R_SCRIPT)
        r = subprocess.run([conda, "run", "-n", "smalln-r", "Rscript", str(d / "s.R"),
                            str(d / "X.csv"), str(d / "y.txt"), str(d / "o.json")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("R limma failed: " + r.stderr[-800:])
        o = json.loads((d / "o.json").read_text())
    f = lambda v: np.array([np.nan if x is None else x for x in v], dtype=float)
    return f(o["t"]), f(o["p"]), f(o["q"]), o["df_prior"], o["limma"]


def _order(score):
    """Indices sorted by descending score; NaN last."""
    s = np.where(np.isfinite(score), score, -np.inf)
    return np.argsort(-s, kind="stable")


def build(name):
    t = loaders.load(name)
    X, y = t.X.astype(float), t.y.astype(int)
    tl, pl, ql, df_prior, lv = _limma_r(X, y)

    pw = np.full(t.p, np.nan); rb = np.full(t.p, np.nan); d = np.full(t.p, np.nan)
    for j in range(t.p):
        a, b = X[y == 1, j], X[y == 0, j]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) >= 2 and len(b) >= 2:
            u = mannwhitneyu(a, b, alternative="two-sided")
            pw[j] = u.pvalue
            rb[j] = 2 * u.statistic / (len(a) * len(b)) - 1
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            d[j] = (a.mean() - b.mean()) / sp if sp > 0 else np.nan

    rank_limma = _order(np.abs(tl))
    # Wilcoxon: smaller p first, then larger |r|
    wscore = np.where(np.isfinite(pw), -np.log10(np.maximum(pw, 1e-300)) + 1e-6 * np.abs(rb), -np.inf)
    rank_wilcoxon = _order(wscore)
    top100 = rank_limma[:100]
    summary = dict(task=name, limma=lv, df_prior=df_prior,
                   n_q05=int(np.nansum(ql < 0.05)), n_tested=int(np.isfinite(tl).sum()),
                   median_abs_d_top100=float(np.nanmedian(np.abs(d[top100]))),
                   sign_agreement_limma_wilcoxon_top50=float(np.mean(np.sign(tl[rank_limma[:50]]) == np.sign(rb[rank_limma[:50]]))))
    np.savez_compressed(DERIVED / f"reference_{name}.npz", t_limma=tl, p_limma=pl, q_limma=ql,
                        p_wilcoxon=pw, r_wilcoxon=rb, cohen_d=d,
                        rank_limma=rank_limma, rank_wilcoxon=rank_wilcoxon)
    (DERIVED / f"reference_{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def load(name):
    p = DERIVED / f"reference_{name}.npz"
    if not p.exists():
        build(name)
    return dict(np.load(p))


if __name__ == "__main__":
    for n in loaders.TASKS:
        print(build(n))
