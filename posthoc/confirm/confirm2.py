"""Confirmation run 2 (plan: confirm2_plan.md, fixed before running). Stages:

  emb        build COEXP-d and ESMPC-d embeddings (+ shuffled controls) for B2 and B5, report coverage
  check      baseline and V2 runners reproduce stored values on repeats 0-9 (run 1's checks)
  baselines  CCP, EN, RF, TabPFN on repeats 200-299
  v2         V2 on repeats 200-299
  v3         V3 multiply at width d with each embedding and its shuffled control, then permuted-label checks

    python confirm2.py <smalln-bench> [stage ...]
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import confirm as c1                                                # run 1's verified runners (same argv)
from bench import cv, embeddings, gpu, loaders                      # noqa: E402
from bench.fusion import Batched, _vendored_model, EPOCHS, LR, WD    # noqa: E402
from bench.paths import RAW                                          # noqa: E402
from bench.preprocess import prepare                                 # noqa: E402
from bench.subsample import seed                                     # noqa: E402

warnings.simplefilter("ignore")
REPO = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[2])
HERE = Path(__file__).parent
EMB = HERE / "emb2"
R0, R = 200, 100
DIMS = (32, 64)
SOURCES = ("coexp", "esmpc")
CPTAC_EXT = ["Luad", "Lscc", "Ccrcc", "Hnscc", "Ucec", "Gbm", "Ov", "Brca"]
DEV = "cuda"


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


# ------------------------------------------------------------------ embeddings
def _zscore_cohort(X):
    """samples x proteins -> z per protein; proteins observed in < 50% of samples left missing."""
    obs = ~np.isnan(X)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1)
    Z = (X - mu) / np.where(sd > 0, sd, np.nan)
    Z[:, obs.mean(0) < 0.5] = np.nan
    return Z


def _svd_embed(A, covered, d):
    """A: proteins x samples (NaN = missing). Returns proteins x d, unit SD per dim over covered proteins."""
    A = np.nan_to_num(A[covered])
    A = A - A.mean(0)
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    E = U[:, :d] * S[:d]
    E /= E.std(0)
    out = np.zeros((len(covered), d), dtype=np.float32)
    out[covered] = E
    return out


def coexp_matrix(task):
    t = loaders.load(task)
    if task == "B2":
        import cptac
        ids = pd.Index([s.split(".")[0] for s in t.protein_ids])
        blocks = []
        for name in CPTAC_EXT:
            P = getattr(cptac, name)().get_proteomics("umich")
            P = P[~P.index.astype(str).str.endswith(".N")]
            cols = pd.Index(P.columns.get_level_values(1).astype(str).str.split(".").str[0])
            keep = ~cols.duplicated()
            Z = pd.DataFrame(_zscore_cohort(P.to_numpy(float)[:, keep]), columns=cols[keep])
            blocks.append(Z.reindex(columns=ids).to_numpy().T)          # proteins x samples
        return np.hstack(blocks)
    m = pd.read_csv(RAW / "ProCan-DepMapSanger_protein_matrix_8498_averaged.txt", sep="\t", low_memory=False)
    m.index = m.pop("Project_Identifier").str.split(";").str[0]
    ml = pd.read_csv(RAW / "model_list_20261010.csv", low_memory=False).set_index("model_id")
    ext = [i for i in m.index if i in ml.index and isinstance(ml.loc[i, "tissue"], str)
           and ml.loc[i, "tissue"] != "Haematopoietic and Lymphoid"]
    assert m.shape[1] == t.p                                          # task proteins are the matrix columns
    return _zscore_cohort(m.loc[ext].to_numpy(float)).T, len(ext)


def stage_emb():
    EMB.mkdir(exist_ok=True)
    for task in ("B2", "B5"):
        t = loaders.load(task)
        res = coexp_matrix(task)
        A, n_ext = (res, res.shape[1]) if task == "B2" else res
        cov_coexp = ~np.all(np.isnan(A), axis=1)
        E = np.nan_to_num(embeddings.load(task))
        for d in DIMS:
            for source, mat, cov in (("coexp", None, cov_coexp), ("esmpc", None, t.has_seq.astype(bool))):
                X = _svd_embed(A, cov, d) if source == "coexp" else c1.esm_pcs(E, t.has_seq, k=d).astype(np.float32)
                idx = np.flatnonzero(cov)
                Xs = X.copy()
                Xs[idx] = X[np.random.default_rng(seed("shuffle-emb", task, source, d)).permutation(idx)]
                np.save(EMB / f"{task}_{source}{d}.npy", X)
                np.save(EMB / f"{task}_{source}{d}_shuf.npy", Xs)
                log(f"emb {task} {source}{d}: coverage {cov.mean():.3f} of {t.p} proteins"
                    + (f" ({n_ext} external samples)" if source == "coexp" else ""))


# ------------------------------------------------------------------ V3 multiply at width d
WIDTH, P_MASK, LAMBDA, CHUNK = 1024, 0.25, 1.0, 50


class BatchedV3(Batched):
    def __init__(self, refs, daes):
        super().__init__(refs)
        st = lambda g: nn.Parameter(torch.stack([g(dd).detach().clone() for dd in daes]))
        self.D1w = st(lambda dd: dd[0].weight); self.D1b = st(lambda dd: dd[0].bias)
        self.D2w = st(lambda dd: dd[2].weight); self.D2b = st(lambda dd: dd[2].bias)
        self.D3w = st(lambda dd: dd[4].weight); self.D3b = st(lambda dd: dd[4].bias)

    def denoise(self, x):
        h = F.relu(self._lin(x.unsqueeze(1), self.D1w, self.D1b))
        h = F.relu(self._lin(h, self.D2w, self.D2b))
        return self._lin(h, self.D3w, self.D3b).squeeze(1)

    def fused(self, E, x):
        return E * torch.sigmoid(self._lin(F.relu(self._lin(x, self.W1, self.b1)), self.W2, self.b2))

    def predict_logits(self, E, x):
        zhat = self.denoise(x)
        return super().forward(E, zhat.unsqueeze(-1)), zhat


def train_batch(fits, E_task, d):
    Model = _vendored_model()
    refs, daes = [], []
    for f in fits:
        torch.manual_seed(f["seed"] % (2**63))
        refs.append(Model(num_classes=2, embedding_dim=d))
        P = f["Ztr"].shape[1]
        daes.append(nn.Sequential(nn.Linear(P, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, P)))
    net = BatchedV3(refs, daes).to(DEV); del refs, daes
    opt = torch.optim.RMSprop(net.parameters(), lr=LR, weight_decay=WD)
    M, n = len(fits), fits[0]["Ztr"].shape[0]
    Xtr = torch.tensor(np.stack([f["Ztr"] for f in fits]), dtype=torch.float32, device=DEV)
    ytr = torch.tensor(np.stack([f["ytr"] for f in fits]), dtype=torch.long, device=DEV)
    E = torch.tensor(np.stack([E_task[f["feat"]] for f in fits]), dtype=torch.float32, device=DEV)
    ar = torch.arange(M, device=DEV)
    g = torch.Generator(device=DEV); g.manual_seed(fits[0]["seed"] % (2**63))
    torch.manual_seed(fits[0]["seed"] % (2**63) + 1)
    net.train()
    for _ in range(EPOCHS):
        order = torch.argsort(torch.rand(M, n, generator=g, device=DEV), dim=1)
        for step in range(n):
            idx = order[:, step]
            x = Xtr[ar, idx]
            keep = (torch.rand(x.shape, generator=g, device=DEV) >= P_MASK).float()
            logits, zhat = net.predict_logits(E, x * keep)
            loss = F.cross_entropy(logits, ytr[ar, idx], reduction="sum") + LAMBDA * ((zhat - x) ** 2).mean(dim=1).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        m_max = max(f["Zte"].shape[0] for f in fits)
        Zte = np.zeros((M, m_max, E.shape[1]), dtype=np.float32)
        for k, f in enumerate(fits):
            Zte[k, : f["Zte"].shape[0]] = f["Zte"]
        Zte = torch.tensor(Zte, device=DEV)
        probs = np.zeros((M, m_max), dtype=np.float32)
        for j in range(m_max):
            probs[:, j] = torch.softmax(net.predict_logits(E, Zte[:, j])[0], dim=1)[:, 1].cpu().numpy()
    return [probs[k, : f["Zte"].shape[0]] for k, f in enumerate(fits)]


def v3(task, N, arm, label, d):
    E = np.load(EMB / f"{task}_{label}.npy")
    fits, yH = [], []
    for r in range(R0, R0 + R):
        _, ctx, S, H, yS, yh, coords = c1.unit(task, N, r, arm)
        p = prepare(ctx, S, H, yS, "ref", seed("impute", *coords, "ext", 0), restrict_seq=True)
        fits.append(dict(Ztr=p.Ztr.astype(np.float32), ytr=yS, Zte=p.Zte.astype(np.float32), feat=p.feat,
                         seed=seed("model", "fusion-v3-run2", d, *coords)))     # shared by real and shuffled
        yH.append(yh)
    probs = []
    for b in range(0, len(fits), CHUNK):
        probs += train_batch(fits[b:b + CHUNK], E, d)
        torch.cuda.empty_cache()
    return [dict(task=task, N=N, r=R0 + i, arm=arm, model=label, auc_ext=cv.external_auc(yh, s))
            for i, (yh, s) in enumerate(zip(yH, probs))]


def stage_v3():
    gpu.guard("fusion")
    f = HERE / "confirm2_v3.csv"
    done = pd.read_csv(f, dtype={"arm": str}) if f.exists() else pd.DataFrame()
    labels = [(f"{s}{d}{suf}", d) for s in SOURCES for d in DIMS for suf in ("", "_shuf")]
    plan = [(t, N, "true", lab, d) for lab, d in labels for t, N in c1.CELLS] + \
           [(t, N, "perm", f"{s}{d}", d) for s in SOURCES for d in DIMS for t, N in c1.PERM_CELLS]
    for task, N, arm, lab, d in plan:
        if len(done) and ((done.task == task) & (done.N == N) & (done.arm == arm) & (done.model == lab)).any():
            continue
        t0 = time.time()
        done = pd.concat([done, pd.DataFrame(v3(task, N, arm, lab, d))], ignore_index=True)
        done.to_csv(f, index=False)
        log(f"v3 {lab} {task} N={N} {arm}: {time.time() - t0:.0f}s")


def stage_table(name, fn):
    f = HERE / f"confirm2_{name}.csv"
    done = pd.read_csv(f, dtype={"arm": str}) if f.exists() else pd.DataFrame()
    for task, N in c1.CELLS:
        if len(done) and ((done.task == task) & (done.N == N)).any():
            continue
        done = pd.concat([done, pd.DataFrame(fn(task, N, range(R0, R0 + R)))], ignore_index=True)
        done.to_csv(f, index=False)
        log(f"{name} {task} N={N} done")


if __name__ == "__main__":
    stages = sys.argv[2:] or ["emb", "check", "baselines", "v2", "v3"]
    log(f"stages {stages}")
    for s in stages:
        if s == "emb":
            stage_emb()
        elif s == "check":
            c1.stage_check()
        elif s == "baselines":
            gpu.guard("tabpfn"); stage_table("baselines", c1.baselines)
        elif s == "v2":
            stage_table("v2", c1.v2)
        elif s == "v3":
            stage_v3()
    log("all stages done")
