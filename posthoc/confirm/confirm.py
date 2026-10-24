"""Confirmation run (plan: confirm_plan.md, fixed before running). Stages, in order:

  check      external-only baseline runner and V2 runner reproduce stored values on repeats 0-9
  baselines  CCP, EN, RF, TabPFN held-out AUC on repeats 100-199
  v2         V2 (EIL) on repeats 100-199
  v3         V3-ESM and V3-shuffled (multiply) on repeats 100-199, plus the permuted-label check

    python confirm.py <smalln-bench> [stage ...]
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from bench import cv, embeddings, gpu, loaders                     # noqa: E402
from bench.fusion import Batched, _vendored_model, EPOCHS, LR, WD   # noqa: E402
from bench.models import fit_predict                                # noqa: E402
from bench.preprocess import prepare                                # noqa: E402
from bench.run import context                                       # noqa: E402
from bench.subsample import draw, permuted_labels, seed             # noqa: E402
from sklearn.linear_model import LogisticRegression                 # noqa: E402
from sklearn.exceptions import ConvergenceWarning                   # noqa: E402

warnings.simplefilter("ignore", ConvergenceWarning)
HERE = Path(__file__).parent
R0, R = 100, 100
CELLS = [("B2", 6), ("B2", 12), ("B2", 30), ("B5", 6), ("B5", 12), ("B5", 30), ("B5", 50)]
PERM_CELLS = [("B2", 12), ("B5", 12)]
BASE = [(1, "ccp", "CCP"), (2, "en", "EN"), (3, "rf", "RF"), (4, "tabpfn", "TabPFN")]
DEV = "cuda"


def unit(task, N, r, arm):
    t, ctx, _ = context(task)
    S, H = draw(t, N, r)
    yS = t.y[S].astype(int)
    if arm == "perm":
        yS = permuted_labels(yS, task, N, r)
    return t, ctx, S, H, yS, t.y[H].astype(int), (task, N, r, arm)


# ------------------------------------------------------------------ baselines (external fit only, as bench.run)
def baselines(task, N, rs, arm="true"):
    rows = []
    for r in rs:
        t, ctx, S, H, yS, yH, coords = unit(task, N, r, arm)
        p = prepare(ctx, S, H, yS, "ref", seed("impute", *coords, "ext", 0))
        row = dict(task=task, N=N, r=r, arm=arm)
        for cid, model, name in BASE:
            s, _ = fit_predict(model, p.Ztr, yS, p.Zte, p.t_feat, seed("model", cid, *coords, "ext", 0))
            row[name] = cv.external_auc(yH, s)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ V2 (EIL), as pilot_fusion_v2.py
def esm_pcs(E, has_seq, k=32):
    X = E[has_seq]; mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    pcs = (E - mu) @ Vt[:k].T
    pcs /= pcs[has_seq].std(0)
    pcs[~has_seq] = 0.0
    return pcs


def v2(task, N, rs, arm="true"):
    t = loaders.load(task)
    E = np.nan_to_num(np.load(f"{REPO}/data/derived/esm2_cls_{task}.npy"))
    PC = esm_pcs(E, t.has_seq)
    rows = []
    for r in rs:
        _, ctx, S, H, yS, yH, coords = unit(task, N, r, arm)
        pf = prepare(ctx, S, H, yS, "ref", seed("impute", *coords, "ext", 0), restrict_seq=True)
        Pm = PC[pf.feat] / np.sqrt(pf.Ztr.shape[1])
        Xtr, Xte = np.hstack([pf.Ztr, pf.Ztr @ Pm]), np.hstack([pf.Zte, pf.Zte @ Pm])
        m = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=5000).fit(Xtr, yS)
        rows.append(dict(task=task, N=N, r=r, arm=arm, V2=cv.external_auc(yH, m.decision_function(Xte))))
    return rows


# ------------------------------------------------------------------ V3 multiply (as pilot_fusion_v3.py)
WIDTH, P_MASK, LAMBDA, CHUNK = 1024, 0.25, 1.0, 50


class BatchedV3(Batched):
    def __init__(self, refs, daes):
        super().__init__(refs)
        st = lambda g: nn.Parameter(torch.stack([g(d).detach().clone() for d in daes]))
        self.D1w = st(lambda d: d[0].weight); self.D1b = st(lambda d: d[0].bias)
        self.D2w = st(lambda d: d[2].weight); self.D2b = st(lambda d: d[2].bias)
        self.D3w = st(lambda d: d[4].weight); self.D3b = st(lambda d: d[4].bias)

    def denoise(self, x):
        h = F.relu(self._lin(x.unsqueeze(1), self.D1w, self.D1b))
        h = F.relu(self._lin(h, self.D2w, self.D2b))
        return self._lin(h, self.D3w, self.D3b).squeeze(1)

    def fused(self, E, x):
        gate = torch.sigmoid(self._lin(F.relu(self._lin(x, self.W1, self.b1)), self.W2, self.b2))
        return E * gate

    def predict_logits(self, E, x):
        zhat = self.denoise(x)
        return super().forward(E, zhat.unsqueeze(-1)), zhat


def train_batch(fits, E_task):
    Model = _vendored_model()
    refs, daes = [], []
    for f in fits:
        torch.manual_seed(f["seed"] % (2**63))
        refs.append(Model(num_classes=2, embedding_dim=480))
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


def embedding(task, kind):
    E = np.nan_to_num(embeddings.load(task))
    if kind == "shuffled":
        has = loaders.load(task).has_seq
        idx = np.flatnonzero(has)
        E = E.copy()
        E[idx] = E[np.random.default_rng(seed("shuffle-esm", task)).permutation(idx)]
    return E


def v3(task, N, arm, kind):
    E = embedding(task, kind)
    fits, yH = [], []
    for r in range(R0, R0 + R):
        _, ctx, S, H, yS, yh, coords = unit(task, N, r, arm)
        p = prepare(ctx, S, H, yS, "ref", seed("impute", *coords, "ext", 0), restrict_seq=True)
        fits.append(dict(Ztr=p.Ztr.astype(np.float32), ytr=yS, Zte=p.Zte.astype(np.float32), feat=p.feat,
                         seed=seed("model", "fusion-v3", "multiply", *coords)))    # same seeds for both kinds
        yH.append(yh)
    probs = []
    for b in range(0, len(fits), CHUNK):
        probs += train_batch(fits[b:b + CHUNK], E)
        torch.cuda.empty_cache()
    return [dict(task=task, N=N, r=R0 + i, arm=arm, model=f"V3_{kind}", auc_ext=cv.external_auc(yh, s))
            for i, (yh, s) in enumerate(zip(yH, probs))]


# ------------------------------------------------------------------ stages
def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def stage_check():
    stored = pd.read_parquet(f"{REPO}/results/main_all_runs.parquet", columns=["task", "N", "r", "arm", "config", "auc_ext"])
    pilot = pd.read_csv(HERE.parent / "pilots" / "pilot.csv")
    gpu.guard("tabpfn")
    ok = True
    for task, N in (("B2", 12), ("B5", 30)):
        new = pd.DataFrame(baselines(task, N, range(10)))
        for _, _, name in BASE:
            ref = stored[(stored.task == task) & (stored.N == N) & (stored.arm == "true") & (stored.config == name)].set_index("r").auc_ext
            d = float(np.max(np.abs(new.set_index("r")[name] - ref.loc[range(10)])))
            log(f"check baseline {name} {task} N={N} r0-9: max |diff| {d:.3g}"); ok &= d < 1e-12
        nv = pd.DataFrame(v2(task, N, range(10))).set_index("r").V2
        pv = pilot[(pilot.task == task) & (pilot.N == N) & (pilot.arm == "true")].set_index("r").EIL.loc[range(10)]
        d = float(np.max(np.abs(nv - pv)))
        log(f"check V2 {task} N={N} r0-9: max |diff| {d:.3g}"); ok &= d < 1e-12
    if not ok:
        raise SystemExit("reproduction check FAILED -- stopping as the plan requires")
    log("checks passed")


def stage_table(name, fn):
    f = HERE / f"confirm_{name}.csv"
    done = pd.read_csv(f) if f.exists() else pd.DataFrame()
    for task, N in CELLS:
        if len(done) and ((done.task == task) & (done.N == N)).any():
            continue
        t0 = time.time()
        rows = pd.DataFrame(fn(task, N, range(R0, R0 + R)))
        done = pd.concat([done, rows], ignore_index=True); done.to_csv(f, index=False)
        log(f"{name} {task} N={N}: {time.time() - t0:.0f}s")


def stage_v3():
    gpu.guard("fusion")
    f = HERE / "confirm_v3.csv"
    done = pd.read_csv(f) if f.exists() else pd.DataFrame()
    plan = [(t, N, "true", k) for k in ("esm", "shuffled") for t, N in CELLS] + \
           [(t, N, "perm", k) for k in ("esm", "shuffled") for t, N in PERM_CELLS]
    for task, N, arm, kind in plan:
        if len(done) and ((done.task == task) & (done.N == N) & (done.arm == arm) & (done.model == f"V3_{kind}")).any():
            continue
        t0 = time.time()
        done = pd.concat([done, pd.DataFrame(v3(task, N, arm, kind))], ignore_index=True); done.to_csv(f, index=False)
        log(f"v3 {kind} {task} N={N} {arm}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    stages = sys.argv[2:] or ["check", "baselines", "v2", "v3"]
    log(f"stages {stages}")
    for s in stages:
        if s == "check":
            stage_check()
        elif s == "baselines":
            gpu.guard("tabpfn"); stage_table("baselines", baselines)
        elif s == "v2":
            stage_table("v2", v2)
        elif s == "v3":
            stage_v3()
    log("all stages done")
