"""Pilot: FusionV3 = V1 (the frozen case-study fusion network) with a denoising front end.

Exploratory and post hoc, not part of the paper. Same draws, same in-fold reference
preprocessing (restricted to proteins with a sequence) and the same held-out sets as V1's
external fit in the benchmark, so the stored V1 held-out AUC is the paired comparator.

Architecture, fixed before running:
  denoiser   per sample, the P = 500 in-fold abundances z -> Linear(P, 1024) - ReLU -
             Linear(1024, 1024) - ReLU - Linear(1024, P) -> z_hat. Trained as a denoising
             autoencoder: 25% of inputs masked to 0 (the training mean in z units),
             reconstruction target the clean z, MSE averaged over proteins, weight 1.
  encoder    V1's per-protein abundance encoder on z_hat: Linear(1, 128) - ReLU -
             Linear(128, 480) - sigmoid  (the protein gate)
  fusion     add:      f = e + gate   (V1's form)
             multiply: f = e * gate
  rest       V1 unchanged: attention pooling, Linear(480, 256) - ReLU - Dropout(0.5) -
             Linear(256, 2); unweighted cross-entropy; RMSprop lr 1e-4, weight decay 1e-4;
             1,000 epochs; batch size 1 per model; no masking at prediction time.

    python pilot_fusion_v3.py <smalln-bench> <R> <out.csv> [smoke]
"""
import sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from bench import cv, embeddings, gpu                               # noqa: E402
from bench.fusion import Batched, _vendored_model, EPOCHS, LR, WD   # noqa: E402
from bench.run import context                                       # noqa: E402
from bench.preprocess import prepare                                # noqa: E402
from bench.subsample import draw, permuted_labels, seed              # noqa: E402

R = int(sys.argv[2])
WIDTH, P_MASK, LAMBDA = 1024, 0.25, 1.0
CHUNK = 50
MODES = tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 else ("add", "multiply")
DEV = "cuda"


class BatchedV3(Batched):
    def __init__(self, refs, daes, mode):
        super().__init__(refs)
        st = lambda g: nn.Parameter(torch.stack([g(d).detach().clone() for d in daes]))
        self.D1w = st(lambda d: d[0].weight); self.D1b = st(lambda d: d[0].bias)
        self.D2w = st(lambda d: d[2].weight); self.D2b = st(lambda d: d[2].bias)
        self.D3w = st(lambda d: d[4].weight); self.D3b = st(lambda d: d[4].bias)
        self.mode = mode

    def denoise(self, x):                                  # x [M, P] -> [M, P]
        h = F.relu(self._lin(x.unsqueeze(1), self.D1w, self.D1b))
        h = F.relu(self._lin(h, self.D2w, self.D2b))
        return self._lin(h, self.D3w, self.D3b).squeeze(1)

    def fused(self, E, x):                                 # x [M, P, 1] (denoised)
        gate = torch.sigmoid(self._lin(F.relu(self._lin(x, self.W1, self.b1)), self.W2, self.b2))
        if self.mode == "noesm":                           # ESM-2 removed: the gate alone (V1's use_esm=False form)
            return gate
        return E + gate if self.mode == "add" else E * gate

    def predict_logits(self, E, x):                        # x [M, P]
        zhat = self.denoise(x)
        return super().forward(E, zhat.unsqueeze(-1)), zhat


def train_batch(fits, E_task, mode):
    Model = _vendored_model()
    refs, daes = [], []
    for f in fits:
        torch.manual_seed(f["seed"] % (2**63))
        refs.append(Model(num_classes=2, embedding_dim=480))
        P = f["Ztr"].shape[1]
        daes.append(nn.Sequential(nn.Linear(P, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(),
                                  nn.Linear(WIDTH, P)))
    net = BatchedV3(refs, daes, mode).to(DEV); del refs, daes
    opt = torch.optim.RMSprop(net.parameters(), lr=LR, weight_decay=WD)
    M, n = len(fits), fits[0]["Ztr"].shape[0]
    Xtr = torch.tensor(np.stack([f["Ztr"] for f in fits]), dtype=torch.float32, device=DEV)   # [M, n, P]
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
            x = Xtr[ar, idx]                                                   # [M, P]
            keep = (torch.rand(x.shape, generator=g, device=DEV) >= P_MASK).float()
            logits, zhat = net.predict_logits(E, x * keep)
            loss = (F.cross_entropy(logits, ytr[ar, idx], reduction="sum")
                    + LAMBDA * ((zhat - x) ** 2).mean(dim=1).sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    out = []
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


def cell(task, N, arm, mode, E_task):
    t, ctx, _ = context(task)
    fits, yH = [], []
    for r in range(R):
        S, H = draw(t, N, r)
        yS = t.y[S].astype(int)
        if arm == "perm":
            yS = permuted_labels(yS, task, N, r)
        coords = (task, N, r, arm)
        p = prepare(ctx, S, H, yS, "ref", seed("impute", *coords, "ext", 0), restrict_seq=True)
        # the no-ESM ablation reuses V3-add's seeds (same initial weights, masks and sample order),
        # so the embedding is the only difference between the two
        fits.append(dict(Ztr=p.Ztr.astype(np.float32), ytr=yS, Zte=p.Zte.astype(np.float32), feat=p.feat,
                         seed=seed("model", "fusion-v3", "add" if mode == "noesm" else mode, *coords)))
        yH.append(t.y[H].astype(int))
    probs = []
    for b in range(0, len(fits), CHUNK):                  # 100 models with the denoiser exceed the 4 GB cap
        probs += train_batch(fits[b:b + CHUNK], E_task, mode)
        torch.cuda.empty_cache()
    return [dict(task=task, N=N, r=r, arm=arm, mode=mode, auc_ext=cv.external_auc(yh, s))
            for r, (yh, s) in enumerate(zip(yH, probs))]


if __name__ == "__main__":
    print(f"GPU cap {gpu.guard('fusion'):.1f} GB", flush=True)
    smoke = len(sys.argv) > 4 and sys.argv[4] == "smoke"
    plan = [("B2", 6, "true")] if smoke else (
        [("B2", N, "true") for N in (6, 12, 30)] + [("B5", N, "true") for N in (6, 12, 30, 50)]
        + [("B2", 12, "perm"), ("B5", 12, "perm")])
    rows = []
    E = {t: np.nan_to_num(embeddings.load(t)) for t in ("B2", "B5")}
    for mode in MODES:
        for task, N, arm in plan:
            t0 = time.time()
            rows += cell(task, N, arm, mode, E[task])
            print(f"{time.strftime('%H:%M:%S')} {mode} {task} N={N} {arm}: {time.time() - t0:.0f}s, "
                  f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
            pd.DataFrame(rows).to_csv(sys.argv[3], index=False)
    print("done", flush=True)
