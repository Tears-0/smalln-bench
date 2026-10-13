"""Frozen fusion arm (config 5), trained by a batched lockstep trainer (F26, F27, F29).

Each fit is the case-study MultiModalProteomicsModel (vendor/fusion/model.py,
binary head): RMSprop lr 1e-4, weight decay 1e-4, 1,000 epochs, batch size 1
per model, dropout 0.5, fp32. Many fits with the same training-set size are
stacked into one batched model; each model sees its own sample at each step,
its own loss term, and RMSprop is elementwise, so every model's update is the
update it would get alone (prereg/measurements: logits agree to 7e-7).

Input: the reference pipeline's in-fold p = 500 proteins, restricted to proteins
with a sequence (F9), z-scored. No LPO and no leaky pass (F29).
Protein ranking: mean attention weight over the training samples.

    python -m bench.fusion --out main
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cv, embeddings, gpu, metrics
from .paths import RUNS, VENDOR_FUSION
from .preprocess import prepare
from .run import ARMS, _sign_S, code_hash, context, plan
from .subsample import GRID, draw, permuted_labels, seed

EPOCHS, LR, WD, P_DROP = 1000, 1e-4, 1e-4, 0.5
MAX_MODELS = 200   # ~1.6 GB at P = 500 in fp32 (measured 3.1 GB at 400)
CONFIG_ID, CONFIG_NAME = 5, "FUS"


def _vendored_model():
    spec = importlib.util.spec_from_file_location("frozen_model", VENDOR_FUSION / "model.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.MultiModalProteomicsModel


class Batched(nn.Module):
    """M stacked copies of MultiModalProteomicsModel (attention pooling, ESM on)."""

    def __init__(self, refs):
        super().__init__()
        st = lambda g: nn.Parameter(torch.stack([g(r).detach().clone() for r in refs]))
        self.W1 = st(lambda r: r.abundance_encoder[0].weight); self.b1 = st(lambda r: r.abundance_encoder[0].bias)
        self.W2 = st(lambda r: r.abundance_encoder[2].weight); self.b2 = st(lambda r: r.abundance_encoder[2].bias)
        self.W3 = st(lambda r: r.attention_scorer[0].weight); self.b3 = st(lambda r: r.attention_scorer[0].bias)
        self.W4 = st(lambda r: r.attention_scorer[2].weight); self.b4 = st(lambda r: r.attention_scorer[2].bias)
        self.W5 = st(lambda r: r.classifier[0].weight); self.b5 = st(lambda r: r.classifier[0].bias)
        self.W6 = st(lambda r: r.classifier[3].weight); self.b6 = st(lambda r: r.classifier[3].bias)

    @staticmethod
    def _lin(x, W, b):
        return torch.baddbmm(b.unsqueeze(1), x, W.transpose(1, 2))

    def fused(self, E, x):
        z = self._lin(F.relu(self._lin(x, self.W1, self.b1)), self.W2, self.b2)
        return E + torch.sigmoid(z)

    def attention(self, f):
        s = self._lin(torch.tanh(self._lin(f, self.W3, self.b3)), self.W4, self.b4)
        return torch.softmax(s, dim=1)                                   # [M, P, 1]

    def forward(self, E, x):
        f = self.fused(E, x)
        a = self.attention(f)
        v = (f * a).sum(dim=1, keepdim=True)
        h = F.dropout(F.relu(self._lin(v, self.W5, self.b5)), P_DROP, self.training)
        return self._lin(h, self.W6, self.b6).squeeze(1)                 # [M, 2]


def train_batch(fits, E_task, dev="cuda"):
    """fits: list of dicts with Ztr [n, P], ytr [n], Zte [m_i, P], feat [P], seed.
    All fits share n. Returns per-fit (test probability of class 1, attention [P])."""
    Model = _vendored_model()
    refs = []
    for f in fits:
        torch.manual_seed(f["seed"] % (2**63))
        refs.append(Model(num_classes=2, embedding_dim=480))
    net = Batched(refs).to(dev); del refs
    opt = torch.optim.RMSprop(net.parameters(), lr=LR, weight_decay=WD)
    M, n = len(fits), fits[0]["Ztr"].shape[0]
    Xtr = torch.tensor(np.stack([f["Ztr"] for f in fits]), dtype=torch.float32, device=dev).unsqueeze(-1)
    ytr = torch.tensor(np.stack([f["ytr"] for f in fits]), dtype=torch.long, device=dev)
    E = torch.tensor(np.stack([E_task[f["feat"]] for f in fits]), dtype=torch.float32, device=dev)
    ar = torch.arange(M, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(fits[0]["seed"] % (2**63))
    torch.manual_seed(fits[0]["seed"] % (2**63) + 1)  # dropout stream
    net.train()
    for _ in range(EPOCHS):
        order = torch.argsort(torch.rand(M, n, generator=g, device=dev), dim=1)
        for step in range(n):
            idx = order[:, step]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(E, Xtr[ar, idx]), ytr[ar, idx], reduction="sum")
            loss.backward()
            opt.step()
    net.eval()
    out = []
    with torch.no_grad():
        att = torch.zeros(M, E.shape[1], device=dev)
        for j in range(n):
            att += net.attention(net.fused(E, Xtr[:, j]))[..., 0]
        att = (att / n).cpu().numpy()
        m_max = max(f["Zte"].shape[0] for f in fits)
        Zte = np.zeros((M, m_max, E.shape[1]), dtype=np.float32)
        for k, f in enumerate(fits):
            Zte[k, : f["Zte"].shape[0]] = f["Zte"]
        Zte = torch.tensor(Zte, device=dev).unsqueeze(-1)
        probs = np.zeros((M, m_max), dtype=np.float32)
        for j in range(m_max):
            probs[:, j] = torch.softmax(net(E, Zte[:, j]), dim=1)[:, 1].cpu().numpy()
    for k, f in enumerate(fits):
        out.append((probs[k, : f["Zte"].shape[0]], att[k]))
    return out


def _unit_fits(task_name, N, r, arm):
    t, ctx, ref = context(task_name)
    S, H = draw(t, N, r)
    yS = t.y[S].astype(int)
    if arm == "perm":
        yS = permuted_labels(yS, task_name, N, r)
    coords = (task_name, N, r, arm)
    prim = cv.primary_folds(yS, N, coords)
    fits = []
    for i, (tr_pos, te_pos) in enumerate(prim):
        p = prepare(ctx, S[tr_pos], S[te_pos], yS[tr_pos], "ref", seed("impute", *coords, "prim", i),
                    restrict_seq=True, tag="prim:honest")
        fits.append(dict(kind="prim", i=i, Ztr=p.Ztr.astype(np.float32), ytr=yS[tr_pos], Zte=p.Zte.astype(np.float32),
                         feat=p.feat, seed=seed("model", CONFIG_ID, *coords, "prim", i)))
    p = prepare(ctx, S, H, yS, "ref", seed("impute", *coords, "ext", 0), restrict_seq=True, tag="ext:honest")
    fits.append(dict(kind="ext", i=0, Ztr=p.Ztr.astype(np.float32), ytr=yS, Zte=p.Zte.astype(np.float32),
                     feat=p.feat, seed=seed("model", CONFIG_ID, *coords, "ext", 0), prep=p))
    meta = dict(S=S, H=H, yS=yS, yH=t.y[H].astype(int), prim=prim, sign_S=_sign_S(t.X, S, yS))
    return fits, meta


def run_cell(task_name, N, units, out_file):
    """Train every FUS fit of one (task, N) cell and write its rows."""
    t, ctx, ref = context(task_name)
    E_task = np.nan_to_num(embeddings.load(task_name))
    per_unit, groups = {}, {}
    for u in units:
        fits, meta = _unit_fits(task_name, N, *u)
        per_unit[u] = (fits, meta)
        for f in fits:
            groups.setdefault(f["Ztr"].shape[0], []).append((u, f))
    results = {}
    for n_train, items in sorted(groups.items()):
        for b in range(0, len(items), MAX_MODELS):
            chunk = items[b:b + MAX_MODELS]
            t0 = time.time()
            res = train_batch([f for _, f in chunk], E_task)
            for (u, f), rr in zip(chunk, res):
                results[(u, f["kind"], f["i"])] = rr
            print(f"   [{time.strftime('%H:%M:%S')}] {task_name} N={N} n_train={n_train} "
                  f"models {b + 1}-{b + len(chunk)}/{len(items)} in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for u, (fits, meta) in per_unit.items():
        r, arm = u
        prim_scores = [results[(u, "prim", i)][0] for i in range(len(meta["prim"]))]
        auc_int = cv.primary_auc(meta["yS"], N, meta["prim"], prim_scores)
        s_ext, att = results[(u, "ext", 0)]
        p_ext = fits[-1]["prep"]
        row = dict(task=task_name, N=N, r=r, arm=arm, config_id=CONFIG_ID, config=CONFIG_NAME,
                   auc_int=auc_int, auc_int_lpo=np.nan, auc_int_leaky=np.nan,
                   auc_ext=cv.external_auc(meta["yH"], s_ext), n_S_pos=int((meta["yS"] == 1).sum()),
                   n_H_pos=int((meta["yH"] == 1).sum()), n_H_neg=int((meta["yH"] == 0).sum()),
                   n_feat=int(p_ext.Ztr.shape[1]), primary=cv.primary_kind(N), seconds=np.nan)
        rank = metrics.full_ranking(p_ext, att)
        row.update(metrics.recovery(rank, ref, meta["sign_S"]))
        row["top100"] = rank[:100].astype(np.int32).tolist()
        rows.append(row)
    pd.DataFrame(rows).assign(code=code_hash()).to_parquet(out_file)


def run(out, tasks, R, Ns=GRID, arms=ARMS, r0=0, block=None):
    print(f"GPU cap {gpu.guard('fusion'):.1f} GB", flush=True)
    cells = plan(tasks, R, Ns)
    d = RUNS / out / "fus"; d.mkdir(parents=True, exist_ok=True)
    (RUNS / out / "plan_fus.json").write_text(json.dumps(dict(cells=cells, R=R, r0=r0, arms=list(arms),
                                                               code=code_hash(), started=time.strftime("%F %T")), indent=2))
    block = block or R
    for name, N in cells:
        for arm in arms:
            for b in range(r0, r0 + R, block):
                f = d / f"{name}_N{N}_{arm}_r{b:03d}.parquet"
                if f.exists():
                    continue
                units = [(r, arm) for r in range(b, min(b + block, r0 + R))]
                print(f"[{time.strftime('%H:%M:%S')}] cell {name} N={N} {arm} r{b}-{min(b + block, r0 + R) - 1}: "
                      f"{len(units)} units", flush=True)
                run_cell(name, N, units, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="main")
    ap.add_argument("--tasks", default="B1,B2,B5")
    ap.add_argument("--N", default=",".join(map(str, GRID)))
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--r0", type=int, default=0)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--arms", default="true,perm")
    a = ap.parse_args()
    run(a.out, a.tasks.split(","), a.R, tuple(int(x) for x in a.N.split(",")), tuple(a.arms.split(",")),
        a.r0, a.block or None)
