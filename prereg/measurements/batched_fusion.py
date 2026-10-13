"""Prototype: train M independent copies of the frozen fusion model in lockstep.

Every parameter gets a leading model axis [M, ...]; linear layers become
batched matmuls. Each model sees its own sample at each step (batch size 1 per
model), its own loss term, and RMSprop is elementwise, so model m's update is
exactly what it would get trained alone. Proteins absent from a fold are
handled by a mask (attention logit -inf), so folds with different protein sets
can share one batch.

Part 1 checks equivalence against rebuttal.model.MultiModalProteomicsModel
(fp32, dropout off, identical init and sample order). Part 2 times one step.
"""
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, r"C:\Users\X\OneDrive\Projects\CSH\src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from rebuttal.model import MultiModalProteomicsModel

dev = "cuda"


class BatchedFusion(nn.Module):
    """M stacked copies of MultiModalProteomicsModel (attention pooling, ESM on)."""

    def __init__(self, refs, p_drop=0.5):
        super().__init__()
        st = lambda get: nn.Parameter(torch.stack([get(r).detach().clone() for r in refs]))
        self.W1 = st(lambda r: r.abundance_encoder[0].weight)   # [M,128,1]
        self.b1 = st(lambda r: r.abundance_encoder[0].bias)     # [M,128]
        self.W2 = st(lambda r: r.abundance_encoder[2].weight)   # [M,480,128]
        self.b2 = st(lambda r: r.abundance_encoder[2].bias)
        self.W3 = st(lambda r: r.attention_scorer[0].weight)    # [M,128,480]
        self.b3 = st(lambda r: r.attention_scorer[0].bias)
        self.W4 = st(lambda r: r.attention_scorer[2].weight)    # [M,1,128]
        self.b4 = st(lambda r: r.attention_scorer[2].bias)
        self.W5 = st(lambda r: r.classifier[0].weight)          # [M,256,480]
        self.b5 = st(lambda r: r.classifier[0].bias)
        self.W6 = st(lambda r: r.classifier[3].weight)          # [M,C,256]
        self.b6 = st(lambda r: r.classifier[3].bias)
        self.p_drop = p_drop

    @staticmethod
    def _lin(x, W, b):  # x [M,P,in], W [M,out,in], b [M,out]
        return torch.baddbmm(b.unsqueeze(1), x, W.transpose(1, 2))

    def forward(self, E, x, mask=None):
        """E [P,480] (shared) or [M,P,480]; x [M,P,1]; mask [M,P] bool (True = keep)."""
        z = self._lin(F.relu(self._lin(x, self.W1, self.b1)), self.W2, self.b2)
        fused = E + torch.sigmoid(z)
        s = self._lin(torch.tanh(self._lin(fused, self.W3, self.b3)), self.W4, self.b4)
        if mask is not None:
            s = s.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        a = torch.softmax(s, dim=1)                              # over proteins
        v = (fused * a).sum(dim=1, keepdim=True)                 # [M,1,480]
        h = F.dropout(F.relu(self._lin(v, self.W5, self.b5)), self.p_drop, self.training)
        return self._lin(h, self.W6, self.b6).squeeze(1)         # [M,C]


def refs_for(M, seed0=100):
    out = []
    for m in range(M):
        torch.manual_seed(seed0 + m)
        out.append(MultiModalProteomicsModel(num_classes=2).to(dev))
    return out


# ---------------- Part 1: equivalence (fp32, no dropout) -------------------
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
M, P, n, steps = 4, 300, 5, 200
g = torch.Generator(device="cpu").manual_seed(7)
E = torch.randn(P, 480, generator=g).to(dev)
X = torch.randn(M, n, P, 1, generator=g).to(dev)      # per-model training sets
Y = torch.randint(0, 2, (M, n), generator=g).to(dev)
mask = torch.ones(M, P, dtype=torch.bool, device=dev)
mask[1, 250:] = False                                  # model 1 has fewer proteins
order = torch.randint(0, n, (M, steps), generator=g)

refs = refs_for(M)
bat = BatchedFusion(refs, p_drop=0.0).to(dev)
for r in refs:
    r.classifier[2].p = 0.0
opt_b = torch.optim.RMSprop(bat.parameters(), lr=1e-4, weight_decay=1e-4)
opts = [torch.optim.RMSprop(r.parameters(), lr=1e-4, weight_decay=1e-4) for r in refs]

for t in range(steps):
    idx = order[:, t]
    xb = X[torch.arange(M), idx]                        # [M,P,1]
    yb = Y[torch.arange(M), idx]
    opt_b.zero_grad()
    F.cross_entropy(bat(E, xb, mask), yb, reduction="sum").backward()
    opt_b.step()
    for m, (r, o) in enumerate(zip(refs, opts)):
        keep = mask[m]
        o.zero_grad()
        F.cross_entropy(r(E[keep], xb[m][keep]), yb[m:m + 1]).backward()
        o.step()

with torch.no_grad():
    xt = torch.randn(M, P, 1, device=dev)
    out_b = bat(E, xt, mask)
    out_r = torch.cat([r(E[mask[m]], xt[m][mask[m]]) for m, r in enumerate(refs)])
    dW = max((bat.W5[m] - r.classifier[0].weight).abs().max().item() for m, r in enumerate(refs))
print("Part 1  equivalence after %d steps: max|logit diff| = %.2e, max|W5 diff| = %.2e"
      % (steps, (out_b - out_r).abs().max().item(), dW))

# ---------------- Part 2: timing (AMP, dropout on, as frozen) --------------
torch.backends.cuda.matmul.allow_tf32 = True


def time_batched(M, P, steps=60, warm=15):
    refs = refs_for(M)
    bat = BatchedFusion(refs).to(dev)
    del refs
    opt = torch.optim.RMSprop(bat.parameters(), lr=1e-4, weight_decay=1e-4)
    sc = GradScaler(dev)
    E = torch.randn(P, 480, device=dev)
    xs = [torch.randn(M, P, 1, device=dev) for _ in range(4)]
    ys = [torch.randint(0, 2, (M,), device=dev) for _ in range(4)]
    for i in range(warm + steps):
        if i == warm:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad()
        with autocast(dev):
            loss = F.cross_entropy(bat(E, xs[i % 4]), ys[i % 4], reduction="sum")
        sc.scale(loss).backward(); sc.step(opt); sc.update(); loss.item()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / steps * 1e3
    mem = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    return ms, mem


print("Part 2  (sequential baseline measured earlier: 3.25-3.5 ms per model-step)")
for P in (2000, 10000):
    for M in (1, 25, 100, 200):
        try:
            ms, mem = time_batched(M, P)
            print("  P=%5d M=%3d  %7.2f ms/step  -> %.3f ms per model-step  (%.1fx)  peak %.1f GB"
                  % (P, M, ms, ms / M, 3.3 / (ms / M), mem))
        except torch.cuda.OutOfMemoryError:
            print("  P=%5d M=%3d  out of memory" % (P, M)); torch.cuda.empty_cache()
