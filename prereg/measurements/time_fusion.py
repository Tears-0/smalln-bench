"""Measure the per-optimizer-step cost of the frozen fusion arm.

Replicates trainer.py's inner loop exactly (RMSprop wd=1e-4, batch size 1,
AMP autocast + GradScaler, deterministic algorithms, loss.item() sync per step)
on synthetic data, for several protein counts P. Nothing is written to the repo.
"""
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, r"C:\Users\X\OneDrive\Projects\CSH\src")

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from rebuttal.model import MultiModalProteomicsModel

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
dev = "cuda"


def time_steps(P, n_classes=2, steps=600, warm=100):
    torch.manual_seed(0)
    model = MultiModalProteomicsModel(num_classes=n_classes).to(dev)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler(dev)
    crit = nn.CrossEntropyLoss()
    embs = torch.randn(P, 480, device=dev)
    xs = [torch.randn(P, 1, device=dev) for _ in range(8)]
    ys = [torch.tensor([i % n_classes], device=dev) for i in range(8)]
    for i in range(warm + steps):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        opt.zero_grad()
        with autocast(dev):
            out = model(embs, xs[i % 8])
            loss = crit(out, ys[i % 8])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss.item()  # trainer.py syncs every step via running += loss.item()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3


print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
print("CPU logical cores:", os.cpu_count())
for P in (500, 2000, 5000, 10000):
    print("P=%5d  %.2f ms / optimizer step" % (P, time_steps(P)))
