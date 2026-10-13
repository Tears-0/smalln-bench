"""Aggregate throughput of K concurrent fusion-training processes on one GPU.

Each worker runs the exact trainer.py step (RMSprop, batch 1, AMP, GradScaler,
deterministic, loss.item() sync) for a fixed wall-clock window and reports
steps done. Usage: python conc_fusion.py K
"""
import os
import subprocess
import sys
import time

WORKER = r'''
import os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, r"C:\Users\X\OneDrive\Projects\CSH\src")
import torch, torch.nn as nn
from torch.amp import GradScaler, autocast
from rebuttal.model import MultiModalProteomicsModel
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
dev = "cuda"; P = 2000
torch.manual_seed(0)
m = MultiModalProteomicsModel(num_classes=2).to(dev)
opt = torch.optim.RMSprop(m.parameters(), lr=1e-4, weight_decay=1e-4)
sc = GradScaler(dev); crit = nn.CrossEntropyLoss()
E = torch.randn(P, 480, device=dev)
xs = [torch.randn(P, 1, device=dev) for _ in range(8)]
ys = [torch.tensor([i % 2], device=dev) for i in range(8)]
def step(i):
    opt.zero_grad()
    with autocast(dev):
        loss = crit(m(E, xs[i % 8]), ys[i % 8])
    sc.scale(loss).backward(); sc.step(opt); sc.update(); loss.item()
for i in range(150): step(i)
torch.cuda.synchronize()
start = float(sys.argv[1]); window = float(sys.argv[2])
while time.time() < start: time.sleep(0.001)
n = 0
while time.time() < start + window:
    step(n); n += 1
print(n)
'''

K = int(sys.argv[1])
window = 20.0
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_conc_worker.py")
open(path, "w").write(WORKER)
start = time.time() + 25.0  # time for all workers to init CUDA and warm up
procs = [subprocess.Popen([sys.executable, path, str(start), str(window)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
         for _ in range(K)]
steps = []
for p in procs:
    out, err = p.communicate()
    try:
        steps.append(int(out.strip().splitlines()[-1]))
    except Exception:
        print("worker failed:", err[-500:])
tot = sum(steps)
print("K=%d  per-worker ms/step: %s  aggregate %.0f steps/s  (%.2f ms/step effective)"
      % (K, [round(window * 1e3 / s, 2) for s in steps], tot / window, window * 1e3 / tot))
