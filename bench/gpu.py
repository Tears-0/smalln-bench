"""GPU memory guard for every process that uses CUDA.

On Windows the NVIDIA driver's "sysmem fallback" lets a process that exceeds
dedicated VRAM spill into shared system RAM instead of failing; under memory
pressure that can stall or crash the whole machine (it did on 2026-10-13).
Capping PyTorch's caching allocator makes an oversized allocation raise
torch.cuda.OutOfMemoryError inside this process instead, which callers catch.

Budgets (16 GB card; ~3 GB is taken by the desktop):
    embeddings  5 GB    fusion  4 GB    tabpfn  4 GB
"""
from __future__ import annotations

import torch

BUDGET_GB = {"embeddings": 5.0, "fusion": 4.0, "tabpfn": 4.0}


def guard(role: str) -> float:
    """Cap this process's PyTorch allocations; returns the cap in GB."""
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    cap = BUDGET_GB[role]
    torch.cuda.set_per_process_memory_fraction(min(cap / total, 0.9), 0)
    return cap


def used_gb() -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
