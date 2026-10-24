"""Seeded, balanced, unit-disjoint draws (paper §3.2-3.3).

S_r holds at most one sample per unit (patient / donor); H_r is every sample
whose unit is not in S_r. Seeds are derived from the run coordinates, so every
configuration sees identical draws.
"""
from __future__ import annotations

import hashlib

import numpy as np

GRID = (6, 9, 12, 20, 30, 50)
H_FLOOR = 20
R_MAIN = 100


def seed(*parts) -> int:
    """Stable 63-bit seed from run coordinates (independent of Python's hash salt)."""
    h = hashlib.sha256(repr(tuple(parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def class_sizes(N: int, r: int):
    """(n_pos, n_neg). Odd N: the larger class alternates with the parity of r."""
    if N % 2 == 0:
        return N // 2, N // 2
    big, small = N // 2 + 1, N // 2
    return (big, small) if r % 2 == 0 else (small, big)


def draw(task, N: int, r: int):
    """Return (S, H): sample indices into task, S ordered positives first."""
    rng = np.random.default_rng(seed("draw", task.name, N, r))
    n_pos, n_neg = class_sizes(N, r)
    used, S = set(), []
    for cls, k in ((1, n_pos), (0, n_neg)):
        cand = np.flatnonzero(task.y == cls)
        rng.shuffle(cand)
        got = 0
        for i in cand:
            u = task.unit[i]
            if u in used:
                continue
            S.append(i); used.add(u); got += 1
            if got == k:
                break
        if got < k:
            raise ValueError(f"{task.name}: cannot draw {k} unit-disjoint samples of class {cls}")
    S = np.array(S)
    H = np.flatnonzero(~np.isin(task.unit, list(used)))
    return S, H


def permuted_labels(y_S: np.ndarray, task_name: str, N: int, r: int):
    """Shuffle labels within S_r (balance preserved); H_r keeps true labels."""
    rng = np.random.default_rng(seed("perm", task_name, N, r))
    return rng.permutation(y_S)


def eligible(task, N: int, R: int) -> bool:
    """Every class keeps >= H_FLOOR samples in H_r, for every one of the R draws."""
    for r in range(R):
        try:
            S, H = draw(task, N, r)
        except ValueError:
            return False
        if min((task.y[H] == 1).sum(), (task.y[H] == 0).sum()) < H_FLOOR:
            return False
    return True
