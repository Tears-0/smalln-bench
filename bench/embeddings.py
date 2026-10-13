"""ESM-2 CLS embeddings per task for the fusion arm (spec §5.1 FUS, F9).

Model esm2_t12_35M_UR50D (fair-esm 2.0.0), readout representations[12][:, 0, :]
-- the CLS/BOS token, as the case study's embeddings were produced. Sequences
longer than MAX_LEN residues are truncated to their first MAX_LEN (counts logged).
Embeddings are cached once per unique sequence (spec §5.3).

Memory: self-attention holds a [batch, 20 heads, L, L] tensor per layer, so the
batch is bounded by batch * L^2 (ATTN_BUDGET), not by token count. With the
process capped by bench.gpu.guard, an oversized batch raises OutOfMemoryError
and is split; a single sequence that still does not fit runs on the CPU.
The cache is written every SAVE_EVERY sequences so an interruption loses little.

    python -m bench.embeddings
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from . import gpu, loaders
from .paths import CASE_STUDY, DERIVED

MAX_LEN = 4000
ATTN_BUDGET = 8_000_000      # batch * (L+2)^2; 20 heads x fp32 -> ~640 MB per attention tensor
MAX_BATCH = 64
SAVE_EVERY = 1000
CACHE = DERIVED / "esm2_cls_cache.npz"
_ESM_OK = set("LAGVSERTIDPKQNFYMHWCXBUZO")


def _model():
    import esm
    assert getattr(esm, "__version__", "") == "2.0.0" and hasattr(esm, "pretrained"), \
        "import esm must resolve to fair-esm 2.0.0 (see env/README.md)"
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    return model.eval(), alphabet.get_batch_converter()


def _key(seq):
    return hashlib.sha1(seq.encode()).hexdigest()


def _clean(seq):
    """Truncate, and map characters ESM-2 cannot tokenise (e.g. Ensembl '*') to X."""
    return "".join(c if c in _ESM_OK else "X" for c in seq[:MAX_LEN])


def _batches(lengths, order):
    i = 0
    while i < len(order):
        j = i + 1
        while (j < len(order) and j - i < MAX_BATCH
               and (j - i + 1) * (min(lengths[order[j]], MAX_LEN) + 2) ** 2 <= ATTN_BUDGET):
            j += 1
        yield order[i:j]
        i = j


@torch.no_grad()
def _forward(model, conv, seqs, device):
    _, _, toks = conv([(str(k), s) for k, s in enumerate(seqs)])
    rep = model.to(device)(toks.to(device), repr_layers=[12])["representations"][12][:, 0, :]
    return rep.float().cpu().numpy()


def _embed_batch(model, conv, seqs):
    try:
        return _forward(model, conv, seqs, "cuda")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(seqs) > 1:
            h = len(seqs) // 2
            return np.vstack([_embed_batch(model, conv, seqs[:h]), _embed_batch(model, conv, seqs[h:])])
        out = _forward(model, conv, seqs, "cpu")   # one very long sequence
        model.to("cuda")
        return out


def embed(seqs, model=None, conv=None, on_progress=None):
    """CLS embeddings [len(seqs), 480] float32, in the order given."""
    if model is None:
        model, conv = _model()
    clean = [_clean(s) for s in seqs]
    lengths = np.array([len(s) for s in clean])
    order = np.argsort(lengths, kind="stable")
    out = np.zeros((len(seqs), 480), dtype=np.float32)
    done = 0
    for b in _batches(lengths, order):
        out[b] = _embed_batch(model, conv, [clean[k] for k in b])
        done += len(b)
        if on_progress:
            on_progress(done, b, out)
    return out


def build_all():
    gpu.guard("embeddings")
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    tasks = {n: loaders.load(n, sequences=True) for n in loaders.TASKS}
    todo = sorted({s for t in tasks.values() for s in t.sequence if s and _key(s) not in cache})
    print(f"{len(cache)} cached, {len(todo)} to embed", flush=True)
    if todo:
        model, conv = _model()
        state = {"last": 0}

        def progress(done, b, out):
            for k in b:
                cache[_key(todo[k])] = out[k]
            if done - state["last"] >= SAVE_EVERY or done == len(todo):
                np.savez(CACHE, **cache)
                state["last"] = done
                print(f"  {done}/{len(todo)} embedded, peak GPU {gpu.used_gb():.2f} GB", flush=True)

        embed(todo, model, conv, on_progress=progress)
    log = {}
    for n, t in tasks.items():
        E = np.full((t.p, 480), np.nan, dtype=np.float32)
        for i, s in enumerate(t.sequence):
            if s:
                E[i] = cache[_key(s)]
        np.save(DERIVED / f"esm2_cls_{n}.npy", E)
        log[n] = dict(proteins=t.p, embedded=int(np.isfinite(E[:, 0]).sum()),
                      truncated=int(sum(len(s) > MAX_LEN for s in t.sequence)))
    (DERIVED / "esm2_cls_log.json").write_text(json.dumps(dict(log, max_len=MAX_LEN), indent=2))
    return log


def load(name):
    return np.load(DERIVED / f"esm2_cls_{name}.npy")


def check_case_study(n=64):
    """Recompute CLS embeddings for case-study sequences (all <= MAX_LEN); compare with the stored file."""
    import pandas as pd
    gpu.guard("embeddings")
    df = pd.read_csv(CASE_STUDY / "not_imputed.csv")
    ref = torch.load(CASE_STUDY / "esm2_t12_35M_cls_embeddings.pt", weights_only=True).float().numpy()
    ok = np.flatnonzero(df["Sequence"].str.len().to_numpy() <= MAX_LEN)
    idx = np.random.default_rng(0).choice(ok, n, replace=False)
    E = embed([df["Sequence"].iloc[i] for i in idx])
    err = np.abs(E - ref[idx]).max()
    cos = np.mean([np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b) for a, b in zip(E, ref[idx])])
    return float(err), float(cos)


if __name__ == "__main__":
    err, cos = check_case_study()
    print(f"case-study check: max |diff| = {err:.2e}, mean cosine = {cos:.6f}, peak GPU {gpu.used_gb():.2f} GB")
    print(build_all())
