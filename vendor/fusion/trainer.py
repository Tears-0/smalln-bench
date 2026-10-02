"""Fold-level training loop for the proposed model.

Faithful to Transferlearning(dl).ipynb: RMSprop(weight_decay=1e-4), batch
size 1 with per-epoch shuffling, AMP autocast + GradScaler, metrics read off
the final epoch. The only removed element is the unused ESM token tensor the
original Dataset returned, which never touched the RNG stream.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .model import MultiModalProteomicsModel


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)


def make_optimizer(name: str, params, lr: float):
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=1e-4)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    raise ValueError(f"unknown optimizer {name!r}")


class _AbundanceDataset(Dataset):
    """One item per sample: ([P, 1] abundances, class label)."""

    def __init__(self, df, sample_cols, labels):
        self.x = [
            torch.tensor(df[c].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(-1)
            for c in sample_cols
        ]
        self.y = list(labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], torch.tensor(self.y[i])


@dataclass
class FoldConfig:
    use_esm: bool = True
    pooling: str = "attention"
    epochs: int = C.EPOCHS
    lr: float = C.LR
    optimizer: str = C.OPTIMIZER
    seed: int = C.SEED
    device: str = "cuda"
    record_history: bool = True
    checkpoints: tuple = ()


@dataclass
class FoldResult:
    y_true: list = field(default_factory=list)
    y_pred: list = field(default_factory=list)
    # softmax over the final-epoch logits, for AUROC. Recorded under no_grad
    # and consuming no RNG, so it cannot change any previously reported number.
    y_prob: list = field(default_factory=list)
    train_loss: np.ndarray = None
    val_loss: np.ndarray = None
    n_proteins: int = 0
    checkpoint_metrics: dict = field(default_factory=dict)
    model: object = None


def train_fold(train_df, val_df, train_samples, val_samples, train_labels, val_labels,
               embeddings, cfg: FoldConfig, keep_model: bool = False) -> FoldResult:
    """Train one fold and return held-out predictions at the final epoch.

    embeddings is the full [N, 480] ESM-2 tensor, or None for the
    abundance-only ablation. Rows are selected via 'Original_Index'.
    """
    device = cfg.device

    # Model is constructed before any data-dependent RNG use, matching the
    # notebook's ordering so a fixed seed reproduces the original init.
    model = MultiModalProteomicsModel(
        use_esm=cfg.use_esm, pooling=cfg.pooling, embedding_dim=C.EMBED_DIM
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    scaler = GradScaler(device)

    if cfg.use_esm:
        tr_idx = torch.tensor(train_df["Original_Index"].to_numpy(), dtype=torch.long, device=device)
        va_idx = torch.tensor(val_df["Original_Index"].to_numpy(), dtype=torch.long, device=device)
        train_embs = embeddings[tr_idx]
        val_embs = embeddings[va_idx]
    else:
        train_embs = val_embs = None

    train_ds = _AbundanceDataset(train_df, train_samples, train_labels)
    val_ds = _AbundanceDataset(val_df, val_samples, val_labels)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    n_epochs = cfg.epochs
    hist_train = np.zeros(n_epochs) if cfg.record_history else None
    hist_val = np.zeros(n_epochs) if cfg.record_history else None
    checkpoint_metrics = {}

    y_true, y_pred = [], []
    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for abundances, label in train_loader:
            optimizer.zero_grad()
            abundances = abundances.to(device)
            abundances = abundances.squeeze(0) if abundances.dim() == 3 else abundances
            with autocast(device):
                output = model(train_embs, abundances)
                loss = criterion(output, label.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        if cfg.record_history:
            hist_train[epoch] = running / len(train_loader)

        model.eval()
        running_val = 0.0
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for abundances, label in val_loader:
                abundances = abundances.to(device)
                abundances = abundances.squeeze(0) if abundances.dim() == 3 else abundances
                output = model(val_embs, abundances)
                running_val += criterion(output, label.to(device)).item()
                y_true.append(label.item())
                y_pred.append(output.argmax(dim=1).cpu().item())
                y_prob.append(torch.softmax(output.float(), dim=1)
                              .squeeze(0).cpu().tolist())
        if cfg.record_history:
            hist_val[epoch] = running_val / len(val_loader)

        if (epoch + 1) in cfg.checkpoints:
            checkpoint_metrics[epoch + 1] = (list(y_true), list(y_pred))

    return FoldResult(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        train_loss=hist_train,
        val_loss=hist_val,
        n_proteins=len(train_df),
        checkpoint_metrics=checkpoint_metrics,
        model=model if keep_model else None,
    )
