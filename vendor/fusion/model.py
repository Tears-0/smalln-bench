"""The proposed multimodal model, with switches for each ablation cell.

Reference configuration (use_esm=True, pooling='attention') is numerically
identical to MultiModalProteomicsModel in Transferlearning(dl).ipynb: the
module order and shapes are unchanged, so a fixed seed reproduces the
submitted initialisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiModalProteomicsModel(nn.Module):
    """Abundance encoder -> gated fusion -> pooling -> classifier.

    use_esm=False   ablation 1A: fusion becomes f_i = sigmoid(z_i); the ESM-2
                    tensor is never read (forward rejects it, so the ablation
                    cannot silently keep using sequence information).
    pooling='mean'  ablation 2A.1: attention is replaced by a plain average
                    over proteins, and the attention scorer is not built.
    """

    def __init__(self, num_classes: int = 3, embedding_dim: int = 480,
                 use_esm: bool = True, pooling: str = "attention"):
        super().__init__()
        if pooling not in ("attention", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.use_esm = use_esm
        self.pooling = pooling

        self.abundance_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

        if pooling == "attention":
            self.attention_scorer = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, precomputed_embs, abundances):
        """precomputed_embs: [P, 480] or None when use_esm=False.
        abundances: [P, 1]."""
        if self.use_esm:
            if precomputed_embs is None:
                raise ValueError("use_esm=True but no ESM-2 embeddings were passed")
        elif precomputed_embs is not None:
            raise ValueError(
                "use_esm=False but ESM-2 embeddings were passed; the "
                "abundance-only ablation must not receive the sequence modality"
            )

        abund_embs = self.abundance_encoder(abundances)
        gate = torch.sigmoid(abund_embs)
        fused_embs = precomputed_embs + gate if self.use_esm else gate

        if self.pooling == "attention":
            attn_scores = self.attention_scorer(fused_embs)
            attn_weights = torch.softmax(attn_scores, dim=0)
            sample_vector = torch.sum(fused_embs * attn_weights, dim=0).unsqueeze(0)
        else:
            sample_vector = fused_embs.mean(dim=0).unsqueeze(0)

        return self.classifier(sample_vector)

    def attention_weights(self, precomputed_embs, abundances):
        """Per-protein attention distribution, for interpretability checks."""
        if self.pooling != "attention":
            raise ValueError("model was built with mean pooling")
        gate = torch.sigmoid(self.abundance_encoder(abundances))
        fused = precomputed_embs + gate if self.use_esm else gate
        return torch.softmax(self.attention_scorer(fused), dim=0)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
