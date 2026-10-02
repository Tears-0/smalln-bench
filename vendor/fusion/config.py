"""Shared configuration for the BHI 2026 rebuttal experiments.

Import this module before torch so CUBLAS_WORKSPACE_CONFIG is set prior to
CUDA context creation (required by torch.use_deterministic_algorithms).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "rebuttal"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Raw (un-imputed) log2 abundance matrix used by the proposed model.
RAW_MATRIX = DATA_DIR / "not_imputed.csv"
# Row-aligned frozen ESM-2 embeddings for RAW_MATRIX (2841 x 480).
EMBEDDINGS = SRC_DIR / "static_protein_embs(NI).pt"

SAMPLE_COLS = ["F61", "F62", "F63", "F64", "F65", "F66", "F67", "F68", "F69"]
GROUPS = {
    "Control": ["F61", "F62", "F63"],
    "8H": ["F64", "F65", "F66"],
    "24H": ["F67", "F68", "F69"],
}
CLASS_NAMES = ["Control", "8H", "24H"]
LABEL_TO_IDX = {"Control": 0, "8H": 1, "24H": 2}
SAMPLE_TO_LABEL = {s: g for g, cols in GROUPS.items() for s in cols}
LABELS = [LABEL_TO_IDX[SAMPLE_TO_LABEL[s]] for s in SAMPLE_COLS]

# Training hyperparameters as used for the submitted results.
EPOCHS = 1000
LR = 1e-4
OPTIMIZER = "rmsprop"
SEED = 42
EMBED_DIM = 480

# SFI-Hybrid parameters as printed in the paper (Eqs. 3-4).
SFI_DOWNSHIFT = 1.5
SFI_WIDTH = 0.3
KNN_NEIGHBORS = 3
