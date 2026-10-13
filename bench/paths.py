"""Repository locations. Everything under data/, models/ and results/ is local only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
REFERENCE = RAW / "reference"
DERIVED = DATA / "derived"
CASE_STUDY = DATA / "case_study"
MODELS = ROOT / "models"
TABPFN_CKPT = MODELS / "tabpfn" / "tabpfn-v3.5-fast-20260909.safetensors"
TABPFN_MD5 = "9340b9f6927ab257228158cbe446f01f"
VENDOR_FUSION = ROOT / "vendor" / "fusion"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"

for _d in (DERIVED, RUNS):
    _d.mkdir(parents=True, exist_ok=True)
