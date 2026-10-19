"""Derived mutation tables that the B5 loader reads (data/derived/).

From the Cell Model Passports mutation release in data/raw/mutations_all_20260724.zip:

  kras_tp53_calls.csv            every KRAS and TP53 call, in file order (2,480 rows)
  models_with_mutation_data.csv  every model with at least one call, sorted (1,569 models);
                                 a line counts as TP53 wild-type only if it was sequenced

Run once before the first `loaders.load("B5")`:

    python -m bench.derive
"""
from __future__ import annotations

import zipfile

import pandas as pd

from .paths import DERIVED, RAW

MUTATIONS = RAW / "mutations_all_20260724.zip"
CALL_COLUMNS = ["gene_symbol", "model_id", "protein_mutation", "cancer_driver", "effect", "vaf", "coding", "source"]


def mutation_tables():
    with zipfile.ZipFile(MUTATIONS) as z:
        r = pd.read_csv(z.open("mutations_all_20260724.csv"), dtype=str, usecols=CALL_COLUMNS)
    calls = r[r.gene_symbol.isin(["KRAS", "TP53"])][CALL_COLUMNS].reset_index(drop=True)
    models = pd.DataFrame({"model_id": sorted(r.model_id.dropna().unique())})
    calls.to_csv(DERIVED / "kras_tp53_calls.csv", index=False)
    models.to_csv(DERIVED / "models_with_mutation_data.csv", index=False)
    return len(calls), len(models)


if __name__ == "__main__":
    n_calls, n_models = mutation_tables()
    print(f"{n_calls} KRAS/TP53 calls, {n_models} models with mutation data -> {DERIVED}")
