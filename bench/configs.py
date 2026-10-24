"""The 15 configurations (paper §3.4, Table 2).

smote: None | 'infold' | 'leaky' (SMOTE on all of S_r before the internal split)
leaky_filter: whether the leaky-internal variant (filter fitted on all of S_r)
              is computed for this config (supervised filter present; not FUS, for cost)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    id: int
    name: str
    model: str
    variant: str
    runner: str               # 'cpu' | 'tabpfn' | 'fus'
    smote: str | None = None
    leaky_filter: bool = True
    lpo: bool = True
    shares_external_with: int | None = None

    @property
    def protein_ranking(self):
        return self.model != "tabpfn" and self.variant != "path"


CONFIGS = [
    Config(1, "CCP", "ccp", "ref", "cpu"),
    Config(2, "EN", "en", "ref", "cpu"),
    Config(3, "RF", "rf", "ref", "cpu"),
    Config(4, "TabPFN", "tabpfn", "ref", "tabpfn"),
    Config(5, "FUS", "fus", "ref", "fus", leaky_filter=False, lpo=False),
    Config(6, "EN-var", "en", "var", "cpu", leaky_filter=False),
    Config(7, "RF-var", "rf", "var", "cpu", leaky_filter=False),
    Config(8, "EN-drop", "en", "drop", "cpu"),
    Config(9, "RF-drop", "rf", "drop", "cpu"),
    Config(10, "EN-path", "en", "path", "cpu"),
    Config(11, "RF-path", "rf", "path", "cpu"),
    Config(15, "EN-smote", "en", "ref", "cpu", smote="infold"),
    Config(16, "RF-smote", "rf", "ref", "cpu", smote="infold"),
    Config(17, "EN-smote-leaky", "en", "ref", "cpu", smote="leaky", leaky_filter=False, shares_external_with=15),
    Config(18, "RF-smote-leaky", "rf", "ref", "cpu", smote="leaky", leaky_filter=False, shares_external_with=16),
]
BY_ID = {c.id: c for c in CONFIGS}
