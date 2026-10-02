"""The accession-to-gene mapping used in Transferlearning(dl).ipynb, cell 21.

Reproduced verbatim so that panels can be mapped the way the original analysis
mapped them. It differs from the plain GN= extraction in bio_enrichment in two
ways that materially change the resulting gene list:

  * when no GN= field is present it falls back to the first or second
    whitespace-delimited token of the description, provided that token has no
    lowercase letter. This admits symbols that a GN= extraction drops.
  * a hand-curated `non_human` blocklist is then subtracted.

Both behaviours are kept exactly as written, including the `accession in
df_fix['query']` test, which checks membership against the Series *index*
rather than its values and is therefore almost always False. Reproducing the
original mapping means reproducing that too.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config as C

NON_HUMAN = {
    'RPL23AL', 'Q53GE9', 'FLJ54507', 'Q59ES8', 'Q9BSM5', 'FLJ55325', 'B3KTA3',
    'FLJ51018', 'Q53FF5', 'ACT', 'H2B', 'Q53FV0', 'Q56G89', 'Q53FB0',
    'FLJ52243', 'Q7Z612', 'B4DHB3', 'FLJ78096', 'FLJ59371', 'Q9UEL2',
    'FLJ56548', 'FLJ50720', 'FLJ32530', 'FLJ52593', 'Q5U071', 'Q96HI1',
    'FLJ38578', 'B7Z3K7', 'FLJ53558', 'FLJ56708', 'A0A384NL58', 'FLJ78930',
    'A0A1U9X8Y4', 'Q5S4N1', 'FLJ78504', 'FLJ57133', 'FLJ96117', 'FLJ53677',
    'Q59GE4', 'A1XP52', 'Q16466', 'B2R5U1', 'Q59GP5', 'Q59EJ3', 'A8K5M4',
    'FLJ57278', 'A5GZ75', 'FLJ75113', 'Q1AHP8', 'FLJ54407', 'Q96GE1',
    'FLJ45706', 'FLJ59343', 'B4DRT3', 'FLJ54032', 'Q15204', 'B2R6L0', 'Q53GA7',
}

_SYNGO = None


def _syngo():
    global _SYNGO
    if _SYNGO is None:
        _SYNGO = pd.read_excel(C.DATA_DIR / "syngo_id_convert_2026-03-18_00-31.xlsx")
    return _SYNGO


def clean_gene(row) -> str:
    desc = row["Description"]
    accession = row["Accession"]
    df_fix = _syngo()

    match = re.search(r"GN=([a-zA-Z0-9\-_]+)", str(desc))
    if match:
        return match.group(1)

    if accession in df_fix["query"]:
        return df_fix[df_fix["Accession"] == accession].iloc[0]["symbol"]

    parts = str(desc).split(" ")
    fall_back1 = parts[0] if parts else ""
    fall_back2 = parts[1] if len(parts) > 1 else ""
    if fall_back1 and not any(c.islower() for c in fall_back1):
        return re.sub(r"[^a-zA-Z0-9 ]", "", fall_back1)
    if fall_back2 and not any(c.islower() for c in fall_back2) and fall_back2 != "H2B":
        return re.sub(r"[^a-zA-Z0-9 ]", "", fall_back2)
    return accession


def map_panel(accessions, df: pd.DataFrame) -> set:
    """Cell 21 end to end: map a panel to gene symbols, drop the blocklist."""
    sub = df[df["Accession"].isin(list(accessions))].copy()
    sub["Gene_Name"] = sub.apply(clean_gene, axis=1)
    return set(sub["Gene_Name"].tolist()) - NON_HUMAN
