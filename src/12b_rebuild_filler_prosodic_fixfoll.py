"""
12b_rebuild_filler_prosodic_fixfoll.py

The filler accent/break datasets from script 12 were built on the older filler file,
whose following-word features were not corrected (845 of the 2,625 break rows have a
different foll_lex_surp than dataset_filler_fixfoll_multillm.csv). This script rebuilds
dataset_filler_accent.csv and dataset_filler_break.csv from the corrected file, with the
same prosodic filter as before (a row is kept only when both the previous and the
following prosodic value exist). Same columns as the old files, so scripts 13, 13c and
13d work as they are.

The old files are copied to *_old.csv first (only if that copy does not exist yet).
Reparandum datasets are not touched.

run from the project folder:  python src/12b_rebuild_filler_prosodic_fixfoll.py
"""

import os
import shutil
import numpy as np
import pandas as pd

PROC = os.path.join("data", "processed")
FIX_CSV = os.path.join(PROC, "dataset_filler_fixfoll_multillm.csv")
PROS_CSV = os.path.join(PROC, "prosodic_features.csv")

VARIANTS = {
    "accent": ["prev_accent_strength", "foll_accent_strength"],
    "break": ["prev_break_index", "foll_break_index"],
}

fix = pd.read_csv(FIX_CSV)
pros = pd.read_csv(PROS_CSV)
print("corrected filler file rows:", len(fix), "| duplicated word_id:", fix.word_id.duplicated().sum())

for variant, cols in VARIANTS.items():
    old_path = os.path.join(PROC, f"dataset_filler_{variant}.csv")
    bak_path = os.path.join(PROC, f"dataset_filler_{variant}_old.csv")
    old = pd.read_csv(old_path)

    # drop any prosodic columns the corrected file might already have, then attach them
    base = fix.drop(columns=[c for c in cols if c in fix.columns])
    new = base.merge(pros[["word_id"] + cols], on="word_id", how="left")
    new = new.dropna(subset=cols)

    missing = [c for c in old.columns if c not in new.columns]
    if missing:
        print(f"\n[{variant}] STOP: these columns of the old file are not in the corrected file:", missing)
        continue
    new = new[list(old.columns)]

    same_ids = set(old.word_id) == set(new.word_id)
    chk = old[["word_id", "disfluency"]].merge(new[["word_id", "disfluency"]], on="word_id",
                                                suffixes=("_old", "_new"))
    print(f"\n[{variant}] old rows {len(old):,} -> new rows {len(new):,} | same word_ids: {same_ids}"
          f" | disfluency label mismatches: {(chk.disfluency_old != chk.disfluency_new).sum()}")
    m = old[["word_id", "foll_lex_surp"]].merge(new[["word_id", "foll_lex_surp"]], on="word_id",
                                                 suffixes=("_old", "_new"))
    print(f"[{variant}] rows where foll_lex_surp changed: "
          f"{(~np.isclose(m.foll_lex_surp_old, m.foll_lex_surp_new, equal_nan=True)).sum():,}")

    if not same_ids or len(new) != len(old):
        print(f"[{variant}] row set is different from the old file, NOT overwriting. send me this output.")
        continue

    if not os.path.exists(bak_path):
        shutil.copy(old_path, bak_path)
    new.to_csv(old_path, index=False)
    print(f"[{variant}] saved -> {old_path}  (old copy: {bak_path})")