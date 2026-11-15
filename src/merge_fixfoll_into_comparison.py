"""
merge_fixfoll_into_comparison.py

Puts the corrected filler rows (results/table_filler_fixfoll_all_models.csv) into
results/table_multi_llm_comparison.csv, so the figure script picks them up.
The old filler rows are replaced; reparandum rows stay as they are.
A backup of the old table is written next to it first.

run from repo root:  python src/merge_fixfoll_into_comparison.py
"""
import os
import shutil
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OLD = os.path.join(RES, "table_multi_llm_comparison.csv")
NEW = os.path.join(RES, "table_filler_fixfoll_all_models.csv")
BACKUP = os.path.join(RES, "table_multi_llm_comparison_before_offset_fix.csv")

old = pd.read_csv(OLD)
new = pd.read_csv(NEW)
print("old table:", len(old), "rows; columns:", list(old.columns))
print("new filler rows:", len(new), "; columns:", list(new.columns))

if not os.path.exists(BACKUP):
    shutil.copy(OLD, BACKUP)
    print("backup ->", BACKUP)

keep = old[old["task"] != "filler"]
missing = [c for c in old.columns if c not in new.columns]
if missing:
    print("note: new table has no column(s) %s, they will be empty for filler rows" % missing)
new = new.reindex(columns=old.columns)

out = pd.concat([new, keep], ignore_index=True)
out.to_csv(OLD, index=False)
print("\nsaved ->", OLD)
print(out.to_string(index=False))