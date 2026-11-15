"""
swap_syntactic.py
Puts the content-aligned ModelBlocks values in place of the old positional ones.
 - keeps a copy of the old file in data/processed_before_fix (only if not already there)
 - copies syntactic_surprisal_aligned.csv over syntactic_surprisal.csv
 - prints row counts so I can check them (expect 721941 and 601579)

run from repo root:  python src/swap_syntactic.py
"""
import os
import shutil
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
BAK_DIR = os.path.join(ROOT, "data", "processed_before_fix")
SRC = os.path.join(PROC, "syntactic_surprisal_aligned.csv")
DST = os.path.join(PROC, "syntactic_surprisal.csv")
BAK = os.path.join(BAK_DIR, "syntactic_surprisal.csv")

if not os.path.exists(SRC):
    raise SystemExit("not found: %s -- run align_mb_by_anchors.py first" % SRC)

os.makedirs(BAK_DIR, exist_ok=True)
if not os.path.exists(BAK):
    shutil.copyfile(DST, BAK)
    print("backed up old file ->", BAK)
else:
    print("backup already there, left as is:", BAK)

a = pd.read_csv(SRC)
print("aligned file : rows %d, prev_syn_surp valid %d, foll_syn_surp valid %d"
      % (len(a), a["prev_syn_surp"].notna().sum(), a["foll_syn_surp"].notna().sum()))
if len(a) != 721941 or a["prev_syn_surp"].notna().sum() != 601579:
    raise SystemExit("counts are not 721941 / 601579, not swapping. Send me this output.")

shutil.copyfile(SRC, DST)
b = pd.read_csv(DST)
print("now in place : rows %d, prev_syn_surp valid %d" % (len(b), b["prev_syn_surp"].notna().sum()))
print("DONE. next: python src/05_build_dataset.py")