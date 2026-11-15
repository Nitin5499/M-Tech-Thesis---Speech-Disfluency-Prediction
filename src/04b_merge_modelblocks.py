"""
04b_merge_modelblocks.py
Reads ModelBlocks tokdecs output and merges syntactic surprisal
values back into the pipeline — replacing PCFG syntactic surprisal.

Run this after synproc finishes generating swbd_clean.tokdecs

INPUT:
  ModelBlocks tokdecs file
  data/processed/words_labeled.csv

OUTPUT:
  data/processed/syntactic_surprisal_modelblocks.csv  (new MB values)
  data/processed/syntactic_surprisal.csv              (overwritten with MB values)
"""

import os
import pandas as pd
import numpy as np

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELED_CSV = os.path.join(ROOT_DIR, "data", "processed", "words_labeled.csv")
MB_TOKDECS  = os.path.join(
    ROOT_DIR,
    "Modelbock release master",
    "modelblocks-release-master",
    "workspace",
    "genmodel", "testing", "swbd_clean.tokdecs"
)
OUT_MB  = os.path.join(ROOT_DIR, "data", "processed",
                        "syntactic_surprisal_modelblocks.csv")
OUT_CSV = os.path.join(ROOT_DIR, "data", "processed",
                        "syntactic_surprisal.csv")

print("=" * 60)
print("Step 4b: Merging ModelBlocks Syntactic Surprisal")
print("=" * 60)

# ── Load ModelBlocks output ───────────────────────────────────
print(f"\nReading ModelBlocks output from:\n  {MB_TOKDECS}")
rows = []
with open(MB_TOKDECS) as f:
    header = f.readline().strip().split()
    print(f"  Columns: {header}")
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            word = parts[0]
            surp = parts[-1]
            rows.append({"word": word, "totsurp": surp})

mb_df = pd.DataFrame(rows)
print(f"  Total rows: {len(mb_df):,}")

# Convert totsurp to float — handle FAIL and -nan
mb_df["totsurp"] = pd.to_numeric(mb_df["totsurp"], errors="coerce")

# Remove ARTICLE markers
mb_df = mb_df[mb_df["word"] != "!ARTICLE"].reset_index(drop=True)

print(f"  After removing !ARTICLE: {len(mb_df):,}")
print(f"  Valid surprisal values: {mb_df['totsurp'].notna().sum():,}")
print(f"  FAIL/nan values: {mb_df['totsurp'].isna().sum():,}")
print(f"  Mean surprisal (valid): {mb_df['totsurp'].mean():.4f}")

# ── Load labeled words ────────────────────────────────────────
print("\nLoading words_labeled.csv...")
words = pd.read_csv(LABELED_CSV)
words = words[words['label'] == 'fluent']
words = words.sort_values(['conv_id','speaker','start']).reset_index(drop=True)
words['word_clean'] = words['word'].fillna('').astype(str)
words['word_clean'] = words['word_clean'].str.replace(
    r'[^a-zA-Z0-9\'\-]', '', regex=True).str.lower()
words = words[words['word_clean'].str.len() > 0].reset_index(drop=True)
print(f"  Fluent words: {len(words):,}")

# ── Align by position ─────────────────────────────────────────
# ModelBlocks output is word by word in the same order as input
print("\nAligning surprisal values to words...")
min_len = min(len(words), len(mb_df))
print(f"  Words in corpus: {len(words):,}")
print(f"  Words in MB output: {len(mb_df):,}")
print(f"  Aligning first {min_len:,} words")

words = words.iloc[:min_len].copy()
mb_df = mb_df.iloc[:min_len].copy()

words["syn_surp_mb"] = mb_df["totsurp"].values

# Replace NaN with mean (for FAIL cases)
mean_surp = words["syn_surp_mb"].mean()
n_replaced = words["syn_surp_mb"].isna().sum()
words["syn_surp_mb"] = words["syn_surp_mb"].fillna(mean_surp)
print(f"  Replaced {n_replaced:,} FAIL values with mean ({mean_surp:.4f})")

# ── Compute prev and foll surprisal ──────────────────────────
words["prev_syn_surp"] = words["syn_surp_mb"]
words["foll_syn_surp"] = (words.groupby(["conv_id","speaker"])
                               ["syn_surp_mb"].shift(-1))

# ── Save output ───────────────────────────────────────────────
out_df = words[["word_id","prev_syn_surp","foll_syn_surp"]].copy()
out_df.to_csv(OUT_MB,  index=False)
out_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved ModelBlocks surprisal -> {OUT_MB}")
print(f"Overwrote syntactic_surprisal.csv -> {OUT_CSV}")
print(f"Rows saved: {len(out_df):,}")
print("\nSample:")
print(out_df.head(10).to_string())

print("\n" + "=" * 60)
print("Step 4b COMPLETE")
print("=" * 60)
print("\nNext steps:")
print("  1. python src/05_build_dataset.py")
print('  2. Rscript src/06_run_glmm.R  (rename results to *_modelblocks)')
print("  3. python src/10_compare_results.py")