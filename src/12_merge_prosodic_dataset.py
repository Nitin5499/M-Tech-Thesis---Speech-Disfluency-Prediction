"""
12_merge_prosodic_dataset.py
==============================
PROSODIC EXTENSION -- Stage 6 of 7.

Joins prosodic_features.csv onto the existing filler_v2 and
reparandum datasets, producing FOUR new dataset files rather than
two -- one accent-focused and one break-focused pair, for both the
filler and reparandum tasks. This split is a deliberate design
decision (see rationale below), not an arbitrary duplication.

WHY TWO SEPARATE COMPARISONS, NOT ONE COMBINED ONE:
  Stage 5 established that accent_strength has near-full corpus
  coverage (828,342/830,187 = 99.8%, since "not accented" = 0.0 is a
  real, legitimate value), while break_index has only ~12% coverage
  (100,192/830,187, since only 150 of 1,284 files have any prosody
  annotation at all, and there is no equivalent "default" value for
  break strength the way there is for accent).

  Requiring BOTH features to be present in one combined dataset would
  cap EVERY row at break_index's ~12% ceiling, needlessly discarding
  the much larger, more reliable accent_strength signal. Building two
  separate comparisons lets accent_strength be evaluated at
  near-full scale, and break_index be evaluated honestly on the
  smaller subset where it actually exists -- each getting a fair test
  on the data actually available for it, rather than both being
  dragged down to the smaller number's scale.

INPUT:
  data/processed/dataset_filler_v2.csv       (existing, unmodified)
  data/processed/dataset_reparandum.csv      (existing, unmodified)
  data/processed/prosodic_features.csv       (stage 5 output)

OUTPUT:
  data/processed/dataset_filler_accent.csv
  data/processed/dataset_filler_break.csv
  data/processed/dataset_reparandum_accent.csv
  data/processed/dataset_reparandum_break.csv

  Each accent file adds: prev_accent_strength, foll_accent_strength
  Each break file adds:  prev_break_index, foll_break_index

  Rows with a missing value in the RELEVANT prosodic column pair are
  dropped -- matching the existing pipeline's dropna(subset=...)
  convention used throughout 05_build_dataset.py and
  05_build_dataset_v2.py. This means dataset_filler_break.csv will be
  dramatically smaller than dataset_filler_accent.csv -- this is the
  expected, honest consequence of ~12% break_index coverage, not a
  bug, and is reported explicitly by this script's row-count logging.

NOTE ON word_id JOIN KEY FOR THE FILLER DATASET SPECIFICALLY:
  dataset_filler_v2.csv (from 05_build_dataset_v2.py) does not use
  the filler's OWN word_id as its row identity in a simple sense --
  each row represents a filler OCCURRENCE, built from the nearest
  preceding/following FLUENT word's features. The filler's own
  word_id IS still present as a column in that dataset (as
  "word_id"), so the join here is on that same column, attaching
  prosodic features of the filler word ITSELF (not the neighboring
  fluent words) -- consistent with treating prosody as a property of
  the disfluency event's own timing, distinct from the lexical/
  syntactic features which are deliberately drawn from the
  surrounding fluent words per the v2 fix's design.

Author: Nitin Kumar
Date: August 2026 (prosodic extension, stage 6)
"""

import os
import pandas as pd
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

FILLER_CSV        = os.path.join(PROCESSED, "dataset_filler_v2.csv")
REPARANDUM_CSV     = os.path.join(PROCESSED, "dataset_reparandum.csv")
PROSODIC_CSV       = os.path.join(PROCESSED, "prosodic_features.csv")

OUT_FILLER_ACCENT     = os.path.join(PROCESSED, "dataset_filler_accent.csv")
OUT_FILLER_BREAK      = os.path.join(PROCESSED, "dataset_filler_break.csv")
OUT_REPARANDUM_ACCENT = os.path.join(PROCESSED, "dataset_reparandum_accent.csv")
OUT_REPARANDUM_BREAK  = os.path.join(PROCESSED, "dataset_reparandum_break.csv")

LOG_FILE = os.path.join(PROCESSED, "pipeline_log.txt")


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def log_section(title):
    line = "\n" + "=" * 60 + f"\n{title}\n" + "=" * 60
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def build_variant(base_df, prosodic_df, cols, out_path, task_name, variant_name):
    """
    Merges the given prosodic columns onto base_df (on word_id),
    drops rows missing either of the two columns, saves, and logs
    the size change plainly.
    """
    merged = base_df.merge(prosodic_df[['word_id'] + cols], on='word_id', how='left')
    n_before = len(merged)

    merged_complete = merged.dropna(subset=cols)
    n_after = len(merged_complete)
    n_dropped = n_before - n_after
    pct_kept = 100 * n_after / n_before if n_before > 0 else 0.0

    log(f"\n  [{task_name} / {variant_name}] {cols[0]} + {cols[1]}")
    log(f"    Rows before dropna: {n_before:,}")
    log(f"    Rows after dropna:  {n_after:,}  ({pct_kept:.1f}% kept, {n_dropped:,} dropped)")

    merged_complete.to_csv(out_path, index=False)
    log(f"    Saved -> {out_path}")

    return n_before, n_after


if __name__ == "__main__":
    log_section("Prosodic Extension -- Stage 6: Merge Prosodic Features into Datasets")

    log("\nLoading existing datasets and prosodic features...")
    filler_df     = pd.read_csv(FILLER_CSV)
    reparandum_df = pd.read_csv(REPARANDUM_CSV)
    prosodic_df   = pd.read_csv(PROSODIC_CSV)

    log(f"  dataset_filler_v2.csv:      {len(filler_df):,} rows")
    log(f"  dataset_reparandum.csv:     {len(reparandum_df):,} rows")
    log(f"  prosodic_features.csv:      {len(prosodic_df):,} rows")

    if 'word_id' not in filler_df.columns:
        raise KeyError(
            "dataset_filler_v2.csv has no 'word_id' column -- check "
            "05_build_dataset_v2.py's output schema before proceeding; "
            "the join in this script assumes that column exists and "
            "refers to the filler word's own identity (see module "
            "docstring note on this)."
        )
    if 'word_id' not in reparandum_df.columns:
        raise KeyError("dataset_reparandum.csv has no 'word_id' column -- check its schema.")

    log_section("Building filler task variants")
    n_filler_before_a, n_filler_after_a = build_variant(
        filler_df, prosodic_df,
        ['prev_accent_strength', 'foll_accent_strength'],
        OUT_FILLER_ACCENT, "filler", "accent"
    )
    n_filler_before_b, n_filler_after_b = build_variant(
        filler_df, prosodic_df,
        ['prev_break_index', 'foll_break_index'],
        OUT_FILLER_BREAK, "filler", "break"
    )

    log_section("Building reparandum task variants")
    n_rep_before_a, n_rep_after_a = build_variant(
        reparandum_df, prosodic_df,
        ['prev_accent_strength', 'foll_accent_strength'],
        OUT_REPARANDUM_ACCENT, "reparandum", "accent"
    )
    n_rep_before_b, n_rep_after_b = build_variant(
        reparandum_df, prosodic_df,
        ['prev_break_index', 'foll_break_index'],
        OUT_REPARANDUM_BREAK, "reparandum", "break"
    )

    log_section("Stage 6 COMPLETE")
    log("Summary of all four dataset variants:")
    log(f"  filler_accent:     {n_filler_after_a:,} rows")
    log(f"  filler_break:      {n_filler_after_b:,} rows")
    log(f"  reparandum_accent: {n_rep_after_a:,} rows")
    log(f"  reparandum_break:  {n_rep_after_b:,} rows")
    log("\nHONEST CAVEAT: the *_break variants will be much smaller than the")
    log("*_accent variants, reflecting break_index's ~12% real annotation")
    log("coverage versus accent_strength's ~99.8% coverage (see stage 5 log")
    log("for the full explanation). This is expected, not an error, and")
    log("should be reported explicitly alongside any break_index result.")
    log("\nNext: 13_run_glmm_prosodic.R -- runs baseline vs. accent-enhanced")
    log("and baseline vs. break-enhanced formulas, on their respective")
    log("dataset variants, for both filler and reparandum tasks.")
    