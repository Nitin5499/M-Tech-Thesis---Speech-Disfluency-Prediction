"""
15_merge_gpt2_dataset.py

GPT-2 surprisal extension, stage 2.

Takes gpt2_surprisal.csv (values computed only on clean words, see
14_gpt2_surprisal.py) and:
  1. Propagates those values to disfluent words (fillers, reparandums)
     using the same propagate_feature() already used for lexical
     surprisal, DLT costs, syntactic surprisal, and the prosodic
     features. Not new logic -- reused on purpose to stay consistent
     with the rest of the pipeline.
  2. Merges the propagated GPT-2 columns onto dataset_filler_v2.csv and
     dataset_reparandum.csv additively -- neither file is modified, this
     just adds two new columns onto a joined copy.

Why propagation is needed here: gpt2_surprisal.csv only has rows for the
721,937 clean words it was computed on. Fillers/reparandums have no row
there at all, so a plain join leaves them missing -- same situation
lexical surprisal, DLT, and syntactic surprisal were in before their own
propagation steps, same as the prosodic features before
11_build_prosodic_features.py.

Why the filler dataset needs its own word_id join: dataset_filler_v2.csv
(from 05_build_dataset_v2.py) represents each row as a filler occurrence
built from the surrounding fluent words' features, but the filler's own
word_id column is still there. Same reasoning as attaching prosody to the
filler's own timing in 12_merge_prosodic_dataset.py -- GPT-2 surprisal
here is attached via the filler occurrence's own word_id, treated as a
property of the disfluency's position in the conversation, not borrowed
from a neighboring fluent word.

in:  data/processed/gpt2_surprisal.csv     (stage 1, clean words only)
     data/processed/words_labeled.csv      (conv_id/speaker/start, for propagate_feature())
     data/processed/dataset_filler_v2.csv
     data/processed/dataset_reparandum.csv
out: data/processed/gpt2_surprisal_propagated.csv
     columns: word_id, prev_gpt2_surp, foll_gpt2_surp (now covers all words)
     data/processed/dataset_filler_gpt2.csv
     data/processed/dataset_reparandum_gpt2.csv

Nitin, Sep 2026 (GPT-2 surprisal extension, stage 2)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

GPT2_CSV       = os.path.join(PROCESSED, "gpt2_surprisal.csv")
LABELED_CSV    = os.path.join(PROCESSED, "words_labeled.csv")
FILLER_CSV     = os.path.join(PROCESSED, "dataset_filler_v2.csv")
REPARANDUM_CSV = os.path.join(PROCESSED, "dataset_reparandum.csv")

OUT_PROPAGATED      = os.path.join(PROCESSED, "gpt2_surprisal_propagated.csv")
OUT_FILLER_GPT2     = os.path.join(PROCESSED, "dataset_filler_gpt2.csv")
OUT_REPARANDUM_GPT2 = os.path.join(PROCESSED, "dataset_reparandum_gpt2.csv")

LOG_FILE = os.path.join(PROCESSED, "pipeline_log.txt")


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def log_section(title):
    line = "\n" + "-" * 50 + f"\n{title}\n" + "-" * 50
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def propagate_feature(df, col_prev, col_foll):
    """
    Same as the version in 05_build_dataset.py / 11_build_prosodic_features.py,
    copied here rather than reinvented. For any word missing a feature
    value, copies forward the nearest preceding clean word's value within
    the same (conv_id, speaker) conversation.
    """
    df = df.sort_values(["conv_id", "speaker", "start"]).reset_index(drop=True)
    result_prev = df[col_prev].tolist()
    result_foll = df[col_foll].tolist()
    n_propagated = 0
    n_still_missing = 0

    for (conv_id, speaker), group in df.groupby(["conv_id", "speaker"], sort=False):
        indices = group.index.tolist()
        prev_vals = group[col_prev].tolist()
        foll_vals = group[col_foll].tolist()

        last_valid_prev = np.nan
        last_valid_foll = np.nan

        for i, idx in enumerate(indices):
            val = prev_vals[i]
            is_missing = (val is None) or (isinstance(val, float) and np.isnan(val))

            if not is_missing:
                last_valid_prev = prev_vals[i]
                last_valid_foll = foll_vals[i]
            else:
                lv_missing = (last_valid_prev is None) or (
                    isinstance(last_valid_prev, float) and np.isnan(last_valid_prev))
                if not lv_missing:
                    result_prev[idx] = last_valid_prev
                    result_foll[idx] = last_valid_foll
                    n_propagated += 1
                else:
                    n_still_missing += 1

    df[col_prev] = result_prev
    df[col_foll] = result_foll
    return df, n_propagated, n_still_missing


if __name__ == "__main__":
    log_section("GPT-2 surprisal extension, stage 2: propagate and merge")

    log("loading gpt2_surprisal.csv (clean words only)...")
    gpt2_df = pd.read_csv(GPT2_CSV)
    log(f"  {len(gpt2_df):,} clean words with a directly-computed value")

    log("loading words_labeled.csv (conv_id/speaker/start)...")
    labeled_df = pd.read_csv(LABELED_CSV)
    log(f"  {len(labeled_df):,} words total in corpus")

    log("merging onto full word set (including disfluent words)...")
    full_df = labeled_df[['word_id', 'conv_id', 'speaker', 'start']].merge(
        gpt2_df[['word_id', 'prev_gpt2_surp', 'foll_gpt2_surp']],
        on='word_id', how='left'
    )
    n_missing_before = full_df['prev_gpt2_surp'].isna().sum()
    log(f"  full word set size: {len(full_df):,}")
    log(f"  words with no GPT-2 value before propagation (fillers/reparandums "
        f"+ any dropped rows from stage 1): {n_missing_before:,}")

    log("propagating prev_/foll_ GPT-2 surprisal to disfluent words...")
    full_df, n_propagated, n_still_missing = propagate_feature(
        full_df, 'prev_gpt2_surp', 'foll_gpt2_surp'
    )
    log(f"  propagated: {n_propagated:,}")
    log(f"  still missing (no clean word earlier in that conversation): {n_still_missing:,}")

    propagated_df = full_df[['word_id', 'prev_gpt2_surp', 'foll_gpt2_surp']]
    propagated_df.to_csv(OUT_PROPAGATED, index=False)
    log(f"  saved -> {OUT_PROPAGATED}")

    log_section("merging into filler and reparandum datasets")

    log("loading dataset_filler_v2.csv and dataset_reparandum.csv...")
    filler_df = pd.read_csv(FILLER_CSV)
    reparandum_df = pd.read_csv(REPARANDUM_CSV)
    log(f"  dataset_filler_v2.csv: {len(filler_df):,} rows")
    log(f"  dataset_reparandum.csv: {len(reparandum_df):,} rows")

    if 'word_id' not in filler_df.columns:
        raise KeyError(
            "dataset_filler_v2.csv has no 'word_id' column -- check "
            "05_build_dataset_v2.py's output schema before continuing"
        )
    if 'word_id' not in reparandum_df.columns:
        raise KeyError("dataset_reparandum.csv has no 'word_id' column -- check its schema")

    gpt2_cols = ['prev_gpt2_surp', 'foll_gpt2_surp']

    filler_gpt2 = filler_df.merge(propagated_df[['word_id'] + gpt2_cols], on='word_id', how='left')
    n_before = len(filler_gpt2)
    filler_gpt2_complete = filler_gpt2.dropna(subset=gpt2_cols)
    n_after = len(filler_gpt2_complete)
    log(f"  [filler] rows before dropna: {n_before:,}")
    log(f"  [filler] rows after dropna: {n_after:,} ({100*n_after/n_before:.1f}% kept, {n_before-n_after:,} dropped)")
    filler_gpt2_complete.to_csv(OUT_FILLER_GPT2, index=False)
    log(f"  saved -> {OUT_FILLER_GPT2}")

    reparandum_gpt2 = reparandum_df.merge(propagated_df[['word_id'] + gpt2_cols], on='word_id', how='left')
    n_before = len(reparandum_gpt2)
    reparandum_gpt2_complete = reparandum_gpt2.dropna(subset=gpt2_cols)
    n_after = len(reparandum_gpt2_complete)
    log(f"  [reparandum] rows before dropna: {n_before:,}")
    log(f"  [reparandum] rows after dropna: {n_after:,} ({100*n_after/n_before:.1f}% kept, {n_before-n_after:,} dropped)")
    reparandum_gpt2_complete.to_csv(OUT_REPARANDUM_GPT2, index=False)
    log(f"  saved -> {OUT_REPARANDUM_GPT2}")

    log_section("stage 2 done")
    log("summary:")
    log(f"  gpt2_surprisal_propagated.csv: {len(propagated_df):,} rows (full corpus)")
    log(f"  dataset_filler_gpt2.csv: {len(filler_gpt2_complete):,} rows")
    log(f"  dataset_reparandum_gpt2.csv: {len(reparandum_gpt2_complete):,} rows")
    log("unlike prosodic break_index (~12% coverage), GPT-2 surprisal was computed")
    log("on the full clean corpus, so coverage here should be close to complete --")
    log("any real drop in row count above is worth checking, not just expected")
    log("next: 16_run_glmm_gpt2.R -- baseline vs baseline+GPT2 comparison, mirroring")
    log("13_run_glmm_prosodic.R's structure, for both filler and reparandum tasks")