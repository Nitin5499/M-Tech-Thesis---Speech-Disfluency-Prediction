"""
05_build_dataset.py
===================
Merges all feature files and builds the final balanced binary datasets.

Full Clean Pipeline — ALL 3 features computed on clean corpus:

  1. Lexical surprisal: computed on CLEAN sentences (clean_word_index.csv)
     — disfluent words propagate from nearest clean word before them

  2. DLT costs: computed on CLEAN sentences (clean_word_index.csv)
     — disfluent words propagate from nearest clean word before them
     — same approach as lexical and syntactic surprisal

  3. Syntactic surprisal: ModelBlocks on CLEAN sentences
     — disfluent words propagate from nearest clean word before them

All 3 features use the same propagation strategy — consistent and
does not create artificial signals in the GLMM.

INPUT:
  data/processed/words_labeled.csv
  data/processed/lexical_surprisal.csv      -- clean words only (721,941)
  data/processed/dlt_costs.csv              -- clean words only (721,941)
  data/processed/syntactic_surprisal.csv    -- ModelBlocks clean words (517,907)

OUTPUT:
  data/processed/words_features.csv
  data/processed/dataset_filler.csv
  data/processed/dataset_reparandum.csv

Author: Nitin Kumar
Date: June 2026
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
PROCESSED   = os.path.join(ROOT_DIR, "data", "processed")

LABELED_CSV  = os.path.join(PROCESSED, "words_labeled.csv")
LEXSURP_CSV  = os.path.join(PROCESSED, "lexical_surprisal.csv")
DLT_CSV      = os.path.join(PROCESSED, "dlt_costs.csv")
SYNSURP_CSV  = os.path.join(PROCESSED, "syntactic_surprisal.csv")
FEATURES_CSV = os.path.join(PROCESSED, "words_features.csv")
FILLER_CSV   = os.path.join(PROCESSED, "dataset_filler.csv")
REP_CSV      = os.path.join(PROCESSED, "dataset_reparandum.csv")
LOG_FILE     = os.path.join(PROCESSED, "pipeline_log.txt")

CONTENT_POS = {
    "NN","NNS","NNP","NNPS",
    "VB","VBD","VBG","VBN","VBP","VBZ",
    "JJ","JJR","JJS","RB","RBR","RBS","CD","FW",
}

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

# ── Generic feature propagation ────────────────────────────────
def propagate_feature(df, col_prev, col_foll):
    """
    For words missing a feature value — propagate from the
    nearest clean word before them in the same conversation.

    Used for ALL 3 features — lexical surprisal, DLT costs,
    and syntactic surprisal — for consistency.
    """
    df = df.sort_values(["conv_id", "speaker", "start"]).reset_index(drop=True)
    result_prev     = df[col_prev].tolist()
    result_foll     = df[col_foll].tolist()
    n_propagated    = 0
    n_still_missing = 0

    for (conv_id, speaker), group in df.groupby(["conv_id", "speaker"], sort=False):
        indices   = group.index.tolist()
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

# ── Load and merge ─────────────────────────────────────────────
def load_and_merge():
    log("Loading words_labeled.csv...")
    df = pd.read_csv(LABELED_CSV)
    df = df.dropna(subset=["duration"]).reset_index(drop=True)
    df = df[df["duration"] > 0]
    df["word"] = df["word"].astype(str)
    log(f"  Total words: {len(df):,}")
    log(f"  Fluent:      {(df['label']=='fluent').sum():,}")
    log(f"  Filler:      {(df['label']=='filler').sum():,}")
    log(f"  Reparandum:  {(df['label']=='reparandum').sum():,}")

    # ── Lexical surprisal — clean + propagate ─────────────────
    log("\nMerging lexical surprisal (clean corpus)...")
    lex_df = pd.read_csv(LEXSURP_CSV)
    log(f"  Clean words in file: {len(lex_df):,}")
    df = df.merge(lex_df, on="word_id", how="left")
    n_lex     = df['prev_lex_surp'].notna().sum()
    n_missing = df['prev_lex_surp'].isna().sum()
    log(f"  prev_lex_surp from clean corpus: {n_lex:,}")
    log(f"  Missing (disfluent words):       {n_missing:,}")
    log("  Propagating to disfluent words...")
    df, n_prop, n_miss = propagate_feature(df, "prev_lex_surp", "foll_lex_surp")
    log(f"  Propagated: {n_prop:,}  Still missing: {n_miss:,}")
    log(f"  Total prev_lex_surp valid: {df['prev_lex_surp'].notna().sum():,}")

    # ── DLT costs — clean + propagate ─────────────────────────
    log("\nMerging DLT costs (clean sentences)...")
    dlt_df = pd.read_csv(DLT_CSV)
    log(f"  Clean words in file: {len(dlt_df):,}")
    df = df.merge(dlt_df, on="word_id", how="left")
    n_missing_dlt = df['prev_IC'].isna().sum()
    log(f"  prev_IC from clean sentences: {df['prev_IC'].notna().sum():,}")
    log(f"  Missing (disfluent words):    {n_missing_dlt:,}")
    log("  Propagating IC and SC to disfluent words...")
    df, n_prop_ic, n_miss_ic = propagate_feature(df, "prev_IC", "foll_IC")
    df, n_prop_sc, n_miss_sc = propagate_feature(df, "prev_SC", "foll_SC")
    log(f"  IC propagated: {n_prop_ic:,}  Still missing: {n_miss_ic:,}")
    log(f"  SC propagated: {n_prop_sc:,}  Still missing: {n_miss_sc:,}")
    log(f"  Total prev_IC valid: {df['prev_IC'].notna().sum():,}")
    log(f"  Total prev_SC valid: {df['prev_SC'].notna().sum():,}")

    # ── Syntactic surprisal — ModelBlocks + propagate ─────────
    if os.path.exists(SYNSURP_CSV):
        log("\nMerging syntactic surprisal (ModelBlocks clean words)...")
        syn_df = pd.read_csv(SYNSURP_CSV)
        log(f"  Clean words in file: {len(syn_df):,}")
        df = df.merge(syn_df, on="word_id", how="left")
        n_mb      = df['prev_syn_surp'].notna().sum()
        n_missing = df['prev_syn_surp'].isna().sum()
        log(f"  prev_syn_surp from ModelBlocks: {n_mb:,}")
        log(f"  Missing (disfluent words):      {n_missing:,}")
        log("  Propagating to disfluent words...")
        df, n_prop, n_miss = propagate_feature(df, "prev_syn_surp", "foll_syn_surp")
        log(f"  Propagated: {n_prop:,}  Still missing: {n_miss:,}")
        log(f"  Total prev_syn_surp valid: {df['prev_syn_surp'].notna().sum():,}")
    else:
        log("\nWARNING: syntactic_surprisal.csv not found")
        df["prev_syn_surp"] = np.nan
        df["foll_syn_surp"] = np.nan

    return df

# ── Duration features ──────────────────────────────────────────
def add_duration_features(df):
    log("\nAdding duration features...")
    df = df.sort_values(["conv_id","speaker","start"]).reset_index(drop=True)
    word_mean = df.groupby("word")["duration"].transform("mean")
    prev_durs = []
    foll_durs = []
    for (conv_id, speaker), group in df.groupby(["conv_id","speaker"], sort=False):
        durs = group["duration"].tolist()
        n = len(durs)
        for i in range(n):
            prev_durs.append(durs[i])
            foll_durs.append(durs[i+1] if i+1 < n else np.nan)
    df["prev_dur"]   = prev_durs
    df["foll_dur"]   = foll_durs
    df["elongation"] = df["duration"] / word_mean
    return df

def add_pos_features(df):
    df["is_content"] = df["pos"].apply(lambda p: 1 if p in CONTENT_POS else 0)
    return df

def handle_rare_words(df):
    freq = df["word"].value_counts()
    rare = set(freq[freq < 30].index)
    df["word_for_model"] = df["word"].apply(
        lambda w: w if w not in rare else "<UNK>"
    )
    log(f"  Rare word types -> <UNK>: {len(rare):,}")
    return df

# ── Build datasets ─────────────────────────────────────────────
def build_binary_datasets(df):
    log_section("Building Binary Datasets")

    required_cols = [
        "prev_lex_surp", "foll_lex_surp",
        "prev_syn_surp", "foll_syn_surp",
        "prev_IC",       "foll_IC",
        "prev_SC",       "foll_SC",
        "prev_dur",      "foll_dur",
    ]

    df_clean = df.dropna(subset=required_cols).reset_index(drop=True)
    log(f"  Rows with ALL genuine features: {len(df_clean):,}")

    fluent      = df_clean[df_clean["label"] == "fluent"]
    fillers     = df_clean[df_clean["label"] == "filler"]
    reparandums = df_clean[df_clean["label"] == "reparandum"]

    log(f"  Fluent:     {len(fluent):,}")
    log(f"  Filler:     {len(fillers):,}")
    log(f"  Reparandum: {len(reparandums):,}")

    # Filler dataset
    n_f = len(fillers)
    if n_f > 0:
        filler_ds = pd.concat([
            fillers.assign(disfluency=1),
            fluent.sample(n=min(n_f, len(fluent)), random_state=42).assign(disfluency=0),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        filler_ds = pd.DataFrame()
        log("  WARNING: No fillers with complete features")

    # Reparandum dataset
    n_r = len(reparandums)
    if n_r > 0:
        rep_ds = pd.concat([
            reparandums.assign(disfluency=1),
            fluent.sample(n=min(n_r, len(fluent)), random_state=42).assign(disfluency=0),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        rep_ds = pd.DataFrame()
        log("  WARNING: No reparandums with complete features")

    log(f"\n  dataset_filler.csv:     {len(filler_ds):,} rows "
        f"({n_f} fillers + {n_f} fluent) — paper target: 24,100")
    log(f"  dataset_reparandum.csv: {len(rep_ds):,} rows "
        f"({n_r} reparandums + {n_r} fluent) — paper target: 29,040")

    return filler_ds, rep_ds

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    log_section("Step 05: Build Feature Dataset (Full Clean Pipeline)")
    log("ALL 3 features computed on clean corpus with propagation:")
    log("  Lexical surprisal:   clean sentences + propagation to disfluent")
    log("  DLT costs:           clean sentences + propagation to disfluent")
    log("  Syntactic surprisal: ModelBlocks clean + propagation to disfluent")

    df = load_and_merge()
    df = add_duration_features(df)
    df = add_pos_features(df)
    df = handle_rare_words(df)

    df.to_csv(FEATURES_CSV, index=False)
    log(f"\nSaved -> {FEATURES_CSV}")
    log(f"  Total rows: {len(df):,}")

    filler_ds, rep_ds = build_binary_datasets(df)

    filler_ds.to_csv(FILLER_CSV, index=False)
    rep_ds.to_csv(REP_CSV, index=False)
    log(f"\nSaved -> {FILLER_CSV}")
    log(f"Saved -> {REP_CSV}")

    log_section("Step 05 COMPLETE — ready for GLMM")
    log("Next: Rscript src/06_run_glmm.R")