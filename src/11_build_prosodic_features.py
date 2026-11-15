"""
11_build_prosodic_features.py

Prosodic extension, stage 5.

Takes word_prosody_bridge.csv (one row per word_id, accent/break values
already resolved) and builds the prev_*/foll_* shifted pairs, same pattern
as everywhere else in the pipeline (prev_lex_surp/foll_lex_surp,
prev_IC/foll_IC, etc.).

Why shift here too: same reason as lexical surprisal / DLT / syntactic
surprisal -- the GLMM needs the prosodic properties of the word BEFORE a
possible disfluency and the word AFTER it, not the disfluency's own
prosody. Built within each clean sentence using clean_word_index.csv for
ordering, same coordinate system every other feature script here uses.

Why propagation is still needed (and reused, not rewritten): three sources
of missing data --
  1. only 150 of 1,284 files (11.7% of the corpus) have any accent/break
     annotation at all -- prosody annotation only covers a subset
  2. even within that subset, not every word is accented and not every
     word has a break entry (breaks mark boundaries after certain words,
     not all of them)
  3. fillers/reparandums have no value of their own, same as every other
     feature here -- prosodic features are only meaningfully computed on
     clean words
All three go through the same propagate_feature() as 05_build_dataset.py
(reproduced here rather than importing across files, but identical logic)
so behavior stays consistent with the rest of the pipeline.

in:  data/processed/word_prosody_bridge.csv (stage 4)
     data/processed/clean_word_index.csv
     data/processed/words_labeled.csv
out: data/processed/prosodic_features.csv
     columns: word_id, prev_accent_strength, foll_accent_strength,
     prev_break_index, foll_break_index

on accent_strength encoding: it's categorical in the raw data (full/weak).
For the GLMM, encoding it as full=1.0, weak=0.5, no accent=0.0. This is a
deliberate choice, not an approximation -- "full" is the strongest
prosodic prominence, "weak" is real but lesser, and no accent entry means
no prominence (per the corpus docs, only accented words show up in
accents.xml at all, so absence is a real 0, not missing data). Don't mix
this up with actually missing data (outside the 150-file subset, or
non-aligned words) -- that stays NaN and goes through propagation instead.

Nitin, Aug 2026 (prosodic extension, stage 5)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

BRIDGE_CSV      = os.path.join(PROCESSED, "word_prosody_bridge.csv")
CLEAN_INDEX_CSV = os.path.join(PROCESSED, "clean_word_index.csv")
LABELED_CSV     = os.path.join(PROCESSED, "words_labeled.csv")
OUTPUT_CSV      = os.path.join(PROCESSED, "prosodic_features.csv")
LOG_FILE        = os.path.join(PROCESSED, "pipeline_log.txt")


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


def encode_accent_strength(strength):
    """
    full -> 1.0, weak -> 0.5, no accent entry -> 0.0 (this word just
    wasn't accented, not missing data -- see module docstring).
    Genuinely missing data is handled separately by propagation.
    """
    if pd.isna(strength):
        return 0.0
    s = str(strength).strip().lower()
    if s == 'full':
        return 1.0
    if s == 'weak':
        return 0.5
    return 0.0


def propagate_feature(df, col_prev, col_foll):
    """
    Same as propagate_feature() in 05_build_dataset.py -- copied here
    instead of importing across files, but identical logic. For any word
    missing a value, copy forward the nearest preceding clean word's
    value within the same (conv_id, speaker) conversation.
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
    log_section("prosodic extension, stage 5: build prev_/foll_ prosodic features")

    log("loading word_prosody_bridge.csv...")
    bridge_df = pd.read_csv(BRIDGE_CSV)
    log(f"  {len(bridge_df):,} words in bridge")

    log("loading clean_word_index.csv (sentence/position ordering)...")
    clean_idx = pd.read_csv(CLEAN_INDEX_CSV)
    log(f"  {len(clean_idx):,} clean words")

    log("loading words_labeled.csv (conv_id/speaker/start, needed for propagation)...")
    labeled_df = pd.read_csv(LABELED_CSV)

    log("encoding accent strength (full=1.0, weak=0.5, none=0.0)...")
    bridge_df['accent_strength_numeric'] = bridge_df['accent_strength'].apply(encode_accent_strength)

    # 0.0 should only mean "not accented" inside the 150 annotated speaker-sides.
    # For the other files there is no annotation at all, so those stay NaN.
    tmp = bridge_df[['word_id', 'accent_strength']].merge(
        labeled_df[['word_id', 'conv_id', 'speaker']], on='word_id', how='left')
    tmp['side'] = tmp['conv_id'].astype(str) + tmp['speaker'].astype(str)
    annotated_sides = set(tmp.loc[tmp['accent_strength'].notna(), 'side'])
    in_annotated = tmp['side'].isin(annotated_sides).values
    bridge_df.loc[~in_annotated, 'accent_strength_numeric'] = np.nan
    log(f"  accent annotated speaker-sides: {len(annotated_sides)}; words outside them set to NaN: {(~in_annotated).sum():,}")

    # restrict to clean words first, same as every other feature script --
    # compute the prev_/foll_ shift on the clean corpus, propagate to
    # disfluent words after. same design as lexical surprisal, DLT,
    # syntactic surprisal.
    clean_bridge = clean_idx.merge(bridge_df, on='word_id', how='left')
    log(f"  clean words with bridge data merged: {len(clean_bridge):,}")

    n_no_accent_data = clean_bridge['accent_strength'].isna().sum()
    n_no_break_data = clean_bridge['break_index_clean'].isna().sum()
    log(f"  clean words with no accent entry (encoded 0.0, not missing): {n_no_accent_data:,}")
    log(f"  clean words with no break_index value (genuinely missing): {n_no_break_data:,}")

    log("building prev_/foll_ shifted pairs within clean sentences...")
    results = []
    groups = list(clean_bridge.groupby('sent_idx', sort=True))
    total = len(groups)

    for g_idx, (sent_idx, group) in enumerate(groups):
        group = group.sort_values('word_pos').reset_index(drop=True)
        n = len(group)
        accent_vals = group['accent_strength_numeric'].tolist()
        break_vals = group['break_index_clean'].tolist()
        ids = group['word_id'].tolist()

        for i in range(n):
            prev_accent = accent_vals[i]
            foll_accent = accent_vals[i + 1] if i < n - 1 else np.nan
            prev_break = break_vals[i]
            foll_break = break_vals[i + 1] if i < n - 1 else np.nan

            results.append({
                "word_id": ids[i],
                "prev_accent_strength": prev_accent,
                "foll_accent_strength": foll_accent,
                "prev_break_index": prev_break,
                "foll_break_index": foll_break,
            })

        if (g_idx + 1) % 5000 == 0:
            log(f"  {g_idx+1}/{total} sentences done")

    prosody_df = pd.DataFrame(results)
    log(f"  built shifted features for {len(prosody_df):,} clean words")

    # need conv_id/speaker/start for propagate_feature() to know conversation
    # boundaries
    prosody_df = prosody_df.merge(
        labeled_df[['word_id', 'conv_id', 'speaker', 'start']], on='word_id', how='left'
    )

    log("merging onto the full word set (including disfluent words) for propagation...")
    full_df = labeled_df[['word_id', 'conv_id', 'speaker', 'start']].merge(
        prosody_df[['word_id', 'prev_accent_strength', 'foll_accent_strength',
                     'prev_break_index', 'foll_break_index']],
        on='word_id', how='left'
    )
    log(f"  full word set size: {len(full_df):,}")

    log("propagating prev_/foll_ accent_strength to disfluent words...")
    full_df, n_prop_acc, n_missing_acc = propagate_feature(
        full_df, 'prev_accent_strength', 'foll_accent_strength'
    )
    log(f"  propagated: {n_prop_acc:,}, still missing: {n_missing_acc:,}")

    log("propagating prev_/foll_ break_index to disfluent words...")
    full_df, n_prop_brk, n_missing_brk = propagate_feature(
        full_df, 'prev_break_index', 'foll_break_index'
    )
    log(f"  propagated: {n_prop_brk:,}, still missing: {n_missing_brk:,}")

    output_df = full_df[['word_id', 'prev_accent_strength', 'foll_accent_strength',
                          'prev_break_index', 'foll_break_index']]
    output_df.to_csv(OUTPUT_CSV, index=False)

    log_section("stage 5 done")
    log(f"saved -> {OUTPUT_CSV}")
    log(f"  total rows: {len(output_df):,}")
    log(f"  rows with a real prev_accent_strength: {output_df['prev_accent_strength'].notna().sum():,}")
    log(f"  rows with a real prev_break_index: {output_df['prev_break_index'].notna().sum():,}")
    log("caveat for the GLMM stage: break_index coverage is going to look much lower than")
    log("every other feature here (which sit at 70-100%), since only 11.7% of the corpus")
    log("has prosody annotation, and break_index doesn't have a meaningful default the way")
    log("accent_strength does (0.0 = 'not accented' is a real value; there's no equivalent")
    log("default for break strength). so accent_strength coverage looks much higher -- that's")
    log("expected, not a bug -- but break_index's ~12% is the honest number for this")
    log("extension's actual annotation scope and is the one to lead with in the writeup.")
    log("next: 12_merge_prosodic_dataset.py -- joins this onto words_features.csv to build")
    log("dataset_filler_v3.csv and dataset_reparandum_v3.csv")