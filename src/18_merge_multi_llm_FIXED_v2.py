"""
18_merge_multi_llm.py

Multi-LLM surprisal extension -- Stage 2.

Merges each surprisal_<tag>.csv from 17_multi_llm_surprisal.py into
dataset_filler_v2.csv and dataset_reparandum.csv, one column pair per model.

FIX #1 (Oct 2026): this script originally read dataset_filler.csv, but
15_merge_gpt2_dataset.py (and therefore 16_run_glmm_gpt2.R's "+0.93 pts"
filler result) used dataset_filler_v2.csv -- a later, larger revision of
the same dataset (41,446 rows vs. ~29,444). Switched FILLER_IN to
dataset_filler_v2.csv to match script 15 / script 16.

FIX #2 (Oct 2026): dataset_filler_v2.csv has a different schema than
dataset_filler.csv -- no conv_id/speaker/start columns, so the
conversation-order ffill/bfill propagate_feature() used for reparandum
cannot run on it (KeyError: 'conv_id'). But it doesn't need to: unlike
dataset_reparandum.csv (one row per disfluent word, needing propagation
from the nearest earlier clean word in the conversation), each filler row
in dataset_filler_v2.csv already names its own two context anchors --
prev_fluent_word_id and foll_fluent_word_id. Each surprisal_<tag>.csv is
keyed by word_id with that clean word's own prev_<tag>_surp/foll_<tag>_surp
already computed (sentence-internal, from script 17). So the correct join
for the filler file is a direct two-step lookup:
    filler row's prev_fluent_word_id -> that word's own prev_<tag>_surp
    filler row's foll_fluent_word_id -> that word's own foll_<tag>_surp
This mirrors exactly what 15_merge_gpt2_dataset.py did for GPT-2 (it
pre-propagates onto the full corpus via words_labeled.csv, which DOES have
conv_id/speaker/start, then merges onto dataset_filler_v2.csv by word_id --
no ffill/bfill at the filler-dataset stage itself). Reparandum keeps using
propagate_feature() unchanged, since dataset_reparandum.csv does carry
conv_id/speaker/start and that path was never broken.

in:  data/processed/surprisal_<tag>.csv     (one per model, from stage 1;
                                              columns: word_id, prev_<tag>_surp,
                                              foll_<tag>_surp)
     data/processed/dataset_filler_v2.csv    (FIXED -- was dataset_filler.csv)
     data/processed/dataset_reparandum.csv   (existing, unchanged)
out: data/processed/dataset_filler_multillm.csv
     data/processed/dataset_reparandum_multillm.csv
       -- original columns + prev_<tag>_surp / foll_<tag>_surp for every
          model whose surprisal_<tag>.csv is present

Nitin, Sept 2026 (multi-LLM extension); fixed Oct 2026
"""

import os
import pandas as pd
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

FILLER_IN  = os.path.join(PROCESSED, "dataset_filler_v2.csv")   # FIXED: was dataset_filler.csv
REP_IN     = os.path.join(PROCESSED, "dataset_reparandum.csv")
FILLER_OUT = os.path.join(PROCESSED, "dataset_filler_multillm.csv")
REP_OUT    = os.path.join(PROCESSED, "dataset_reparandum_multillm.csv")

# same tag list as 17_multi_llm_surprisal.py's MODELS registry -- kept as
# a plain list here (not importing the other script) so this still works
# if only some models finished running
KNOWN_TAGS = ["gpt2", "dgpt2", "gpt2m", "pyt160", "opt125"]


def propagate_feature(dataset_df, feature_df, prev_col, foll_col):
    """
    Conversation-order ffill/bfill, for datasets that DO carry
    conv_id/speaker/start (reparandum). A disfluent word has no surprisal
    of its own in the per-word feature file (computed on clean,
    disfluency-stripped text), so it inherits the value from the nearest
    preceding clean word.
    """
    merged = dataset_df.merge(
        feature_df[["word_id", prev_col, foll_col]],
        on="word_id", how="left"
    )
    merged = merged.sort_values(["conv_id", "speaker", "start"])
    merged[prev_col] = merged.groupby(["conv_id", "speaker"])[prev_col].ffill()
    merged[foll_col] = merged.groupby(["conv_id", "speaker"])[foll_col].bfill()
    return merged.sort_index()


def merge_filler_via_neighbor_ids(dataset_df, feature_df, tag):
    """
    dataset_filler_v2.csv has no conv_id/speaker/start, so it can't use
    propagate_feature()'s ffill/bfill. It has TWO kinds of rows, which
    need two different lookups (confirmed via inspection: the split is
    exact -- disfluency==1 rows all have neighbor IDs, disfluency==0 rows
    never do):

      disfluency==1 (actual filler occurrence): prev_fluent_word_id /
        foll_fluent_word_id name the surrounding fluent words. Look up
        the PRECEDING word's row and read both prev_<tag>_surp and
        foll_<tag>_surp from it (same rule as script 15).

      disfluency==0 (balanced fluent/negative example): the row's own
        word_id IS the fluent word in question -- there's no "preceding
        fluent word to a filler" concept here. Look up this word's own
        prev_<tag>_surp / foll_<tag>_surp directly by word_id, same as
        every other feature column in this row was computed.

    Without this split, all disfluency==0 rows (50% of the file) were
    silently dropped at the GLMM completeness-check stage -- the earlier
    version of this function only handled the disfluency==1 case.
    """
    prev_col, foll_col = f"prev_{tag}_surp", f"foll_{tag}_surp"
    lookup = feature_df.set_index("word_id")

    is_filler = dataset_df["disfluency"] == 1

    prev_result = pd.Series(np.nan, index=dataset_df.index)
    foll_result = pd.Series(np.nan, index=dataset_df.index)

    # disfluency == 1: BOTH values come from the row of the PRECEDING fluent word.
    # In surprisal_<tag>.csv, row i holds surp(word i) in prev_<tag>_surp and
    # surp(word i+1) in foll_<tag>_surp (checked: foll[i] == prev[i+1] in every row).
    # So the preceding word's row already gives the surprisal of the word that follows
    # it. Looking up the FOLLOWING word's own foll value (what this function did
    # before) is off by one word.
    prev_result[is_filler] = dataset_df.loc[is_filler, "prev_fluent_word_id"].map(lookup[prev_col])
    foll_result[is_filler] = dataset_df.loc[is_filler, "prev_fluent_word_id"].map(lookup[foll_col])

    # disfluency == 0: look up the row's OWN word_id directly
    prev_result[~is_filler] = dataset_df.loc[~is_filler, "word_id"].map(lookup[prev_col])
    foll_result[~is_filler] = dataset_df.loc[~is_filler, "word_id"].map(lookup[foll_col])

    dataset_df[prev_col] = prev_result.values
    dataset_df[foll_col] = foll_result.values
    return dataset_df


def merge_all_models(dataset_path, label, is_filler):
    print(f"\n{'='*55}\nMerging multi-LLM surprisal into {label}\n{'='*55}")

    df = pd.read_csv(dataset_path)
    print(f"  Loaded {dataset_path}: {len(df):,} rows")

    n_merged = 0
    for tag in KNOWN_TAGS:
        feat_path = os.path.join(PROCESSED, f"surprisal_{tag}.csv")
        if not os.path.exists(feat_path):
            print(f"  [skip] surprisal_{tag}.csv not found -- run stage 1 for this model first")
            continue

        feat_df = pd.read_csv(feat_path)
        prev_col, foll_col = f"prev_{tag}_surp", f"foll_{tag}_surp"

        if is_filler:
            df = merge_filler_via_neighbor_ids(df, feat_df, tag)
        else:
            df = propagate_feature(df, feat_df, prev_col, foll_col)

        n_prev_valid = df[prev_col].notna().sum()
        n_foll_valid = df[foll_col].notna().sum()
        print(f"  [{tag}] merged. prev valid: {n_prev_valid:,}/{len(df):,}  "
              f"foll valid: {n_foll_valid:,}/{len(df):,}")
        n_merged += 1

    print(f"\n  {n_merged} model(s) merged into {label}")
    return df


if __name__ == "__main__":
    print("=" * 55)
    print("Multi-LLM Surprisal Extension -- Stage 2: Merge")
    print("=" * 55)

    if not (os.path.exists(FILLER_IN) and os.path.exists(REP_IN)):
        raise FileNotFoundError(
            "dataset_filler_v2.csv / dataset_reparandum.csv not found -- "
            "run 05_build_dataset_v2.py (filler) and 05_build_dataset.py "
            "(reparandum) first, same prerequisite as GPT-2 stage 2"
        )

    filler_df = merge_all_models(FILLER_IN, "dataset_filler_v2", is_filler=True)
    filler_df.to_csv(FILLER_OUT, index=False)
    print(f"\nSaved -> {FILLER_OUT}")

    rep_df = merge_all_models(REP_IN, "dataset_reparandum", is_filler=False)
    rep_df.to_csv(REP_OUT, index=False)
    print(f"Saved -> {REP_OUT}")

    print("\n" + "=" * 55)
    print("Stage 2 COMPLETE")
    print("=" * 55)
    print("\nNext: Rscript src/19_run_glmm_multi_llm.R")