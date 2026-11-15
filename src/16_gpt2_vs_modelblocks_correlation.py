"""
16_gpt2_vs_modelblocks_correlation.py

Checks whether GPT-2 surprisal and ModelBlocks syntactic surprisal capture the
same signal. Both prev_* columns are "this word's own surprisal given what came
before it". Neither file is propagated: unscored words are genuine NaN, so an
inner join on word_id with NaNs dropped gives exactly the words where both
models produced a real, independent estimate.

Uses the CORRECTED (content-aligned) syntactic_surprisal.csv.

in:  data/processed/syntactic_surprisal.csv  (word_id, prev_syn_surp, foll_syn_surp)
     data/processed/surprisal_gpt2.csv       (word_id + prev_* / foll_* surprisal columns, from 17_multi_llm_surprisal.py)
out: data/processed/gpt2_modelblocks_overlap_aligned.csv  (new name; old overlap file untouched)

run from repo root:  python src/16_gpt2_vs_modelblocks_correlation.py
"""

import os
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

SYN_CSV = os.path.join(PROCESSED, "syntactic_surprisal.csv")
GPT2_CSV = os.path.join(PROCESSED, "surprisal_gpt2.csv")
OUT_CSV = os.path.join(PROCESSED, "gpt2_modelblocks_overlap_aligned.csv")


def load(path, col):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if "word_id" not in header or col not in header:
        raise SystemExit("%s needs columns word_id and %s, but has: %s" % (path, col, header))
    return pd.read_csv(path, usecols=["word_id", col])


def run_correlation(col_a, col_b, label):
    syn_df = load(SYN_CSV, col_a)
    gpt2_df = load(GPT2_CSV, col_b)

    merged = syn_df.merge(gpt2_df, on="word_id", how="inner")
    n_before = len(merged)
    merged = merged.dropna(subset=[col_a, col_b])
    n_after = len(merged)

    print(f"\n--- {label} ---")
    print(f"words present in both files: {n_before:,}")
    print(f"words with a real (non-NaN) value in both: {n_after:,} "
          f"({100 * n_after / max(n_before, 1):.1f}% of the overlap)")

    if n_after < 2:
        print("not enough overlapping data to correlate")
        return merged, None, None

    pearson_r, pearson_p = stats.pearsonr(merged[col_a], merged[col_b])
    spearman_r, spearman_p = stats.spearmanr(merged[col_a], merged[col_b])

    print(f"Pearson r  = {pearson_r:.4f}  (p = {pearson_p:.3e})")
    print(f"Spearman r = {spearman_r:.4f}  (p = {spearman_p:.3e})")
    print(f"ModelBlocks {col_a}: mean={merged[col_a].mean():.3f}, std={merged[col_a].std():.3f}")
    print(f"GPT-2       {col_b}: mean={merged[col_b].mean():.3f}, std={merged[col_b].std():.3f}")

    return merged, pearson_r, spearman_r


def find_col(path, kind):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    hits = [c for c in cols if c.lower().startswith(kind) and "surp" in c.lower()]
    if len(hits) != 1:
        raise SystemExit("could not pick one %s column in %s, columns are: %s" % (kind, path, cols))
    print("using %s column: %s" % (kind, hits[0]))
    return hits[0]


if __name__ == "__main__":
    PREV_G = find_col(GPT2_CSV, "prev")
    FOLL_G = find_col(GPT2_CSV, "foll")
    prev_merged, prev_r, prev_s = run_correlation(
        "prev_syn_surp", PREV_G, "prev_* (word's own surprisal)")
    foll_merged, foll_r, foll_s = run_correlation(
        "foll_syn_surp", FOLL_G, "foll_* (next word's surprisal)")

    prev_merged.to_csv(OUT_CSV, index=False)
    print(f"\nsaved overlap table (prev_*) -> {OUT_CSV}")

    if prev_r is not None:
        if abs(prev_r) > 0.5:
            print("\n-> fairly strong correlation: GPT-2 largely re-captures ModelBlocks' signal")
        elif abs(prev_r) > 0.2:
            print("\n-> moderate correlation: some shared signal, plus real independent variance")
        else:
            print("\n-> weak correlation: GPT-2 carries information ModelBlocks does not")
        print("(interpret with the filler-task gain from the rerun; this script only measures overlap)")