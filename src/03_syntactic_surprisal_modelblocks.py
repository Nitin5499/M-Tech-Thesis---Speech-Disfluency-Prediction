"""
Import ModelBlocks syntactic surprisal and attach it to word_ids.

Using clean_word_index.csv here (fillers/reparandums already stripped out),
sentence boundaries come from the Switchboard markers, not just split on
punctuation. Alignment back to word_id is done via (sent_idx, word_pos).

Run modes:
  import -> reads tokmeasures, maps surprisal to word_ids, writes csv
  check  -> just checks the index and tokmeasures line up before I waste
            an hour debugging a merge

03a_clean_corpus.py and 03b_prepare_modelblocks.py need to have run before
this (they build clean_sentences.txt / .delim.linetoks). No separate export
step needed anymore.

in:  data/modelblocks/clean_word_index.csv
     data/modelblocks/clean_sentences.tokmeasures
out: data/processed/syntactic_surprisal.csv  (prev/foll surprisal per word_id)

Nitin, May 2026
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
MB_DIR    = os.path.join(ROOT_DIR, "data", "modelblocks")

CLEAN_INDEX = os.path.join(MB_DIR, "clean_word_index.csv")
TOKMEASURES = os.path.join(MB_DIR, "clean_sentences.tokmeasures")
OUTPUT_CSV  = os.path.join(PROCESSED, "syntactic_surprisal.csv")
LOG_FILE    = os.path.join(PROCESSED, "pipeline_log.txt")


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


def _isnan(val):
    # float('nan') check that doesn't blow up on non-floats
    try:
        return math.isnan(val)
    except (TypeError, ValueError):
        return True


def import_tokmeasures():
    log_section("03c: importing ModelBlocks surprisal")

    log("loading word index...")
    if not os.path.exists(CLEAN_INDEX):
        raise FileNotFoundError(f"{CLEAN_INDEX} not found -- run 03b first")
    index_df = pd.read_csv(CLEAN_INDEX)
    log(f"  words: {len(index_df):,}, sentences: {index_df['sent_idx'].nunique():,}")

    log("loading tokmeasures...")
    if not os.path.exists(TOKMEASURES):
        raise FileNotFoundError(
            f"{TOKMEASURES} not found -- run synproc first, then:\n"
            "  cat clean_sentences.tokdecs | awk '{print $1, $NF}' > clean_sentences.tokmeasures"
        )

    # walk the tokmeasures file line by line, resetting word_pos at every
    # !ARTICLE marker (that's how ModelBlocks marks sentence starts)
    mb_rows = []
    sent_idx = -1
    word_pos = 0
    header_skipped = False
    n_fail = 0
    n_valid = 0

    with open(TOKMEASURES, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_skipped:
                if line.lower().startswith("word"):
                    header_skipped = True
                    continue
            if line.startswith("!ARTICLE"):
                sent_idx += 1
                word_pos = 0
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0]
            try:
                totsurp = float(parts[1])
                if math.isnan(totsurp) or math.isinf(totsurp):
                    totsurp = np.nan
                    n_fail += 1
                else:
                    n_valid += 1
            except (ValueError, OverflowError):
                totsurp = np.nan
                n_fail += 1

            mb_rows.append({
                "sent_idx": sent_idx,
                "word_pos": word_pos,
                "mb_word": word,
                "totsurp": totsurp,
            })
            word_pos += 1

    mb_df = pd.DataFrame(mb_rows)
    log(f"  read {len(mb_df):,} words, {mb_df['sent_idx'].nunique():,} sentences")
    log(f"  valid surprisal: {n_valid:,}, failed/nan/inf: {n_fail:,} "
        f"({n_valid/max(len(mb_df),1)*100:.1f}% coverage)")

    # merge on (sent_idx, word_pos) -- this is where things usually break if
    # the sentence splitting between the two files doesn't match exactly
    log("merging onto word_ids...")
    merged = index_df.merge(mb_df, on=["sent_idx", "word_pos"], how="left")

    mismatches = merged[
        merged["mb_word"].notna() &
        (merged["word"].str.lower() != merged["mb_word"].str.lower())
    ]
    if len(mismatches) == 0:
        log("  words line up, 0 mismatches")
    else:
        log(f"  WARNING: {len(mismatches):,} mismatches, check alignment")
        print(mismatches[["word_id", "word", "mb_word", "totsurp"]].head(5).to_string())

    log(f"  valid: {merged['totsurp'].notna().sum():,}, missing: {merged['totsurp'].isna().sum():,}")

    # build prev/foll surprisal per word -- foll is just the next word's
    # surprisal within the same sentence, nan for the last word
    log("building prev_syn_surp / foll_syn_surp...")
    results = []
    for _, group in merged.groupby("sent_idx"):
        group = group.sort_values("word_pos").reset_index(drop=True)
        n = len(group)
        surps = group["totsurp"].tolist()
        ids = group["word_id"].tolist()

        for i in range(n):
            prev_s = surps[i]
            foll_s = surps[i + 1] if i + 1 < n else np.nan
            results.append({
                "word_id": ids[i],
                "prev_syn_surp": round(prev_s, 6) if not _isnan(prev_s) else np.nan,
                "foll_syn_surp": round(foll_s, 6) if not _isnan(foll_s) else np.nan,
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    log_section("done")
    log(f"saved -> {OUTPUT_CSV}")
    log(f"  rows: {len(result_df):,}, prev ok: {result_df['prev_syn_surp'].notna().sum():,}, "
        f"foll ok: {result_df['foll_syn_surp'].notna().sum():,}")
    log(f"  coverage: {result_df['prev_syn_surp'].notna().sum()/len(result_df)*100:.1f}%")
    log("next: python src/05_build_dataset.py")


def check_alignment():
    log_section("check: index vs tokmeasures")

    index_df = pd.read_csv(CLEAN_INDEX)
    log(f"index: {len(index_df):,} words, {index_df['sent_idx'].nunique():,} sentences")

    if not os.path.exists(TOKMEASURES):
        log("tokmeasures not found -- synproc hasn't been run yet")
        return

    with open(TOKMEASURES) as f:
        lines = f.readlines()
    articles = sum(1 for l in lines if '!ARTICLE' in l)
    words = sum(1 for l in lines if l.strip() and '!ARTICLE' not in l
                and not l.lower().startswith('word'))
    log(f"tokmeasures: {words:,} word lines, {articles:,} !ARTICLE markers")

    log("first sentence, for a sanity check:")
    sent0 = index_df[index_df['sent_idx'] == 0]
    print(sent0[['word_pos', 'word_id', 'word']].to_string())


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "import"

    if mode == "import":
        import_tokmeasures()
    elif mode == "check":
        check_alignment()
    else:
        print("usage: python src/03_syntactic_surprisal_modelblocks.py [import|check]")