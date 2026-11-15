"""
02_lexical_surprisal.py
=======================
Trains a trigram language model on OANC-GrAF plain text files.
Computes lexical surprisal on the CLEAN corpus only.

DC Correction Applied:
  - Computes surprisal using only clean fluent words
  - Fillers and reparandums are excluded from context and computation
  - Uses clean_word_index.csv for proper sentence boundaries

Formula: S(w_k) = -log2 P(w_k | w_{k-2}, w_{k-1})

INPUT:
  data/OANC-GrAF/                      -- language model training corpus
  data/processed/clean_word_index.csv  -- clean words with sentence boundaries

OUTPUT:
  data/processed/lexical_surprisal.csv -- prev and foll surprisal per word_id

Author: Nitin Kumar
Date: May 2026
"""

import os
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
OANC_DIR    = os.path.join(ROOT_DIR, "data", "OANC-GrAF")
CLEAN_INDEX = os.path.join(ROOT_DIR, "data", "processed", "clean_word_index.csv")
LOG_FILE    = os.path.join(ROOT_DIR, "data", "processed", "pipeline_log.txt")
OUTPUT_CSV  = os.path.join(ROOT_DIR, "data", "processed", "lexical_surprisal.csv")

SKIP_SUFFIXES = (
    ".anc", ".xml", ".hepple", ".logical",
    ".css", ".dtd", ".xsd", ".html", ".htm",
    "-hepple", "-logical", "-np", "-vp", "-s",
)

# ── Logger ─────────────────────────────────────────────────────
def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def log_section(title):
    line = "\n" + "=" * 60 + f"\n{title}\n" + "=" * 60
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ── File helpers ───────────────────────────────────────────────
def is_plain_text(fname):
    lower = fname.lower()
    for s in SKIP_SUFFIXES:
        if lower.endswith(s):
            return False
    if fname.startswith("."):
        return False
    if lower.endswith(".txt"):
        return True
    if "." not in fname:
        return True
    return False

# ── Load OANC ──────────────────────────────────────────────────
def load_oanc_text(oanc_dir):
    log(f"Loading OANC-GrAF from: {oanc_dir}")
    sentences   = []
    file_count  = 0
    token_count = 0

    for dirpath, _, filenames in os.walk(oanc_dir):
        for fname in filenames:
            if not is_plain_text(fname):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        tokens = line.strip().lower().split()
                        if tokens:
                            sentences.append(tokens)
                            token_count += len(tokens)
                file_count += 1
            except Exception:
                continue

    log(f"  Files read:   {file_count:,}")
    log(f"  Sentences:    {len(sentences):,}")
    log(f"  Total tokens: {token_count:,}")

    if not sentences:
        raise ValueError(f"No text found in {oanc_dir}")
    return sentences

# ── Train trigram ──────────────────────────────────────────────
def train_trigram(sentences):
    log("Training Laplace-smoothed trigram model...")
    unigram      = defaultdict(int)
    trigram      = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    bigram_total = defaultdict(lambda: defaultdict(int))

    for sent in sentences:
        padded = ["<s>", "<s>"] + sent + ["</s>"]
        for i in range(2, len(padded)):
            w0, w1, w2 = padded[i-2], padded[i-1], padded[i]
            unigram[w2] += 1
            trigram[w0][w1][w2] += 1
            bigram_total[w0][w1] += 1

    V = len(unigram)
    log(f"  Vocabulary size: {V:,}")

    def surprisal_fn(word, ctx2, ctx1):
        w  = str(word).lower() if word else "<unk>"
        c2 = str(ctx2).lower() if ctx2 else "<s>"
        c1 = str(ctx1).lower() if ctx1 else "<s>"
        num = trigram[c2][c1][w] + 1
        den = bigram_total[c2][c1] + V
        return -math.log2(num / den) if den > 0 else 0.0

    return surprisal_fn

# ── Compute surprisal on clean corpus ─────────────────────────
def compute_surprisal_clean(clean_idx, surprisal_fn):
    log("Computing lexical surprisal on clean corpus...")
    log("  Using sentence boundaries from clean_word_index.csv")

    results = []
    groups  = list(clean_idx.groupby('sent_idx', sort=True))
    total   = len(groups)

    for g_idx, (sent_idx, group) in enumerate(groups):
        group = group.sort_values('word_pos').reset_index(drop=True)
        n     = len(group)
        words = group['word'].astype(str).str.lower().tolist()
        ids   = group['word_id'].tolist()

        for i in range(n):
            # Use sentence boundaries — context resets at sentence start
            w_km2 = words[i-2] if i >= 2 else "<s>"
            w_km1 = words[i-1] if i >= 1 else "<s>"
            w_k   = words[i]
            w_kp1 = words[i+1] if i < n-1 else "</s>"

            prev_s = surprisal_fn(w_k,   w_km2, w_km1)
            foll_s = surprisal_fn(w_kp1, w_km1, w_k) if i < n-1 else np.nan

            results.append({
                "word_id":       ids[i],
                "prev_lex_surp": round(prev_s, 6),
                "foll_lex_surp": round(foll_s, 6) if not np.isnan(foll_s) else np.nan,
            })

        if (g_idx + 1) % 5000 == 0:
            log(f"  {g_idx+1}/{total} sentences done...")

    return pd.DataFrame(results)


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    log_section("Step 02: Lexical Surprisal (Clean Corpus)")
    log("DC Correction: Computing on clean words only")
    log("Sentence boundaries: from clean_word_index.csv")

    # Train language model on OANC
    sentences    = load_oanc_text(OANC_DIR)
    surprisal_fn = train_trigram(sentences)

    # Load clean word index
    log("\nLoading clean_word_index.csv...")
    clean_idx = pd.read_csv(CLEAN_INDEX)
    log(f"  Clean words:     {len(clean_idx):,}")
    log(f"  Clean sentences: {clean_idx['sent_idx'].nunique():,}")

    # Compute surprisal
    surp_df = compute_surprisal_clean(clean_idx, surprisal_fn)

    # Save
    surp_df.to_csv(OUTPUT_CSV, index=False)

    log_section("Step 02 COMPLETE")
    log(f"Saved -> {OUTPUT_CSV}")
    log(f"  Total rows:        {len(surp_df):,}")
    log(f"  prev_lex_surp ok:  {surp_df['prev_lex_surp'].notna().sum():,}")
    log(f"  foll_lex_surp ok:  {surp_df['foll_lex_surp'].notna().sum():,}")
    log(f"  Mean prev_lex_surp: {surp_df['prev_lex_surp'].mean():.4f}")
    log("\nSample output:")
    print(surp_df.head(8).to_string())
    log("\nNext step: python src/04_dlt_costs.py")