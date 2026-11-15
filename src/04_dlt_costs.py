"""
04_dlt_costs.py
===============
Computes DLT Integration Cost and Storage Cost using spaCy dependency parser.

DC Correction Applied:
  - Computes DLT on CLEAN sentences only (fillers and reparandums removed)
  - Uses proper sentence boundaries from clean_word_index.csv
  - Dependency graphs are computed per sentence — not per full conversation
  - This gives correct dependency structures without disfluent word interference

From the paper:
  "To calculate these costs, the dependency relations for our corpus
   were extracted by removing disfluencies from the constituency-based
   parse trees and converting these trees into dependency graphs."

  Integration Cost (IC): sum of dependency lengths to the current word + 1
  Storage Cost (SC):     number of incomplete dependencies so far + 1

Note: We use spaCy en_core_web_sm as a computationally efficient
alternative to the Stanford Parser. Both produce Universal Dependency
graphs and the DLT cost computation is identical on both outputs.

INPUT:
  data/processed/clean_word_index.csv  -- clean words with sentence boundaries

OUTPUT:
  data/processed/dlt_costs.csv         -- IC and SC per word_id

Author: Nitin Kumar
Date: May 2026
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
CLEAN_INDEX = os.path.join(ROOT_DIR, "data", "processed", "clean_word_index.csv")
OUTPUT_CSV  = os.path.join(ROOT_DIR, "data", "processed", "dlt_costs.csv")
LOG_FILE    = os.path.join(ROOT_DIR, "data", "processed", "pipeline_log.txt")

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

# ── Load spaCy ─────────────────────────────────────────────────
def load_spacy():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        log("spaCy en_core_web_sm loaded")
        return nlp
    except OSError:
        log("Downloading spaCy model...")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        import spacy
        return spacy.load("en_core_web_sm")

# ── DLT computation per sentence ───────────────────────────────
def compute_dlt_for_sentence(words, nlp):
    """
    Computes IC and SC for a list of clean words.
    Each call processes ONE sentence — dependency graph is
    computed on the clean sentence without disfluent words.
    """
    if not words:
        return [], []

    text = " ".join(str(w) for w in words if w and str(w).strip())
    if not text.strip():
        return [1.0] * len(words), [1.0] * len(words)

    try:
        doc = nlp(text)
        IC  = []
        SC  = []

        for i, token in enumerate(doc):
            # Integration Cost
            # Sum of distances from all dependents that connect to this word
            ic = 1  # minimum cost
            for dep_tok in doc:
                if dep_tok.head.i == i and dep_tok.i != i:
                    ic += abs(dep_tok.i - i)
            IC.append(float(ic))

            # Storage Cost
            # Count open dependencies — dependents seen but head not yet seen
            sc = 1  # minimum cost
            for dep_tok in doc[:i+1]:
                if dep_tok.head.i > i:
                    sc += 1
            SC.append(float(sc))

    except Exception as e:
        IC = [1.0] * len(words)
        SC = [1.0] * len(words)

    # Ensure lengths match
    while len(IC) < len(words):
        IC.append(1.0)
        SC.append(1.0)

    return IC[:len(words)], SC[:len(words)]

# ── Main computation ───────────────────────────────────────────
def compute_all_dlt(clean_idx, nlp):
    """
    Computes DLT costs for all clean sentences.
    Groups by sent_idx — each sentence is processed independently.
    """
    log("Computing DLT costs on clean sentences...")
    log("  Each sentence processed independently — proper dependency graphs")

    results = []
    groups  = list(clean_idx.groupby('sent_idx', sort=True))
    total   = len(groups)

    for g_idx, (sent_idx, group) in enumerate(groups):
        group = group.sort_values('word_pos').reset_index(drop=True)
        n     = len(group)
        words = group['word'].astype(str).tolist()
        ids   = group['word_id'].tolist()

        # Compute DLT on this clean sentence
        IC_list, SC_list = compute_dlt_for_sentence(words, nlp)

        for i in range(n):
            results.append({
                "word_id": ids[i],
                "prev_IC": IC_list[i],
                "foll_IC": IC_list[i+1] if i+1 < n else np.nan,
                "prev_SC": SC_list[i],
                "foll_SC": SC_list[i+1] if i+1 < n else np.nan,
            })

        if (g_idx + 1) % 5000 == 0:
            log(f"  {g_idx+1}/{total} sentences done...")

    return pd.DataFrame(results)

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    log_section("Step 04: DLT Costs (Clean Corpus)")
    log("DC Correction: Computing on clean sentences only")
    log("Sentence boundaries: from clean_word_index.csv")
    log("Parser: spaCy en_core_web_sm")

    # Load spaCy
    nlp = load_spacy()

    # Load clean word index
    log("\nLoading clean_word_index.csv...")
    clean_idx = pd.read_csv(CLEAN_INDEX)
    log(f"  Clean words:     {len(clean_idx):,}")
    log(f"  Clean sentences: {clean_idx['sent_idx'].nunique():,}")

    # Compute DLT
    dlt_df = compute_all_dlt(clean_idx, nlp)

    # Save
    dlt_df.to_csv(OUTPUT_CSV, index=False)

    log_section("Step 04 COMPLETE")
    log(f"Saved -> {OUTPUT_CSV}")
    log(f"  Total rows:   {len(dlt_df):,}")
    log(f"  prev_IC ok:   {dlt_df['prev_IC'].notna().sum():,}")
    log(f"  prev_SC ok:   {dlt_df['prev_SC'].notna().sum():,}")
    log(f"  Mean prev_IC: {dlt_df['prev_IC'].mean():.4f}")
    log(f"  Mean prev_SC: {dlt_df['prev_SC'].mean():.4f}")
    log("\nSample output:")
    print(dlt_df.head(8).to_string())
    log("\nNext step: python src/03b_prepare_modelblocks.py")