"""
03b_prepare_modelblocks.py
==========================
Prepares the clean corpus for ModelBlocks synproc.

Takes clean_sentences.txt and creates the delim format
that ModelBlocks expects — one !ARTICLE marker per sentence.

Also saves the word index for alignment after synproc runs.

INPUT:
  data/processed/clean_sentences.txt   -- clean sentences one per line
  data/processed/clean_word_index.csv  -- word_id to sentence mapping

OUTPUT:
  data/modelblocks/clean_sentences.delim.linetoks  -- ModelBlocks input
  data/modelblocks/clean_word_index.csv            -- copy of index for MB

Author: Nitin Kumar
Date: May 2026
"""

import os
import shutil
import pandas as pd
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
CLEAN_SENTS = os.path.join(ROOT_DIR, "data", "processed", "clean_sentences.txt")
CLEAN_INDEX = os.path.join(ROOT_DIR, "data", "processed", "clean_word_index.csv")
MB_DIR      = os.path.join(ROOT_DIR, "data", "modelblocks")
OUT_DELIM   = os.path.join(MB_DIR, "clean_sentences.delim.linetoks")
OUT_INDEX   = os.path.join(MB_DIR, "clean_word_index.csv")
LOG_FILE    = os.path.join(ROOT_DIR, "data", "processed", "pipeline_log.txt")

os.makedirs(MB_DIR, exist_ok=True)

# ── Logger ─────────────────────────────────────────────────────
def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def log_section(title):
    line = "\n" + "=" * 60 + f"\n{title}\n" + "=" * 60
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    log_section("Step 03b: Prepare ModelBlocks Input")
    log("Input: clean_sentences.txt")
    log("Output: clean_sentences.delim.linetoks")

    # Load clean sentences
    log("\nLoading clean_sentences.txt...")
    with open(CLEAN_SENTS, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    log(f"  Loaded {len(sentences):,} clean sentences")

    # Verify word index
    log("\nLoading clean_word_index.csv...")
    idx = pd.read_csv(CLEAN_INDEX)
    log(f"  Words in index: {len(idx):,}")
    log(f"  Sentences in index: {idx['sent_idx'].nunique():,}")

    # Verify counts match
    if len(sentences) != idx['sent_idx'].nunique():
        log(f"  WARNING: sentence count mismatch!")
        log(f"  sentences.txt: {len(sentences):,}")
        log(f"  index:         {idx['sent_idx'].nunique():,}")
    else:
        log(f"  OK Sentence counts match: {len(sentences):,}")

    # Create delim file
    log("\nCreating delim file with !ARTICLE markers...")
    with open(OUT_DELIM, 'w', encoding='utf-8', newline='\n') as f:
        for sent in sentences:
            f.write('!ARTICLE\n')
            f.write(sent + '\n')

    # Verify delim file
    with open(OUT_DELIM, 'r') as f:
        line_count = sum(1 for _ in f)
    expected = len(sentences) * 2
    log(f"  Lines in delim file: {line_count:,} (expected {expected:,})")

    # Show sample
    log("\nSample delim file (first 8 lines):")
    with open(OUT_DELIM, 'r') as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            print(f"  {i+1}: {line.rstrip()}")

    # Copy word index to modelblocks folder
    shutil.copy(CLEAN_INDEX, OUT_INDEX)
    log(f"\nCopied word index -> {OUT_INDEX}")

    log_section("Step 03b COMPLETE")
    log(f"Output files:")
    log(f"  {OUT_DELIM}")
    log(f"  {OUT_INDEX}")
    log(f"\nTotal sentences prepared: {len(sentences):,}")
    log(f"Delim file lines: {line_count:,}")
    log("\n" + "="*60)
    log("NEXT STEP: Run ModelBlocks synproc")
    log("="*60)
    log("In WSL terminal, navigate to ModelBlocks workspace and run:")
    log("")
    log("  cat [path]/clean_sentences.delim.linetoks | \\")
    log("  bin/synproc -p10 -c -b5000 \\")
    log("  genmodel/model/wsj02to21.nodashtags.4sm.synprocmodel \\")
    log("  > genmodel/testing/clean_sentences.tokdecs \\")
    log("  2> genmodel/testing/clean_sentences.tokdecs.log")
    log("")
    log("Expected time: 6-8 hours for 74,497 sentences")
    log("Run overnight and import results tomorrow with 03c_import_mb_surprisal.py")