"""
03b_parallel_modelblocks.py
============================
Splits clean_sentences.delim.linetoks into N chunks
and generates the WSL commands to run them in parallel.

Usage:
  python src/03b_parallel_modelblocks.py

This script:
  1. Splits the delim file into N equal chunks
  2. Saves each chunk to data/modelblocks/chunks/
  3. Prints the exact WSL commands to run all chunks in parallel

Author: Nitin Kumar
Date: May 2026
"""

import os
import math
import pandas as pd
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────
N_CHUNKS = 10  # number of parallel chunks

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
MB_DIR     = os.path.join(ROOT_DIR, "data", "modelblocks")
DELIM_FILE = os.path.join(MB_DIR, "clean_sentences.delim.linetoks")
CHUNKS_DIR = os.path.join(MB_DIR, "chunks")
LOG_FILE   = os.path.join(ROOT_DIR, "data", "processed", "pipeline_log.txt")

WORKSPACE  = "/mnt/d/M-Tech_in_DataEngineering/Major Project/M-Tech Thesis/M-Tech-Thesis---Speech-Disfluency-Prediction/Modelbock release master/modelblocks-release-master/workspace"
MODEL      = "genmodel/model/wsj02to21.nodashtags.4sm.synprocmodel"
CHUNKS_WSL = "/mnt/d/M-Tech_in_DataEngineering/Major Project/M-Tech Thesis/M-Tech-Thesis---Speech-Disfluency-Prediction/data/modelblocks/chunks"

os.makedirs(CHUNKS_DIR, exist_ok=True)

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

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    log_section("Step 03b: Parallel ModelBlocks Preparation")
    log(f"Splitting into {N_CHUNKS} chunks for parallel processing")

    # Read delim file — pairs of lines (!ARTICLE + sentence)
    log(f"\nReading {DELIM_FILE}...")
    pairs = []
    with open(DELIM_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Group into (article, sentence) pairs
    i = 0
    while i < len(lines):
        if lines[i].strip() == '!ARTICLE' and i+1 < len(lines):
            pairs.append((lines[i], lines[i+1]))
            i += 2
        else:
            i += 1

    total_sents = len(pairs)
    log(f"  Total sentence pairs: {total_sents:,}")

    # Split into chunks
    chunk_size = math.ceil(total_sents / N_CHUNKS)
    log(f"  Chunk size: ~{chunk_size:,} sentences each")

    chunk_files = []
    for ci in range(N_CHUNKS):
        start = ci * chunk_size
        end   = min(start + chunk_size, total_sents)
        if start >= total_sents:
            break

        chunk_pairs = pairs[start:end]
        chunk_file  = os.path.join(CHUNKS_DIR, f"chunk_{ci:02d}.delim.linetoks")

        with open(chunk_file, 'w', encoding='utf-8', newline='\n') as f:
            for article, sentence in chunk_pairs:
                f.write(article)
                f.write(sentence)

        chunk_files.append((ci, chunk_file, len(chunk_pairs)))
        log(f"  Chunk {ci:02d}: {len(chunk_pairs):,} sentences -> {os.path.basename(chunk_file)}")

    log(f"\nCreated {len(chunk_files)} chunk files in {CHUNKS_DIR}")

    # Generate WSL run script
    log("\n" + "="*60)
    log("WSL COMMANDS TO RUN IN PARALLEL")
    log("="*60)
    log("Open separate WSL terminals and run one command per terminal:")
    log("(Or run all with & for background execution)")
    log("")

    print("\n" + "="*60)
    print("COPY THESE COMMANDS INTO WSL")
    print("="*60)
    print(f"\n# Navigate to workspace first:")
    print(f"cd \"{WORKSPACE}\"")
    print()

    for ci, chunk_file, n_sents in chunk_files:
        chunk_wsl = f"{CHUNKS_WSL}/chunk_{ci:02d}.delim.linetoks"
        out_file  = f"genmodel/testing/chunk_{ci:02d}.tokdecs"
        log_file  = f"genmodel/testing/chunk_{ci:02d}.tokdecs.log"
        print(f"# Chunk {ci:02d} — {n_sents:,} sentences")
        print(f"cat \"{chunk_wsl}\" | bin/synproc -p2 -c -b5000 {MODEL} > {out_file} 2> {log_file} &")
        print()

    print("# Wait for all to finish:")
    print("wait")
    print()
    print("# Check all output files:")
    for ci, _, _ in chunk_files:
        print(f"wc -l genmodel/testing/chunk_{ci:02d}.tokdecs")

    log("\n" + "="*60)
    log("NOTE: -p2 per chunk (2 workers each) x 10 chunks = 20 total workers")
    log("This is equivalent to -p20 but split across chunks")
    log("Expected time: 1-2 hours instead of 6-8 hours")
    log("="*60)

    # Also save commands to a file
    cmd_file = os.path.join(MB_DIR, "run_parallel.sh")
    with open(cmd_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Parallel ModelBlocks synproc — {N_CHUNKS} chunks\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"cd \"{WORKSPACE}\"\n\n")
        for ci, chunk_file, n_sents in chunk_files:
            chunk_wsl = f"{CHUNKS_WSL}/chunk_{ci:02d}.delim.linetoks"
            out_file  = f"genmodel/testing/chunk_{ci:02d}.tokdecs"
            log_file  = f"genmodel/testing/chunk_{ci:02d}.tokdecs.log"
            f.write(f"# Chunk {ci:02d} — {n_sents:,} sentences\n")
            f.write(f"cat \"{chunk_wsl}\" | bin/synproc -p2 -c -b5000 {MODEL} > {out_file} 2> {log_file} &\n\n")
        f.write("wait\n")
        f.write("echo 'All chunks complete'\n")

    log(f"\nSaved run script -> {cmd_file}")
    log("You can also run: bash data/modelblocks/run_parallel.sh")
    log_section("Step 03b COMPLETE")
    log("Next: Run the WSL commands above")