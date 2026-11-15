"""
17_consolidate_results.py
============================
Builds ONE consolidated results table pulling together every feature
comparison run across this project so far -- the baseline replication
numbers, the prosodic extension, and the GPT-2 extension -- rather
than leaving them spread across three separate result files that
have to be cross-referenced manually.

This does NOT recompute anything. It only reads existing result
files already produced by earlier scripts and reshapes them into one
consistent table. If a source file is missing, that row is skipped
and logged, not silently invented.

INPUT (all existing, unmodified):
  results/table2_random_effects.csv   (reparandum baseline, if present)
  results/table2_filler_v2.csv        (filler baseline, if present)
  results/table_prosodic_comparison.csv
  results/table_gpt2_comparison.csv

OUTPUT:
  results/table_all_features_comparison.csv
    columns: task, feature_family, n_rows, acc_baseline, acc_enhanced,
             acc_diff, lrt_p, significant, notes

Author: Nitin Kumar
Date: September 2026
"""

import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

LOG_FILE = os.path.join(ROOT_DIR, "data", "processed", "pipeline_log.txt")


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def safe_read(path):
    if not os.path.exists(path):
        log(f"  [SKIP] Not found: {path}")
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    log("=" * 60)
    log("Consolidating all feature comparison results")
    log("=" * 60)

    rows = []

    # ---- Prosodic extension results (already in the right shape) ----
    prosodic_path = os.path.join(RESULTS_DIR, "table_prosodic_comparison.csv")
    prosodic_df = safe_read(prosodic_path)
    if prosodic_df is not None:
        for _, r in prosodic_df.iterrows():
            rows.append({
                "task": r.get("task", ""),
                "feature_family": f"Prosodic ({r.get('variant', '')})",
                "n_rows": r.get("n_rows", ""),
                "acc_baseline": r.get("acc_baseline", ""),
                "acc_enhanced": r.get("acc_enhanced", ""),
                "acc_diff": r.get("acc_diff", ""),
                "lrt_p": r.get("lrt_p", ""),
                "significant": "Yes" if (pd.notna(r.get("lrt_p")) and float(r.get("lrt_p", 1)) < 0.05) else "No",
                "notes": "Break: ~12% corpus coverage (annotation-limited). Accent: near-full coverage.",
            })
        log(f"  Added {len(prosodic_df)} rows from prosodic comparison")

    # ---- GPT-2 extension results ----
    gpt2_path = os.path.join(RESULTS_DIR, "table_gpt2_comparison.csv")
    gpt2_df = safe_read(gpt2_path)
    if gpt2_df is not None:
        for _, r in gpt2_df.iterrows():
            rows.append({
                "task": r.get("task", ""),
                "feature_family": "GPT-2 surprisal",
                "n_rows": r.get("n_rows", ""),
                "acc_baseline": r.get("acc_baseline", ""),
                "acc_enhanced": r.get("acc_enhanced", ""),
                "acc_diff": r.get("acc_diff", ""),
                "lrt_p": r.get("lrt_p", ""),
                "significant": "Yes" if (pd.notna(r.get("lrt_p")) and float(r.get("lrt_p", 1)) < 0.05) else "No",
                "notes": "Baseline already includes ModelBlocks syntactic surprisal (71.7% coverage).",
            })
        log(f"  Added {len(gpt2_df)} rows from GPT-2 comparison")

    if len(rows) == 0:
        log("  [WARNING] No source result files found -- nothing to consolidate.")
    else:
        out_df = pd.DataFrame(rows)
        out_path = os.path.join(RESULTS_DIR, "table_all_features_comparison.csv")
        out_df.to_csv(out_path, index=False)
        log(f"\nSaved consolidated table -> {out_path}")
        log(f"Total rows: {len(out_df)}")
        print("\n" + out_df.to_string(index=False))

    log("\nNOTE: this table covers the EXTENSION comparisons (prosodic, GPT-2),")
    log("each already comparing against the existing 10-feature baseline.")
    log("The original MTP-1 -> corrected replication numbers (reparandum")
    log("65.75 -> 71.17, filler 99.96 -> 78.66) are a separate, earlier")
    log("comparison -- against the PAPER, not a baseline-vs-enhanced test --")
    log("and are reported alongside this table in the writeup, not merged")
    log("into the same rows, since they measure a different kind of thing.")