"""
demo_check.py
Quick pre-meeting check: do the files I want to show exist, and what do
they look like on screen? Nothing here is slow and nothing is changed.

Run from the project root:
    python src/demo_check.py     (or wherever this file is saved)
"""

import os
import pandas as pd

# project root = the folder that has data/ and results/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE

# (label, path relative to root, how many rows to show)
CHECKS = [
    ("GPT-2 word surprisal",       "data/processed/surprisal_gpt2.csv",              6),
    ("filler rows with 5 models",  "data/processed/dataset_filler_multillm.csv",     3),
    ("five-model results",         "results/table_multi_llm_comparison.csv",        10),
    ("prosody results",            "results/table_prosodic_comparison.csv",         10),
    ("prosody bridge",             "data/processed/word_prosody_bridge.csv",         3),
]

for label, rel, n in CHECKS:
    path = os.path.join(ROOT, *rel.split("/"))
    print("\n" + "=" * 60)
    print(label, "->", rel)
    if not os.path.exists(path):
        print("  MISSING: this file is not where I expected it")
        continue
    df = pd.read_csv(path)
    print(f"  rows: {len(df):,}   columns: {len(df.columns)}")
    # wide frames wrap badly on a projector, so show a few columns only
    cols = list(df.columns)[:8]
    print(df[cols].head(n).to_string(index=False))

# the one count I quote on the prosody slide
bridge = os.path.join(ROOT, "data", "processed", "word_prosody_bridge.csv")
if os.path.exists(bridge):
    b = pd.read_csv(bridge)
    if "mapping_case" in b.columns:
        print("\n" + "=" * 60)
        print("mapping_case counts (prosody slide)")
        print(b["mapping_case"].value_counts().to_string())

print("\nDone. Anything marked MISSING needs a different path before the meeting.")