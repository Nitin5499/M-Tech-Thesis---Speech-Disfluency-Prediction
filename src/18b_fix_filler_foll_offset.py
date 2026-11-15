"""
test_filler_foll_offset.py

Checks one suspicion in the filler dataset (05_build_dataset_v2.py).

For a filler between fluent words A and B ("A uh B"), the foll_* features
should describe B. 05_build_dataset_v2.py takes foll_* from B's own row,
but a row's foll_* holds the NEXT word's value, so that gives the word AFTER B.
18_merge_multi_llm_FIXED_v2.py already fixed this for the GPT-2 columns (it reads
foll_* from A's row, where foll_* = B), so baseline and GPT-2 features currently
describe different words.

This script:
 1. checks on words_features.csv that foll_X[i] really equals prev_X[i+1]
    (if it does not, the suspicion is wrong and we stop there)
 2. takes the existing dataset_filler_multillm.csv and, for filler rows only, replaces
    foll_* with B's own value (the same rows, nothing else changes)
 3. writes data/processed/dataset_filler_fixfoll_multillm.csv

Nothing existing is touched. Then run the R test on that file:
    "C:\\Program Files\\R\\R-4.5.3\\bin\\Rscript.exe" src/test_filler_foll_offset.R

run from repo root:  python src/test_filler_foll_offset.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
FEATURES_CSV = os.path.join(PROC, "words_features.csv")
OUT_CSV = os.path.join(PROC, "dataset_filler_fixfoll_multillm.csv")

PAIRS = [("prev_lex_surp", "foll_lex_surp"), ("prev_syn_surp", "foll_syn_surp"),
         ("prev_IC", "foll_IC"), ("prev_SC", "foll_SC"), ("prev_dur", "foll_dur")]
FEATURE_COLS = [c for p in PAIRS for c in p]
FOLL_COLS = [f for _, f in PAIRS]

print("loading words_features.csv ...")
feat = pd.read_csv(FEATURES_CSV)
feat = feat.sort_values(["conv_id", "speaker", "start"]).reset_index(drop=True)
print("  %d rows" % len(feat))

# ---- 1. is foll_X[i] == prev_X[i+1] ? ------------------------------------
same = (feat["conv_id"].values[:-1] == feat["conv_id"].values[1:]) & \
       (feat["speaker"].values[:-1] == feat["speaker"].values[1:])
both_fluent = (feat["label"].values[:-1] == "fluent") & (feat["label"].values[1:] == "fluent")
ok = same & both_fluent
print("\ncheck 1: does foll_X of a fluent word equal prev_X of the next fluent word?")
print("  (pairs of neighbouring fluent rows in the same conversation: %d)" % ok.sum())
fracs = []
for p, f in PAIRS:
    a = feat[f].values[:-1][ok]
    b = feat[p].values[1:][ok]
    m = ~(np.isnan(a) | np.isnan(b))
    frac = np.isclose(a[m], b[m], atol=1e-6).mean() if m.sum() else float("nan")
    fracs.append(frac)
    print("  %-14s -> %-14s equal in %.1f%% of %d pairs" % (f, p, 100 * frac, m.sum()))
if np.nanmean(fracs) < 0.8:
    print("\nSTOP: foll is mostly NOT the next word's prev, so my suspicion may be wrong.")
    print("Send me this output before running anything else.")
    raise SystemExit(1)

# ---- 2. keep the SAME rows as the real run, change only foll_* of filler rows --
# Take dataset_filler_multillm.csv (the file the old results came from) and replace
# each filler row's foll_* with the prev_* of the following fluent word B, i.e. B's own
# value, instead of B's foll_* (= the word after B). Negatives and GPT-2 columns stay as
# they are, so the only difference from the old run is this one-word shift.
OLD_CSV = os.path.join(PROC, "dataset_filler_multillm.csv")
print("\nloading", OLD_CSV)
df = pd.read_csv(OLD_CSV)
print("  %d rows (%d filler, %d fluent)" % (len(df), (df["disfluency"] == 1).sum(), (df["disfluency"] == 0).sum()))

look = feat.drop_duplicates("word_id").set_index("word_id")
is_f = df["disfluency"] == 1
bid = df.loc[is_f, "foll_fluent_word_id"]
changed = {}
for p, f in PAIRS:
    old_vals = df.loc[is_f, f].values.copy()
    new_vals = bid.map(look[p]).values
    df.loc[is_f, f] = new_vals
    both = ~(np.isnan(old_vals) | np.isnan(new_vals))
    changed[f] = (np.abs(old_vals[both] - new_vals[both]) > 1e-6).mean()
    print("  %-14s: filler rows whose value changed: %.1f%%, now missing: %d"
          % (f, 100 * changed[f], int(np.isnan(new_vals).sum())))

need = FEATURE_COLS + ["prev_gpt2_surp", "foll_gpt2_surp"]
print("\n  complete rows now: %d   (old file: see 37,543 in the old run log)"
      % df[need].notna().all(axis=1).sum())

df.to_csv(OUT_CSV, index=False)
print("\nsaved ->", OUT_CSV)
print("reference (old data): 45,984 rows, 37,543 complete, baseline 77.68, +GPT-2 78.54 (+0.86)")
print("next: Rscript src/test_filler_foll_offset.R")