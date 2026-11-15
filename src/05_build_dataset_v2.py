"""
05_build_dataset_v2.py

Fixing the filler task dataset. What was wrong: dataset_filler.csv used the
filler word itself (uh, um, ...) as the classified unit. There are only 9
distinct filler forms in the whole corpus, so the (1|word) random effect in
the GLMM could just memorize which word forms are fillers -- accuracy came
out at 99.95%, vs 81.21% in the paper. That's not a real result, the model
is cheating off word identity.

Paper section 3 says the features for each class come from the fluent words
immediately before/after the target -- disfluencies like "uh" don't carry
the same linguistic features as fluent words. So the filler token itself is
never actually a modeled data point. Each filler occurrence = one row, but
the row's features come from the nearest fluent word before and after it,
not from the filler.

What this script does (doesn't touch anything from v1):
  - for every filler occurrence in words_labeled.csv, walk outward to find
    the nearest FLUENT word before and after it (skip over other
    disfluencies in between -- "immediately precede/follow" = nearest
    fluent word, that's my reading of the paper's wording)
  - build one row per filler occurrence: prev_* features come from the
    preceding fluent word's own prev_* columns, foll_* from the following
    fluent word's own foll_* columns (borrowing already-computed features,
    not recomputing anything)
  - word_for_model (the random effect) also comes from the neighboring
    fluent word, not the filler token -- since the filler is never really
    a modeled point per the paper, this seemed like the right call, but
    it's a design choice and worth flagging if results look off
  - rows with no fluent neighbor on one side just get dropped, same as v1

in:  data/processed/words_features.csv (from 05_build_dataset.py)
     data/processed/words_labeled.csv
out: data/processed/dataset_filler_v2.csv
     data/processed/build_dataset_v2_log.txt

original dataset_filler.csv / words_features.csv untouched, keeping both
around so I can compare.

Nitin, Aug 2026
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

FEATURES_CSV  = os.path.join(PROCESSED, "words_features.csv")
LABELED_CSV   = os.path.join(PROCESSED, "words_labeled.csv")
OUT_FILLER_V2 = os.path.join(PROCESSED, "dataset_filler_v2.csv")
LOG_FILE      = os.path.join(PROCESSED, "build_dataset_v2_log.txt")

FEATURE_COLS = [
    "prev_lex_surp", "foll_lex_surp",
    "prev_syn_surp", "foll_syn_surp",
    "prev_IC",       "foll_IC",
    "prev_SC",       "foll_SC",
    "prev_dur",      "foll_dur",
]


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


def main():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("build_dataset_v2 (filler fix) run log\n")
        f.write(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    log_section("filler dataset v2 -- paper-matched methodology")
    log("positive class = surrounding fluent words, not the filler token")
    log("dataset_filler.csv / words_features.csv not touched")

    log("loading words_features.csv...")
    feat = pd.read_csv(FEATURES_CSV)
    log(f"  {len(feat):,} rows")

    log("loading words_labeled.csv...")
    lab = pd.read_csv(LABELED_CSV)
    log(f"  {len(lab):,} rows")

    # features file already has 'label' from v1 but just double check
    if "label" not in feat.columns:
        feat = feat.merge(lab[["word_id", "label"]], on="word_id", how="left")

    feat = feat.sort_values(["conv_id", "speaker", "start"]).reset_index(drop=True)

    log("scanning conversations for fillers + nearest fluent neighbors "
        "(skipping over other disfluencies in between)...")

    new_rows = []
    n_fillers_total = 0
    n_used = 0
    n_skipped_no_prev = 0
    n_skipped_no_foll = 0

    for (conv_id, speaker), group in feat.groupby(["conv_id", "speaker"], sort=False):
        group = group.reset_index(drop=True)
        labels = group["label"].tolist()
        n = len(group)

        for i in range(n):
            if labels[i] != "filler":
                continue
            n_fillers_total += 1

            # walk back to nearest fluent word
            j = i - 1
            while j >= 0 and labels[j] != "fluent":
                j -= 1
            if j < 0:
                n_skipped_no_prev += 1
                continue
            prev_fluent = group.iloc[j]

            # walk forward to nearest fluent word
            k = i + 1
            while k < n and labels[k] != "fluent":
                k += 1
            if k >= n:
                n_skipped_no_foll += 1
                continue
            foll_fluent = group.iloc[k]

            filler_row = group.iloc[i]

            # word_id kept just for traceability -- actual features below
            # come from the neighboring fluent words, not this row
            new_row = {
                "word_id": filler_row["word_id"],
                "filler_word": filler_row["word"],
                "prev_fluent_word_id": prev_fluent["word_id"],
                "prev_fluent_word": prev_fluent["word"],
                "foll_fluent_word_id": foll_fluent["word_id"],
                "foll_fluent_word": foll_fluent["word"],
                # random effect uses the preceding fluent word's identity
                # since the filler itself was never meant to be a modeled
                # word -- open to revisiting if this doesn't hold up
                "word_for_model": prev_fluent.get("word_for_model", prev_fluent["word"]),
                "pos": prev_fluent["pos"],
            }
            for col in FEATURE_COLS:
                new_row[col] = prev_fluent[col] if col.startswith("prev_") else foll_fluent[col]

            new_rows.append(new_row)
            n_used += 1

    log(f"  filler occurrences found: {n_fillers_total:,}")
    log(f"  used (both neighbors found): {n_used:,}")
    log(f"  skipped, no fluent word before: {n_skipped_no_prev:,}")
    log(f"  skipped, no fluent word after: {n_skipped_no_foll:,}")

    filler_positive = pd.DataFrame(new_rows)
    filler_positive = filler_positive.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    log(f"  after dropping rows missing features: {len(filler_positive):,}")

    # negative sample -- plain fluent words, same size as the positive set
    log("sampling matching fluent rows for the negative class...")
    fluent_pool = feat[feat["label"] == "fluent"].dropna(subset=FEATURE_COLS).reset_index(drop=True)
    log(f"  fluent pool available: {len(fluent_pool):,}")

    n_pos = len(filler_positive)
    fluent_sample = fluent_pool.sample(n=min(n_pos, len(fluent_pool)), random_state=42).reset_index(drop=True)

    fluent_sample_out = fluent_sample[["word_id", "word", "pos"] + FEATURE_COLS].copy()
    fluent_sample_out = fluent_sample_out.rename(columns={"word": "filler_word"})  # just so columns line up for concat
    fluent_sample_out["prev_fluent_word_id"] = np.nan
    fluent_sample_out["prev_fluent_word"] = np.nan
    fluent_sample_out["foll_fluent_word_id"] = np.nan
    fluent_sample_out["foll_fluent_word"] = np.nan
    fluent_sample_out["word_for_model"] = fluent_sample["word_for_model"] if "word_for_model" in fluent_sample.columns else fluent_sample["word"]
    fluent_sample_out["disfluency"] = 0

    filler_positive["disfluency"] = 1

    cols_order = ["word_id", "filler_word", "prev_fluent_word_id", "prev_fluent_word",
                  "foll_fluent_word_id", "foll_fluent_word", "word_for_model", "pos",
                  "disfluency"] + FEATURE_COLS

    filler_positive = filler_positive[cols_order]
    fluent_sample_out = fluent_sample_out[cols_order]

    final = pd.concat([filler_positive, fluent_sample_out], ignore_index=True)
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)

    final.to_csv(OUT_FILLER_V2, index=False)

    log_section("dataset_filler_v2.csv done")
    log(f"saved -> {OUT_FILLER_V2}")
    log(f"  total rows: {len(final):,} ({n_pos} filler-context + {len(fluent_sample_out)} fluent)")
    log("  original dataset_filler.csv left unchanged for comparison")
    log("next: run the GLMM on this and compare against 99.95% (old) and 81.21% (paper)")


if __name__ == "__main__":
    main()