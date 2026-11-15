"""
align_mb_by_content.py
Re-aligns the ModelBlocks output to clean_word_index.csv by comparing the actual
words of each sentence, not just sentence number and position. This is for the
drift seen in check_mb_alignment.py (178,805 word mismatches).

It also drops the "word totsurp" header lines that each chunk file leaves in the
middle of the concatenated tokmeasures (they show up as an extra word "word").

SAFE: writes a NEW file, data/processed/syntactic_surprisal_aligned.csv and
a report. It does NOT touch syntactic_surprisal.csv.

run from repo root:  python src/align_mb_by_content.py
"""
import os
import math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB_DIR = os.path.join(ROOT, "data", "modelblocks")
PROC = os.path.join(ROOT, "data", "processed")
INDEX = os.path.join(MB_DIR, "clean_word_index.csv")
TOK = os.path.join(MB_DIR, "clean_sentences.tokmeasures")
OLD = os.path.join(PROC, "syntactic_surprisal.csv")
OUT = os.path.join(PROC, "syntactic_surprisal_aligned.csv")
REPORT = os.path.join(PROC, "mb_alignment_report.txt")
WINDOW = 60  # how many sentences to look ahead when the two files disagree

lines_out = []


def say(msg=""):
    print(msg)
    lines_out.append(msg)


# ---- ModelBlocks sentences -------------------------------------------------
mb_sents, cur, n_header = [], None, 0
with open(TOK, encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p = line.split()
        if line.startswith("!ARTICLE"):
            cur = []
            mb_sents.append(cur)
            continue
        if len(p) < 2:
            continue
        if p[0].lower() == "word" and p[1].lower().startswith(("tot", "surp")):
            n_header += 1  # chunk header, not a real word
            continue
        if cur is None:
            cur = []
            mb_sents.append(cur)
        try:
            v = float(p[1])
            if math.isnan(v) or math.isinf(v):
                v = np.nan
        except ValueError:
            v = np.nan
        cur.append((p[0].lower(), v))

# ---- index sentences -------------------------------------------------------
idx = pd.read_csv(INDEX)
idx = idx.sort_values(["sent_idx", "word_pos"])
ix_sents = []
for _, g in idx.groupby("sent_idx", sort=True):
    ix_sents.append(list(zip(g["word_id"], g["word"].astype(str).str.lower())))

say("index sentences      : %d (%d words)" % (len(ix_sents), len(idx)))
say("modelblocks sentences: %d (%d words)" % (len(mb_sents), sum(len(s) for s in mb_sents)))
say("header lines dropped : %d" % n_header)


def same(ix_s, mb_s):
    return len(ix_s) == len(mb_s) and all(a[1] == b[0] for a, b in zip(ix_s, mb_s))


# ---- walk both lists, resync when they disagree ---------------------------
i = j = 0
pairs, unmatched_ix, events = [], [], []
n_ix, n_mb = len(ix_sents), len(mb_sents)
while i < n_ix and j < n_mb:
    if same(ix_sents[i], mb_sents[j]):
        pairs.append((i, j))
        i += 1
        j += 1
        continue
    found = None
    for k in range(1, WINDOW + 1):
        if j + k < n_mb and same(ix_sents[i], mb_sents[j + k]):
            found = ("modelblocks has %d extra sentence(s)" % k, 0, k)
            break
        if i + k < n_ix and same(ix_sents[i + k], mb_sents[j]):
            found = ("modelblocks is missing %d sentence(s)" % k, k, 0)
            break
    if found is None:
        unmatched_ix.append(i)  # different text, give up on this one
        events.append((i, "no match, skipped one sentence"))
        i += 1
        j += 1
    else:
        events.append((i, found[0]))
        unmatched_ix.extend(range(i, i + found[1]))
        i += found[1]
        j += found[2]
unmatched_ix.extend(range(i, n_ix))

say("\nsentences matched by content : %d of %d (%.1f%%)"
    % (len(pairs), n_ix, 100 * len(pairs) / n_ix))
say("index sentences with no ModelBlocks counterpart: %d" % len(unmatched_ix))
say("resync events: %d  (first 15 below)" % len(events))
for e in events[:15]:
    say("  index sentence %6d : %s" % e)

# ---- build prev / foll the same way the old script did --------------------
rows = []
for a, b in pairs:
    ids = [w[0] for w in ix_sents[a]]
    sv = [w[1] for w in mb_sents[b]]
    n = len(ids)
    for k in range(n):
        rows.append((ids[k], sv[k], sv[k + 1] if k + 1 < n else np.nan))
out = pd.DataFrame(rows, columns=["word_id", "prev_syn_surp", "foll_syn_surp"])
# keep every index word, NaN where not aligned
allw = idx[["word_id"]].drop_duplicates()
out = allw.merge(out, on="word_id", how="left")
out["prev_syn_surp"] = out["prev_syn_surp"].round(6)
out["foll_syn_surp"] = out["foll_syn_surp"].round(6)
out.to_csv(OUT, index=False)

say("\nwords in index                 : %d" % len(out))
say("prev_syn_surp valid (aligned)   : %d (%.1f%%)"
    % (out["prev_syn_surp"].notna().sum(), 100 * out["prev_syn_surp"].notna().mean()))
say("foll_syn_surp valid (aligned)   : %d" % out["foll_syn_surp"].notna().sum())

# ---- compare with the file the models were built from ---------------------
if os.path.exists(OLD):
    old = pd.read_csv(OLD)
    m = old.merge(out, on="word_id", suffixes=("_old", "_new"))
    both = m["prev_syn_surp_old"].notna() & m["prev_syn_surp_new"].notna()
    diff = both & ((m["prev_syn_surp_old"] - m["prev_syn_surp_new"]).abs() > 1e-4)
    only_old = m["prev_syn_surp_old"].notna() & m["prev_syn_surp_new"].isna()
    say("\nvs current syntactic_surprisal.csv (%d valid values):" % m["prev_syn_surp_old"].notna().sum())
    say("  same value in both               : %d" % (both.sum() - diff.sum()))
    say("  DIFFERENT value (old was misaligned): %d" % diff.sum())
    say("  valid in old but not alignable now: %d" % only_old.sum())
    say("  share of old valid values that differ or vanish: %.1f%%"
        % (100 * (diff.sum() + only_old.sum()) / max(m["prev_syn_surp_old"].notna().sum(), 1)))

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines_out) + "\n")
say("\nwrote %s\nwrote %s\n(syntactic_surprisal.csv was not touched)" % (OUT, REPORT))