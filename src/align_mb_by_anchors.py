"""
align_mb_by_anchors.py
Order-independent re-alignment of the ModelBlocks output to clean_word_index.csv.

Why: show_mb_mismatch_samples.py's first line says only 2,240 of the 74,497 index
sentences are missing from the ModelBlocks output, yet the in-order aligner could
only match 42,350. So the text is mostly there but not in the same order (or
shifted by chunk). This script finds, for every ModelBlocks sentence, where it
sits in the index by looking at runs of 3 consecutive sentences, so a reordered or
shifted chunk still lines up.

It writes a NEW file data/processed/syntactic_surprisal_aligned.csv (replacing
the one from align_mb_by_content.py) and a report. syntactic_surprisal.csv
is NOT touched.

run from repo root:  python src/align_mb_by_anchors.py
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
K = 3
FILLERISH = {"uh", "um", "uh-huh", "uhhuh", "hm", "mm", "huh", "er", "ah", "mmhm", "mmhmm"}

lines_out = []


def say(msg=""):
    print(msg)
    lines_out.append(msg)


# ---- read ModelBlocks sentences -------------------------------------------
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
            n_header += 1
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

# ---- read index sentences --------------------------------------------------
idx = pd.read_csv(INDEX).sort_values(["sent_idx", "word_pos"])
ix_ids, ixw = [], []
for _, g in idx.groupby("sent_idx", sort=True):
    ix_ids.append(list(g["word_id"]))
    ixw.append(tuple(g["word"].astype(str).str.lower()))
mbw = [tuple(w for w, _ in s) for s in mb_sents]
n_ix, n_mb = len(ixw), len(mbw)
say("index sentences: %d   modelblocks sentences: %d   header lines dropped: %d"
    % (n_ix, n_mb, n_header))

# ---- anchors: runs of K consecutive sentences that occur once in the index --
pos = {}
for i in range(n_ix - K + 1):
    key = tuple(ixw[i:i + K])
    pos[key] = None if key in pos else i  # None = appears twice, not usable

off = [None] * n_mb
for j in range(n_mb - K + 1):
    p = pos.get(tuple(mbw[j:j + K]))
    if p is not None:
        for t in range(K):
            off[j + t] = p - j
n_anch = sum(o is not None for o in off)
say("modelblocks sentences covered by a unique anchor: %d (%.1f%%)" % (n_anch, 100 * n_anch / n_mb))

prev_off, last = [None] * n_mb, None
for j in range(n_mb):
    if off[j] is not None:
        last = off[j]
    prev_off[j] = last
next_off, last = [None] * n_mb, None
for j in range(n_mb - 1, -1, -1):
    if off[j] is not None:
        last = off[j]
    next_off[j] = last

used = set()
assign = {}  # mb sentence j -> index sentence i
for j in range(n_mb):
    for o in (off[j], prev_off[j], next_off[j]):
        if o is None:
            continue
        i = j + o
        if 0 <= i < n_ix and i not in used and mbw[j] == ixw[i]:
            assign[j] = i
            used.add(i)
            break

say("\nmodelblocks sentences placed in the index : %d of %d (%.1f%%)"
    % (len(assign), n_mb, 100 * len(assign) / n_mb))
say("index sentences that got a ModelBlocks value : %d of %d (%.1f%%)"
    % (len(used), n_ix, 100 * len(used) / n_ix))

# ---- segments: where the offset changes -----------------------------------
segs, start = [], None
for j in sorted(assign):
    o = assign[j] - j
    if start is None or o != start[2] or j != start[0] + start[1]:
        if start is not None:
            segs.append(start)
        start = [j, 1, o]
    else:
        start[1] += 1
if start is not None:
    segs.append(start)
big = [s for s in segs if s[1] >= 200]
say("\nlong stretches (>=200 sentences) that line up with a constant offset: %d" % len(big))
say("   mb_start  length  index_start  (offset = index_start - mb_start)")
for s in big[:40]:
    say("   %8d  %6d  %11d  offset %+d" % (s[0], s[1], s[0] + s[2], s[2]))
moves = sum(1 for a, b in zip(big, big[1:]) if b[0] + b[2] < a[0] + a[2])
say("stretches that jump BACKWARD in the index (= chunk order differs): %d" % moves)

# ---- ModelBlocks sentences that did not fit anywhere ----------------------
miss = [j for j in range(n_mb) if j not in assign]
withf = sum(1 for j in miss if any(w in FILLERISH for w in mbw[j]))
say("\nmodelblocks sentences NOT placed: %d  (of which contain uh/um/etc: %d)" % (len(miss), withf))
for j in miss[:5]:
    say("   e.g. mb #%d: %s" % (j, " ".join(mbw[j][:20])))

# ---- build prev / foll ----------------------------------------------------
rows = []
for j, i in assign.items():
    ids = ix_ids[i]
    sv = [v for _, v in mb_sents[j]]
    n = len(ids)
    for k in range(n):
        rows.append((ids[k], sv[k], sv[k + 1] if k + 1 < n else np.nan))
out = pd.DataFrame(rows, columns=["word_id", "prev_syn_surp", "foll_syn_surp"])
out = idx[["word_id"]].drop_duplicates().merge(out, on="word_id", how="left")
out["prev_syn_surp"] = out["prev_syn_surp"].round(6)
out["foll_syn_surp"] = out["foll_syn_surp"].round(6)
out.to_csv(OUT, index=False)

say("\nwords in index                : %d" % len(out))
say("prev_syn_surp valid (aligned)  : %d (%.1f%%)"
    % (out["prev_syn_surp"].notna().sum(), 100 * out["prev_syn_surp"].notna().mean()))

if os.path.exists(OLD):
    old = pd.read_csv(OLD)
    m = old.merge(out, on="word_id", suffixes=("_old", "_new"))
    ov = m["prev_syn_surp_old"].notna()
    both = ov & m["prev_syn_surp_new"].notna()
    diff = both & ((m["prev_syn_surp_old"] - m["prev_syn_surp_new"]).abs() > 1e-4)
    gone = ov & m["prev_syn_surp_new"].isna()
    new_only = (~ov) & m["prev_syn_surp_new"].notna()
    say("\nvs current syntactic_surprisal.csv (%d valid values):" % ov.sum())
    say("  same value in both                  : %d" % (both.sum() - diff.sum()))
    say("  DIFFERENT value (old was on wrong word): %d  (%.1f%% of old)" % (diff.sum(), 100 * diff.sum() / ov.sum()))
    say("  valid in old, not placeable now     : %d" % gone.sum())
    say("  valid now, missing in old           : %d" % new_only.sum())

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines_out) + "\n")
say("\nwrote %s\nwrote %s\n(syntactic_surprisal.csv was not touched)" % (OUT, REPORT))