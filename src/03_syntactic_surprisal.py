"""
03_syntactic_surprisal.py (fast version)
Syntactic surprisal using PCFG on combined trees.
Uses vectorised numpy operations instead of nested Python loops.

INPUT:  data/combined.linetrees + data/processed/words_labeled.csv
OUTPUT: data/processed/syntactic_surprisal.csv
"""

import os
import math
import numpy as np
import pandas as pd
from nltk.tree import Tree

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
COMBINED_FILE = os.path.join(ROOT_DIR, "data", "combined.linetrees")
SWBD_FILE     = os.path.join(ROOT_DIR, "data", "swbd.linetrees")
LABELED_CSV   = os.path.join(ROOT_DIR, "data", "processed", "words_labeled.csv")
OUTPUT_CSV    = os.path.join(ROOT_DIR, "data", "processed", "syntactic_surprisal.csv")


def load_linetrees(filepath):
    print(f"Loading linetrees ...")
    trees = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trees.append(Tree.fromstring(line))
            except Exception:
                continue
    print(f"  Loaded {len(trees):,} trees")
    return trees


def build_fast_models(trees):
    print("\nBuilding fast probability models ...")

    pos_set  = set(["<s>", "</s>"])
    word_set = set(["<unk>"])

    for tree in trees:
        try:
            for word, pos in tree.pos():
                pos_set.add(pos.split("-")[0].split("=")[0])
                word_set.add(word.lower())
        except Exception:
            continue

    pos_list  = sorted(pos_set)
    word_list = sorted(word_set)
    pos2idx   = {p: i for i, p in enumerate(pos_list)}
    word2idx  = {w: i for i, w in enumerate(word_list)}

    n_pos  = len(pos_list)
    n_word = len(word_list)

    trans_counts = np.ones((n_pos, n_pos))
    lex_counts   = np.ones((n_pos, n_word))

    for tree in trees:
        try:
            leaves = tree.pos()
            tags   = ["<s>"] + [p.split("-")[0].split("=")[0] for _, p in leaves] + ["</s>"]
            words  = [w.lower() for w, _ in leaves]

            for i in range(1, len(tags)):
                pi = pos2idx.get(tags[i-1], 0)
                ci = pos2idx.get(tags[i],   0)
                trans_counts[pi, ci] += 1

            for i, (w, p) in enumerate(leaves):
                pi = pos2idx.get(p.split("-")[0].split("=")[0], 0)
                wi = word2idx.get(w.lower(), word2idx["<unk>"])
                lex_counts[pi, wi] += 1

        except Exception:
            continue

    trans_prob = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    lex_prob   = lex_counts   / lex_counts.sum(axis=1,   keepdims=True)

    print(f"  POS types:  {n_pos}")
    print(f"  Vocab size: {n_word}")

    return trans_prob, lex_prob, pos2idx, word2idx, pos_list, n_pos


def compute_surprisal_fast(words, trans_prob, lex_prob,
                            pos2idx, word2idx, n_pos):
    pos_dist   = np.ones(n_pos) / n_pos
    surprisals = []
    unk_idx    = word2idx.get("<unk>", 0)

    for word in words:
        # ── FIX: handle NaN and non-string words ──────────────
        if not isinstance(word, str) or word != word:
            word = "<unk>"
        # ──────────────────────────────────────────────────────
        w_idx  = word2idx.get(word.lower(), unk_idx)
        pos_pred = pos_dist @ trans_prob
        lex_w    = lex_prob[:, w_idx]
        joint    = pos_pred * lex_w
        p_word   = joint.sum()
        surp     = -math.log2(p_word) if p_word > 0 else 20.0
        surprisals.append(surp)
        total    = joint.sum()
        pos_dist = joint / total if total > 0 else pos_pred

    return surprisals


def compute_all_fast(df, trans_prob, lex_prob, pos2idx, word2idx, n_pos):
    print("\nComputing syntactic surprisal (fast mode) ...")
    df      = df.sort_values(["conv_id","speaker","start"]).reset_index(drop=True)
    results = []
    groups  = list(df.groupby(["conv_id","speaker"], sort=False))
    total   = len(groups)

    for g_idx, ((conv_id, speaker), group) in enumerate(groups):
        group = group.reset_index(drop=True)
        n     = len(group)
        words = group["word"].astype(str).tolist()
        ids   = group["word_id"].tolist()

        surprisals = compute_surprisal_fast(
            words, trans_prob, lex_prob, pos2idx, word2idx, n_pos
        )

        for i in range(n):
            results.append({
                "word_id":       ids[i],
                "prev_syn_surp": round(surprisals[i], 6),
                "foll_syn_surp": round(surprisals[i+1], 6) if i+1 < n else float("nan"),
            })

        if (g_idx+1) % 100 == 0:
            print(f"  {g_idx+1}/{total} conversations done ...")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 55)
    print("Step 3: Syntactic Surprisal (FAST mode)")
    print("(PCFG on PTB + Switchboard combined trees)")
    print("=" * 55)

    if os.path.exists(COMBINED_FILE):
        linetrees_file = COMBINED_FILE
        print(f"\nUsing combined linetrees (PTB + Switchboard)")
    elif os.path.exists(SWBD_FILE):
        linetrees_file = SWBD_FILE
        print(f"\nUsing Switchboard only linetrees")
    else:
        raise FileNotFoundError("Run: python src/convert_swbd_syntax.py first.")

    trees = load_linetrees(linetrees_file)
    trans_prob, lex_prob, pos2idx, word2idx, pos_list, n_pos = build_fast_models(trees)

    print("\nLoading words_labeled.csv ...")
    df = pd.read_csv(LABELED_CSV)
    df = df.dropna(subset=["duration","start"])
    df["word"] = df["word"].astype(str)
    print(f"  {len(df):,} words loaded")

    result_df = compute_all_fast(
        df, trans_prob, lex_prob, pos2idx, word2idx, n_pos
    )

    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")
    print(f"Rows:   {len(result_df):,}")
    print("\nSample:")
    print(result_df.head(5).to_string())
    print("\n" + "=" * 55)
    print("Step 3 COMPLETE")
    print("=" * 55)
    print(f"Training trees: {len(trees):,}")
    print("Source: Switchboard NXT + Penn Treebank (NLTK)")