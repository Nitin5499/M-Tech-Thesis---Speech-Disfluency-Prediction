"""
recompute_all_features.py
=========================
Recomputes lexical surprisal and DLT costs for ALL words
(including fillers and reparandums) so the GLMM datasets
are not empty.

The DC correction is applied at the ModelBlocks level —
synproc sees only clean sentences. But the GLMM needs
feature values for ALL words to classify them.

For disfluent words — features are computed from their
position in the full conversation context.

INPUT:
  data/processed/words_labeled.csv  -- all 830,187 words
  data/OANC-GrAF/                   -- for lexical surprisal

OUTPUT:
  data/processed/lexical_surprisal.csv  -- all words
  data/processed/dlt_costs.csv          -- all words

Author: Nitin Kumar
Date: June 2026
"""

import os
import math
import numpy as np
import pandas as pd
import spacy
from collections import defaultdict
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
PROCESSED   = os.path.join(ROOT_DIR, "data", "processed")
OANC_DIR    = os.path.join(ROOT_DIR, "data", "OANC-GrAF")
LABELED_CSV = os.path.join(PROCESSED, "words_labeled.csv")
LEX_CSV     = os.path.join(PROCESSED, "lexical_surprisal.csv")
DLT_CSV     = os.path.join(PROCESSED, "dlt_costs.csv")
LOG_FILE    = os.path.join(PROCESSED, "pipeline_log.txt")

SKIP_SUFFIXES = (
    ".anc", ".xml", ".hepple", ".logical",
    ".css", ".dtd", ".xsd", ".html", ".htm",
)

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def log_section(title):
    line = "\n" + "=" * 60 + f"\n{title}\n" + "=" * 60
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ── Lexical Surprisal ──────────────────────────────────────────
def is_plain_text(fname):
    lower = fname.lower()
    for s in SKIP_SUFFIXES:
        if lower.endswith(s):
            return False
    if fname.startswith("."):
        return False
    if lower.endswith(".txt"):
        return True
    if "." not in fname:
        return True
    return False

def load_oanc(oanc_dir):
    log(f"Loading OANC-GrAF...")
    sentences = []
    file_count = 0
    for dirpath, _, filenames in os.walk(oanc_dir):
        for fname in filenames:
            if not is_plain_text(fname):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        tokens = line.strip().lower().split()
                        if tokens:
                            sentences.append(tokens)
                file_count += 1
            except:
                continue
    log(f"  Files: {file_count:,}  Sentences: {len(sentences):,}")
    return sentences

def train_trigram(sentences):
    log("Training trigram model...")
    unigram = defaultdict(int)
    trigram = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    bigram_total = defaultdict(lambda: defaultdict(int))
    for sent in sentences:
        padded = ["<s>", "<s>"] + sent + ["</s>"]
        for i in range(2, len(padded)):
            w0, w1, w2 = padded[i-2], padded[i-1], padded[i]
            unigram[w2] += 1
            trigram[w0][w1][w2] += 1
            bigram_total[w0][w1] += 1
    V = len(unigram)
    log(f"  Vocabulary: {V:,}")
    def surprisal_fn(word, ctx2, ctx1):
        w  = str(word).lower() if word else "<unk>"
        c2 = str(ctx2).lower() if ctx2 else "<s>"
        c1 = str(ctx1).lower() if ctx1 else "<s>"
        num = trigram[c2][c1][w] + 1
        den = bigram_total[c2][c1] + V
        return -math.log2(num / den) if den > 0 else 0.0
    return surprisal_fn

def compute_lexical_surprisal(df, surprisal_fn):
    log("Computing lexical surprisal for ALL words...")
    df = df.sort_values(["conv_id","speaker","start"]).reset_index(drop=True)
    results = []
    groups = list(df.groupby(["conv_id","speaker"], sort=False))
    for g_idx, ((conv_id, speaker), group) in enumerate(groups):
        group = group.reset_index(drop=True)
        n = len(group)
        words = group["word"].astype(str).str.lower().tolist()
        ids = group["word_id"].tolist()
        for i in range(n):
            w_km2 = words[i-2] if i >= 2 else "<s>"
            w_km1 = words[i-1] if i >= 1 else "<s>"
            w_k   = words[i]
            w_kp1 = words[i+1] if i < n-1 else "</s>"
            prev_s = surprisal_fn(w_k, w_km2, w_km1)
            foll_s = surprisal_fn(w_kp1, w_km1, w_k) if i < n-1 else np.nan
            results.append({
                "word_id": ids[i],
                "prev_lex_surp": round(prev_s, 6),
                "foll_lex_surp": round(foll_s, 6) if not np.isnan(foll_s) else np.nan,
            })
        if (g_idx + 1) % 100 == 0:
            log(f"  {g_idx+1}/{len(groups)} conversations done...")
    return pd.DataFrame(results)

# ── DLT Costs ──────────────────────────────────────────────────
def compute_dlt_sentence(words, nlp):
    if not words:
        return [], []
    text = " ".join(str(w) for w in words if w and str(w).strip())
    if not text.strip():
        return [1.0] * len(words), [1.0] * len(words)
    try:
        doc = nlp(text)
        IC = []
        SC = []
        for i, token in enumerate(doc):
            ic = 1
            for dep_tok in doc:
                if dep_tok.head.i == i and dep_tok.i != i:
                    ic += abs(dep_tok.i - i)
            IC.append(float(ic))
            sc = 1
            for dep_tok in doc[:i+1]:
                if dep_tok.head.i > i:
                    sc += 1
            SC.append(float(sc))
    except:
        IC = [1.0] * len(words)
        SC = [1.0] * len(words)
    while len(IC) < len(words):
        IC.append(1.0)
        SC.append(1.0)
    return IC[:len(words)], SC[:len(words)]

def compute_dlt_all(df, nlp):
    log("Computing DLT costs for ALL words...")
    df = df.sort_values(["conv_id","speaker","start"]).reset_index(drop=True)
    results = []
    groups = list(df.groupby(["conv_id","speaker"], sort=False))
    for g_idx, ((conv_id, speaker), group) in enumerate(groups):
        group = group.reset_index(drop=True)
        n = len(group)
        words = group["word"].astype(str).tolist()
        ids = group["word_id"].tolist()
        IC_list, SC_list = compute_dlt_sentence(words, nlp)
        for i in range(n):
            results.append({
                "word_id": ids[i],
                "prev_IC": IC_list[i],
                "foll_IC": IC_list[i+1] if i+1 < n else np.nan,
                "prev_SC": SC_list[i],
                "foll_SC": SC_list[i+1] if i+1 < n else np.nan,
            })
        if (g_idx + 1) % 100 == 0:
            log(f"  {g_idx+1}/{len(groups)} conversations done...")
    return pd.DataFrame(results)

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    log_section("Recompute Features for ALL Words")
    log("Loading words_labeled.csv...")
    df = pd.read_csv(LABELED_CSV)
    df = df.dropna(subset=["duration","start"])
    df["word"] = df["word"].fillna("").astype(str)
    log(f"  Total words: {len(df):,}")
    log(f"  Fillers: {(df['label']=='filler').sum():,}")
    log(f"  Reparandums: {(df['label']=='reparandum').sum():,}")
    log(f"  Fluent: {(df['label']=='fluent').sum():,}")

    # ── Lexical Surprisal ─────────────────────────────────────
    log_section("Step 1: Lexical Surprisal")
    sentences = load_oanc(OANC_DIR)
    surprisal_fn = train_trigram(sentences)
    lex_df = compute_lexical_surprisal(df, surprisal_fn)
    lex_df.to_csv(LEX_CSV, index=False)
    log(f"Saved -> {LEX_CSV}")
    log(f"  Rows: {len(lex_df):,}")
    log(f"  prev_lex_surp ok: {lex_df['prev_lex_surp'].notna().sum():,}")

    # ── DLT Costs ─────────────────────────────────────────────
    log_section("Step 2: DLT Costs")
    try:
        nlp = spacy.load("en_core_web_sm")
        log("spaCy loaded")
    except:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    dlt_df = compute_dlt_all(df, nlp)
    dlt_df.to_csv(DLT_CSV, index=False)
    log(f"Saved -> {DLT_CSV}")
    log(f"  Rows: {len(dlt_df):,}")
    log(f"  prev_IC ok: {dlt_df['prev_IC'].notna().sum():,}")

    log_section("COMPLETE")
    log("Lexical surprisal and DLT recomputed for all 830,187 words")
    log("Next: python src/05_build_dataset.py")