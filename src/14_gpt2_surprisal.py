"""
14_gpt2_surprisal.py

GPT-2 surprisal extension, stage 1.

Computes GPT-2 surprisal for every word in the clean corpus, as an extra
feature alongside the existing trigram lexical surprisal and the
ModelBlocks syntactic surprisal.

Why this doesn't repeat the earlier LLM pilot's mistake: that attempt used
Llama 3.2 through Ollama, which is an instruction-tuned chat model -- it
doesn't do genuine next-token completion the way a base model does. GPT-2
through Hugging Face Transformers is an actual base/completion model, and
I validated it on real corpus sentences (test_gpt2_surprisal.py) before
using it here.

Already checked (see test_gpt2_surprisal.py):
  - GPT-2's tokenizer splits filler words (uh, um) into single clean
    tokens on the examples I tried -- still logging per-sentence below
    rather than assuming this holds across the whole corpus
  - first word of each sentence gets GPT-2's own BOS token as padding,
    same idea as the trigram model's "<s> <s>" padding, so every word
    gets a real value

in:  data/processed/clean_word_index.csv
out: data/processed/gpt2_surprisal.csv
     columns: word_id, prev_gpt2_surp, foll_gpt2_surp

Nitin, Aug 2026 (GPT-2 surprisal extension, stage 1)
"""

import os
import math
import pandas as pd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

CLEAN_INDEX_CSV = os.path.join(PROCESSED, "clean_word_index.csv")
OUTPUT_CSV      = os.path.join(PROCESSED, "gpt2_surprisal.csv")
LOG_FILE        = os.path.join(PROCESSED, "pipeline_log.txt")


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


def load_gpt2():
    log("loading GPT-2 tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()  # inference mode, turns off dropout
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    log(f"  loaded, running on {device}")
    return tokenizer, model, device


def compute_sentence_surprisal(words, tokenizer, model, device):
    """
    Takes a list of clean (disfluency-free) words forming one sentence,
    returns per-word surprisal in bits -- summed correctly when a word
    splits into multiple GPT-2 subword tokens.

    Went through a couple of wrong approaches before this one:
      v1 just assumed one GPT-2 token per word and zipped surprisals
      straight against the word list. Broke silently for the 52.52% of
      sentences where a word actually split into more than one token.

      v2 tried fixing it with hand-computed character-offset spans
      matched against the tokenizer's return_offsets_mapping. This was
      worse -- 100% of sentences and 93% of words ended up with no valid
      token assignment. Turned out GPT-2 doesn't treat a leading space as
      its own character -- it gets absorbed into the following token
      (that's the "Ġ" prefix you see if you print the raw tokens), so a
      token's reported start offset doesn't reliably land inside a word
      span built from simple string concatenation. Trying to out-guess
      GPT-2's own space handling was the wrong approach entirely.

    What actually works (this version): tokenize the sentence one word at
    a time, in order, using encode() on each word with a leading space
    (matching how it'd tokenize mid-sentence). No offset-matching needed
    -- since I encoded each word myself, I already know exactly how many
    tokens it produced. Still run the whole sequence through the model in
    one forward pass (full sentence context for predictions), just with
    token-to-word ownership tracked from construction instead of guessed
    afterward.
    """
    bos_id = tokenizer.bos_token_id

    # encode each word separately to know how many tokens it produces and
    # build the full sequence with known ownership. leading space added
    # before every word to match normal mid-sentence tokenization -- the
    # first word gets its leading space naturally by following BOS,
    # consistent with how the first validation test worked
    all_token_ids = [bos_id]
    word_token_counts = []
    for w in words:
        piece_ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(piece_ids) == 0:
            # shouldn't happen (caller filters empty words) but don't
            # crash on it if it slips through
            piece_ids = tokenizer.encode(w, add_special_tokens=False)
        word_token_counts.append(len(piece_ids))
        all_token_ids.extend(piece_ids)

    no_fragmentation = all(c == 1 for c in word_token_counts)

    input_ids = torch.tensor([all_token_ids]).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits, dim=-1)

    # per-token surprisal, skipping the BOS position
    token_surprisals = []
    for i in range(1, input_ids.shape[1]):
        actual_token_id = input_ids[0, i].item()
        predicted_log_prob = log_probs[0, i - 1, actual_token_id].item()
        token_surprisals.append(-predicted_log_prob / math.log(2))

    # walk through the token surprisals, consuming word_token_counts[w]
    # of them per word -- ownership known by construction
    word_surprisals = []
    cursor = 0
    for count in word_token_counts:
        piece_surps = token_surprisals[cursor:cursor + count]
        word_surprisals.append(sum(piece_surps))
        cursor += count

    n_words_with_no_token = sum(1 for c in word_token_counts if c == 0)

    return word_surprisals, no_fragmentation, n_words_with_no_token


if __name__ == "__main__":
    log_section("GPT-2 surprisal extension, stage 1: compute surprisal on clean corpus")

    tokenizer, model, device = load_gpt2()

    log("loading clean_word_index.csv...")
    clean_idx = pd.read_csv(CLEAN_INDEX_CSV)
    log(f"  {len(clean_idx):,} clean words")

    log("computing GPT-2 surprisal sentence by sentence...")
    log("(clean, disfluency-free text only)")

    results = []
    n_fragmented_sentences = 0
    n_bad_word_rows = 0
    n_sentences_with_bad_words = 0
    n_words_no_token_total = 0
    groups = list(clean_idx.groupby('sent_idx', sort=True))
    total = len(groups)

    for g_idx, (sent_idx, group) in enumerate(groups):
        group = group.sort_values('word_pos').reset_index(drop=True)
        raw_words = group['word'].tolist()
        ids_all = group['word_id'].tolist()

        # clean_word_index.csv shouldn't have a missing/non-string word
        # value, but the first real run died at sentence ~8000/74497 with
        # "expected str instance, float found" -- one row's word value was
        # a raw float, presumably a NaN cell that survived as an actual
        # float instead of text. dropping and counting these instead of
        # str()-coercing them, since str(nan) -> the literal text "nan",
        # which would get fed to GPT-2 as a real spoken word
        words = []
        ids = []
        n_dropped_this_sentence = 0
        for w, wid in zip(raw_words, ids_all):
            if isinstance(w, str) and w.strip() != "":
                words.append(w)
                ids.append(wid)
            else:
                n_dropped_this_sentence += 1

        if n_dropped_this_sentence > 0:
            n_bad_word_rows += n_dropped_this_sentence
            n_sentences_with_bad_words += 1

        n = len(words)
        if n == 0:
            continue

        surprisals, no_fragmentation, n_no_token = compute_sentence_surprisal(
            words, tokenizer, model, device
        )
        if not no_fragmentation:
            n_fragmented_sentences += 1
        n_words_no_token_total += n_no_token

        for i in range(n):
            prev_s = surprisals[i]
            foll_s = surprisals[i + 1] if i + 1 < n else np.nan

            results.append({
                "word_id": ids[i],
                "prev_gpt2_surp": round(prev_s, 6),
                "foll_gpt2_surp": round(foll_s, 6) if not (isinstance(foll_s, float) and np.isnan(foll_s)) else np.nan,
            })

        if (g_idx + 1) % 2000 == 0:
            log(f"  {g_idx+1}/{total} sentences done")

    log(f"  done, all {total} sentences processed")
    log(f"  sentences with at least one word split into multiple GPT-2 tokens: "
        f"{n_fragmented_sentences:,} ({100*n_fragmented_sentences/total:.2f}%)")
    log(f"  rows dropped for a non-string/empty word value: {n_bad_word_rows:,} "
        f"(across {n_sentences_with_bad_words:,} sentences)")
    log(f"  words with no token assignment during aggregation (should be ~0): {n_words_no_token_total:,}")
    if n_words_no_token_total > 0:
        log("  WARNING: non-zero here means some words didn't get tokens assigned -- check those before trusting results")
    if n_bad_word_rows > 0:
        log("  note: dropped word_ids get no row in gpt2_surprisal.csv, so they'll need")
        log("  propagation the same way fillers/reparandums do when merged in stage 2")

    gpt2_df = pd.DataFrame(results)
    log(f"total words with GPT-2 surprisal: {len(gpt2_df):,}")

    n_null_prev = gpt2_df['prev_gpt2_surp'].isna().sum()
    n_null_foll = gpt2_df['foll_gpt2_surp'].isna().sum()
    log(f"  missing prev_gpt2_surp: {n_null_prev:,} (should be 0, every word gets a value)")
    log(f"  missing foll_gpt2_surp: {n_null_foll:,} (expected, last word of each sentence has no next word)")

    gpt2_df.to_csv(OUTPUT_CSV, index=False)

    log_section("stage 1 done (per-word tokenization fix applied)")
    log(f"saved -> {OUTPUT_CSV}")
    log("this run tokenizes each word separately to track exact token counts, then sums")
    log("multi-token surprisals per word -- fixes the misalignment from the earlier version")
    log("which assumed one token per word (and the offset-matching attempt before that,")
    log("which was worse -- see compute_sentence_surprisal() docstring)")
    log("next: propagation to disfluent words happens through the existing")
    log("propagate_feature() logic when this merges into 05_build_dataset.py's output --")
    log("not a new propagation step, reusing what's already there")