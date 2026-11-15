"""
17_multi_llm_surprisal.py

Multi-LLM surprisal extension -- Dr. Ranjan's second action item.

Generalizes 14_gpt2_surprisal.py (which was GPT-2-only) to run the SAME
per-word tokenization/aggregation logic across several small open causal
LMs, so we can compare how surprisal from different LLMs predicts
disfluency, not just GPT-2 vs ModelBlocks.

Models chosen with CPU runtime as the hard constraint -- GPT-2 (124M) took
~2.5 hours for the full corpus on CPU (see pipeline_log.txt from
14_gpt2_surprisal.py), so anything much bigger than ~350M params risks a
multi-day run. All five are genuine base/completion causal LMs (NOT
instruction-tuned chat models) -- reusing an instruction-tuned model was
already tried and rejected earlier in this project (see this file's sibling
script's docstring: Llama 3.2 via Ollama doesn't do genuine next-token
completion the way a base model does).

Models:
  gpt2                    124M   already done in 14_gpt2_surprisal.py,
                                 included here again for a uniform
                                 single-script comparison table
  distilgpt2               82M   fastest, a lower-capacity anchor
  gpt2-medium             355M   upper end of what's CPU-feasible
  EleutherAI/pythia-160m   160M  different family + pretraining corpus
                                 (the Pile, not WebText) -- tests whether
                                 the GPT-2 vs ModelBlocks story is a GPT-2
                                 quirk or holds across model families
  facebook/opt-125m        125M  another different family/corpus, for the
                                 same reason

Per-word tokenization/aggregation is IDENTICAL to the proven, debugged
logic in 14_gpt2_surprisal.py (see that file's compute_sentence_surprisal
docstring for the two wrong approaches that were tried and rejected before
landing on this one): each word is tokenized separately first via
tokenizer.encode(" " + word, add_special_tokens=False) so token-to-word
ownership is known by construction, never guessed from character offsets.
A single forward pass still sees the whole sentence for real context.

in:  data/processed/clean_word_index.csv        (same input as GPT-2 stage 1)
out: data/processed/surprisal_<model_tag>.csv    (one file per model)
     columns: word_id, prev_<tag>_surp, foll_<tag>_surp
     data/processed/multi_llm_run_summary.csv    (timing/coverage per model)

Usage:
  python src/17_multi_llm_surprisal.py                  # run all models
  python src/17_multi_llm_surprisal.py --model distilgpt2   # just one
  python src/17_multi_llm_surprisal.py --list            # show model list, exit

Nitin, Sept 2026 (multi-LLM extension, per Dr. Ranjan's review comments)
"""

import os
import sys
import math
import argparse
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

CLEAN_INDEX_CSV = os.path.join(PROCESSED, "clean_word_index.csv")
LOG_FILE        = os.path.join(PROCESSED, "pipeline_log.txt")
SUMMARY_CSV     = os.path.join(PROCESSED, "multi_llm_run_summary.csv")

# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# tag is used in output filenames/columns -- keep it short, no slashes
# ─────────────────────────────────────────────────────────────
MODELS = {
    "gpt2":        {"hf_name": "gpt2",                    "tag": "gpt2",    "params": "124M"},
    "distilgpt2":  {"hf_name": "distilgpt2",               "tag": "dgpt2",   "params": "82M"},
    "gpt2-medium": {"hf_name": "gpt2-medium",               "tag": "gpt2m",   "params": "355M"},
    "pythia-160m": {"hf_name": "EleutherAI/pythia-160m",    "tag": "pyt160",  "params": "160M"},
    "opt-125m":    {"hf_name": "facebook/opt-125m",         "tag": "opt125",  "params": "125M"},
}


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


def load_model(hf_name):
    log(f"Loading tokenizer and model: {hf_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(hf_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # not every tokenizer has a BOS token (e.g. some GPT-2-family configs
    # leave it unset) -- fall back to the tokenizer's own pad/eos token,
    # which every one of these five models does define, so the first word
    # in a sentence still gets a real "previous context" to condition on
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError(
            f"{hf_name} has neither a bos_token_id nor an eos_token_id -- "
            "can't pad the first word of each sentence. Needs a manual fix."
        )

    log(f"  Loaded. Running on device: {device}. BOS/pad id used: {bos_id}")
    return tokenizer, model, device, bos_id


def compute_sentence_surprisal(words, tokenizer, model, device, bos_id):
    """
    Same logic as 14_gpt2_surprisal.py's compute_sentence_surprisal(),
    generalized to any AutoTokenizer/AutoModelForCausalLM pair.

    Tokenizes each word SEPARATELY first (with a leading space, matching
    normal mid-sentence tokenization) so token-to-word ownership is known
    by construction rather than reconstructed afterward from character
    offsets -- the offset-matching approach was tried for GPT-2 and
    produced 93% of words with no valid token assignment; tokenizing
    per-word up front avoids that failure mode entirely, for any BPE/
    byte-level tokenizer.
    """
    all_token_ids = [bos_id]
    word_token_counts = []
    for w in words:
        piece_ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(piece_ids) == 0:
            piece_ids = tokenizer.encode(w, add_special_tokens=False)
        word_token_counts.append(len(piece_ids))
        all_token_ids.extend(piece_ids)

    no_fragmentation = all(c == 1 for c in word_token_counts)

    input_ids = torch.tensor([all_token_ids]).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits, dim=-1)

    token_surprisals = []
    for i in range(1, input_ids.shape[1]):
        actual_token_id = input_ids[0, i].item()
        predicted_log_prob = log_probs[0, i - 1, actual_token_id].item()
        token_surprisals.append(-predicted_log_prob / math.log(2))

    word_surprisals = []
    cursor = 0
    for count in word_token_counts:
        piece_surps = token_surprisals[cursor:cursor + count]
        word_surprisals.append(sum(piece_surps))
        cursor += count

    n_words_with_no_token = sum(1 for c in word_token_counts if c == 0)

    return word_surprisals, no_fragmentation, n_words_with_no_token


def run_one_model(model_key, clean_idx):
    cfg = MODELS[model_key]
    hf_name, tag = cfg["hf_name"], cfg["tag"]
    output_csv = os.path.join(PROCESSED, f"surprisal_{tag}.csv")

    log_section(f"Multi-LLM surprisal -- {model_key} ({hf_name}, {cfg['params']})")

    if os.path.exists(output_csv):
        log(f"  {output_csv} already exists -- skipping (delete it to force a rerun)")
        return {"model": model_key, "hf_name": hf_name, "tag": tag,
                "status": "skipped_existing", "seconds": 0, "n_words": None}

    t_start = time.time()
    tokenizer, model, device, bos_id = load_model(hf_name)

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

        words, ids = [], []
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
            words, tokenizer, model, device, bos_id
        )
        if not no_fragmentation:
            n_fragmented_sentences += 1
        n_words_no_token_total += n_no_token

        for i in range(n):
            prev_s = surprisals[i]
            foll_s = surprisals[i + 1] if i + 1 < n else np.nan
            results.append({
                "word_id": ids[i],
                f"prev_{tag}_surp": round(prev_s, 6),
                f"foll_{tag}_surp": round(foll_s, 6) if not (isinstance(foll_s, float) and np.isnan(foll_s)) else np.nan,
            })

        if (g_idx + 1) % 5000 == 0:
            elapsed = time.time() - t_start
            rate = (g_idx + 1) / elapsed
            eta_min = (total - (g_idx + 1)) / rate / 60
            log(f"  {g_idx+1}/{total} sentences done... (ETA ~{eta_min:.0f} min)")

    elapsed = time.time() - t_start
    log(f"  Done -- all {total} sentences processed in {elapsed/60:.1f} min")
    log(f"  Sentences with >=1 fragmented word: {n_fragmented_sentences:,} "
        f"({100*n_fragmented_sentences/total:.2f}%)")
    log(f"  Rows dropped for non-string/empty word: {n_bad_word_rows:,}")
    log(f"  Words with no token assignment (should be 0): {n_words_no_token_total:,}")

    df_out = pd.DataFrame(results)
    log(f"  Total words with {model_key} surprisal: {len(df_out):,}")

    df_out.to_csv(output_csv, index=False)
    log(f"  Saved -> {output_csv}")

    # free memory before the next model loads -- five models loaded
    # simultaneously would be wasteful and risks OOM on a laptop
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_key, "hf_name": hf_name, "tag": tag,
        "status": "completed", "seconds": round(elapsed, 1),
        "n_words": len(df_out),
        "pct_fragmented_sentences": round(100 * n_fragmented_sentences / total, 2),
        "n_no_token_assigned": n_words_no_token_total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                         help="Run only this one model (default: run all)")
    parser.add_argument("--list", action="store_true",
                         help="Print the model registry and exit")
    args = parser.parse_args()

    if args.list:
        print("Registered models:")
        for k, v in MODELS.items():
            print(f"  {k:14s} {v['hf_name']:30s} {v['params']:6s} tag={v['tag']}")
        sys.exit(0)

    log_section("Multi-LLM Surprisal Extension -- Stage 1")

    if not os.path.exists(CLEAN_INDEX_CSV):
        raise FileNotFoundError(
            f"{CLEAN_INDEX_CSV} not found -- run the earlier pipeline steps first "
            "(same input file 14_gpt2_surprisal.py used)"
        )
    clean_idx = pd.read_csv(CLEAN_INDEX_CSV)
    log(f"Loaded clean_word_index.csv: {len(clean_idx):,} words, "
        f"{clean_idx['sent_idx'].nunique():,} sentences")

    models_to_run = [args.model] if args.model else list(MODELS.keys())
    log(f"Models queued: {models_to_run}")

    summary_rows = []
    for model_key in models_to_run:
        try:
            row = run_one_model(model_key, clean_idx)
        except Exception as e:
            log(f"  ERROR running {model_key}: {e}")
            row = {"model": model_key, "hf_name": MODELS[model_key]["hf_name"],
                   "tag": MODELS[model_key]["tag"], "status": f"error: {e}",
                   "seconds": None, "n_words": None}
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    # append rather than overwrite, so re-running --model X doesn't erase
    # earlier models' summary rows
    if os.path.exists(SUMMARY_CSV):
        prior = pd.read_csv(SUMMARY_CSV)
        summary_df = pd.concat([prior[~prior["model"].isin(summary_df["model"])],
                                 summary_df], ignore_index=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    log_section("Stage 1 COMPLETE -- Summary")
    print(summary_df.to_string(index=False))
    log(f"Saved summary -> {SUMMARY_CSV}")
    log("\nNext: python src/18_merge_multi_llm.py   (merges each surprisal_<tag>.csv")
    log("into dataset_filler.csv / dataset_reparandum.csv via the existing")
    log("propagate_feature() logic, same as GPT-2 was merged in stage 2)")