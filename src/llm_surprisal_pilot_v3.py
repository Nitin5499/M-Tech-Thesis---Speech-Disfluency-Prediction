"""
llm_surprisal_pilot.py  (v3 -- raw mode fix)
================================================================
PILOT / PRELIMINARY EXPERIMENT -- Future Work extension.

CHANGE LOG:
  v1: used isolated word as context, no real preceding words.
      Result: implausible near-zero surprisal (mean 0.24 bits).
      Cause: no real context to predict from.

  v2: fixed context to use REAL 2 preceding words from the actual
      sentence (matching trigram model's window exactly).
      Result: STILL implausible (mean 0.31 bits in test batch),
      model generated "It"/"Here" regardless of context.
      Cause diagnosed: llama3.2 is INSTRUCTION-TUNED. Ollama's
      default /api/generate wraps every prompt in a chat template
      (e.g. "[INST] ... [/INST]"), so the model treats "pretty
      much" not as text-to-continue but as a conversational
      message, and responds with a chatty opener ("It sounds
      like...") rather than performing next-word completion.

  v3 (this version): two fixes applied together --
      1. raw: true -- tells Ollama to skip the chat template
         entirely and send the prompt exactly as-is to the model,
         so it sees plain text rather than a wrapped conversational
         turn.
      2. A short completion-style system-style prefix is prepended
         directly into the raw prompt (since raw mode bypasses the
         system field's templating too), explicitly framing the
         task as continuation, not conversation.
      Both are verified on a small test batch BEFORE any full run,
      same discipline as v2. If the test batch still looks
      degenerate, the script stops itself automatically.

STATUS: PILOT on a documented random sample (Optional up to 300
words), not a full corpus run.

INPUT:
  data/processed/dataset_reparandum.csv    -- sample source (word_id list)
  data/processed/clean_word_index.csv      -- for reconstructing real
                                               sentence context per word

OUTPUT:
  data/processed/llm_surprisal_pilot_v3.csv
  data/processed/llm_surprisal_pilot_v3_log.txt

Author: Nitin Kumar
Date: August 2026 (interim evaluation pilot, v3)
"""

import os
import math
import random
import requests
import pandas as pd
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED   = os.path.join(ROOT_DIR, "data", "processed")
REP_CSV     = os.path.join(PROCESSED, "dataset_reparandum.csv")
CLEAN_INDEX = os.path.join(PROCESSED, "clean_word_index.csv")
OUTPUT_CSV  = os.path.join(PROCESSED, "llm_surprisal_pilot_v3.csv")
LOG_FILE    = os.path.join(PROCESSED, "llm_surprisal_pilot_v3_log.txt")

# ── Config ─────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "llama3.2"
SAMPLE_SIZE  = 300
TEST_BATCH   = 15      # slightly larger test batch this time -- more confidence before committing
RANDOM_SEED  = 42

# The raw-mode prefix. Because raw:true means NO chat template is
# applied, this text becomes part of the literal prompt the model
# sees -- it is NOT a system message, it's plain text instructing
# continuation behavior, followed immediately by the actual context.
RAW_PREFIX = (
    "Complete the following sentence fragment with exactly ONE most "
    "likely next word. Output ONLY that single word, nothing else, "
    "no punctuation, no explanation.\n\nFragment: "
)

# ── Logger ─────────────────────────────────────────────────────
def log(msg):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def log_section(title):
    line = "\n" + "=" * 60 + f"\n{title}\n" + "=" * 60
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ── Build sentence lookup from clean_word_index.csv ────────────
def build_sentence_lookup(clean_idx_path):
    log("Loading clean_word_index.csv to reconstruct sentence context...")
    idx = pd.read_csv(clean_idx_path)
    log(f"  Loaded {len(idx):,} clean words, {idx['sent_idx'].nunique():,} sentences")

    sentences = {}
    for sent_idx, group in idx.groupby('sent_idx', sort=True):
        group_sorted = group.sort_values('word_pos')
        sentences[sent_idx] = group_sorted['word'].astype(str).tolist()

    word_to_location = {}
    for _, row in idx.iterrows():
        word_to_location[row['word_id']] = (row['sent_idx'], row['word_pos'])

    log(f"  Built lookup for {len(word_to_location):,} word_ids")
    return sentences, word_to_location


def get_preceding_context(word_id, sentences, word_to_location):
    loc = word_to_location.get(word_id)
    if loc is None:
        return None, None, "word_id_not_found_in_clean_index"

    sent_idx, word_pos = loc
    sent_words = sentences.get(sent_idx, [])

    w_km2 = sent_words[word_pos - 2] if word_pos >= 2 else ""
    w_km1 = sent_words[word_pos - 1] if word_pos >= 1 else ""
    return w_km2, w_km1, "ok"


# ── Ollama connectivity + raw-mode check ───────────────────────
def check_ollama():
    log("Checking Ollama connectivity, raw mode, and logprobs support...")
    test_prompt = RAW_PREFIX + "the cat sat on the"
    test_payload = {
        "model": MODEL_NAME,
        "prompt": test_prompt,
        "stream": False,
        "raw": True,
        "options": {"num_predict": 3, "temperature": 0},
        "logprobs": True,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=test_payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        log(f"  Ollama responded. Keys: {list(data.keys())}")
        log(f"  Test generation for 'the cat sat on the' -> '{data.get('response','').strip()}'")
        if "logprobs" not in data:
            log("  ERROR: 'logprobs' key not found.")
            return False
        return True
    except Exception as e:
        log(f"  ERROR: Could not reach Ollama at {OLLAMA_URL}")
        log(f"  {e}")
        raise SystemExit(1)


# ── Surprisal via logprobs, raw mode, real 2-word context ──────
def get_word_surprisal(w_km2, w_km1, target_word):
    context = f"{w_km2} {w_km1}".strip()
    if not context:
        context = "the"   # minimal fallback for sentence-initial words -- avoids empty prompt
    prompt = RAW_PREFIX + context

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {"num_predict": 1, "temperature": 0},
        "logprobs": True,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return None, None, f"request_error: {e}"

    model_token = data.get("response", "").strip()
    lp_info = data.get("logprobs")
    try:
        token_logprob = lp_info[0]["logprob"] if isinstance(lp_info, list) and lp_info else None
    except (KeyError, IndexError, TypeError):
        token_logprob = None

    if token_logprob is None:
        return None, model_token, "logprob_extract_failed"

    surprisal_bits = -token_logprob / math.log(2)
    clean_model_tok = model_token.lower().strip(".,!?\"'")
    clean_target    = target_word.lower().strip(".,!?\"'")
    match_flag = "match" if clean_model_tok == clean_target else "no_match"
    return surprisal_bits, model_token, match_flag


# ── Main ───────────────────────────────────────────────────────
def main():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("LLM Surprisal Pilot v3 (raw mode) -- Run Log\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    log_section("LLM Surprisal Pilot v3 -- raw:true + completion prefix")
    log(f"Model: {MODEL_NAME} (local via Ollama, raw mode)")
    log(f"Sample size: up to {SAMPLE_SIZE} words (seed={RANDOM_SEED})")
    log("Context window: real 2 preceding words, matching trigram model")
    log("STATUS: PILOT / PRELIMINARY")

    if not check_ollama():
        log("Aborting: Ollama check failed.")
        raise SystemExit(1)

    sentences, word_to_location = build_sentence_lookup(CLEAN_INDEX)

    log("\nLoading dataset_reparandum.csv...")
    df = pd.read_csv(REP_CSV)
    log(f"  Total rows available: {len(df):,}")

    random.seed(RANDOM_SEED)
    sample_idx = random.sample(range(len(df)), min(SAMPLE_SIZE, len(df)))
    sample = df.iloc[sample_idx].reset_index(drop=True)
    log(f"  Sampled: {len(sample):,} rows")

    # ── TEST BATCH FIRST -- mandatory gate before full run ─────
    log_section(f"TEST BATCH -- verifying on first {TEST_BATCH} words")
    test_results = []
    for i in range(min(TEST_BATCH, len(sample))):
        row = sample.iloc[i]
        word_id = row["word_id"]
        word    = str(row["word"])
        w_km2, w_km1, loc_status = get_preceding_context(word_id, sentences, word_to_location)

        if loc_status != "ok":
            log(f"  [{i+1}] word_id={word_id} word='{word}' -- SKIPPED: {loc_status}")
            continue

        surp, model_token, status = get_word_surprisal(w_km2, w_km1, word)
        log(f"  [{i+1}] context='{w_km2} {w_km1}' target='{word}' "
            f"-> generated='{model_token}' surprisal={surp} status={status}")
        if surp is not None:
            test_results.append(surp)

    if not test_results:
        log("\n  ERROR: no valid results in test batch. Stopping.")
        raise SystemExit(1)

    test_mean = sum(test_results) / len(test_results)
    test_min  = min(test_results)
    test_max  = max(test_results)
    log(f"\n  Test batch stats: mean={test_mean:.4f}  min={test_min:.4f}  max={test_max:.4f} bits")
    log(f"  For reference, trigram model mean was ~15.10 bits.")

    # Plausibility gate: require BOTH a reasonable mean AND real
    # variance across different contexts (v1/v2 failure mode was
    # near-identical low values regardless of context).
    test_range = test_max - test_min
    if test_mean < 1.0 or test_range < 0.5:
        log("\n  *** WARNING: results still look degenerate")
        log(f"  *** (mean={test_mean:.4f}, range={test_range:.4f})")
        log("  *** This suggests the raw-mode fix did not fully resolve the issue.")
        log("  *** STOPPING -- not proceeding to full run.")
        log("  *** Manual review of the test batch log lines above is needed")
        log("  *** before any further attempt.")
        raise SystemExit(1)
    else:
        log("\n  Test batch shows plausible mean AND real variance across contexts.")
        log("  Proceeding to full sample run.")

    # ── FULL SAMPLE RUN ────────────────────────────────────────
    log_section("FULL SAMPLE RUN")
    results = []
    n_ok = 0
    n_fail = 0
    n_mismatch = 0

    for i, row in sample.iterrows():
        word_id = row["word_id"]
        word    = str(row["word"])
        w_km2, w_km1, loc_status = get_preceding_context(word_id, sentences, word_to_location)

        if loc_status != "ok":
            results.append({"word_id": word_id, "word": word, "llm_surp": None,
                             "model_token": None, "status": loc_status})
            n_fail += 1
            continue

        surp, model_token, status = get_word_surprisal(w_km2, w_km1, word)

        if surp is not None:
            n_ok += 1
            if status == "no_match":
                n_mismatch += 1
        else:
            n_fail += 1

        results.append({
            "word_id": word_id,
            "word": word,
            "context_prev2": w_km2,
            "context_prev1": w_km1,
            "llm_surp": round(surp, 6) if surp is not None else None,
            "model_generated_token": model_token,
            "status": status,
        })

        if (i + 1) % 50 == 0:
            log(f"  {i+1}/{len(sample)} processed... (ok={n_ok}, fail={n_fail}, mismatch={n_mismatch})")

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    log_section("PILOT RUN COMPLETE (v3)")
    log(f"Saved -> {OUTPUT_CSV}")
    log(f"  Total processed: {len(result_df):,}")
    log(f"  Successful: {n_ok:,}  Failed: {n_fail:,}  Token mismatch: {n_mismatch:,}")
    if n_ok > 0:
        valid = result_df[result_df["llm_surp"].notna()]
        log(f"  Mean LLM surprisal: {valid['llm_surp'].mean():.4f} bits")
        log(f"  Min: {valid['llm_surp'].min():.4f}  Max: {valid['llm_surp'].max():.4f}")
        log(f"  Std dev: {valid['llm_surp'].std():.4f}")
        log(f"  (trigram model: mean 15.10, for comparison)")


if __name__ == "__main__":
    main()