"""
test_gpt2_surprisal.py
========================
FIRST VALIDATION SCRIPT -- not part of the numbered pipeline yet.

Purpose: confirm GPT-2 loads correctly, and validate the surprisal
computation approach on a REAL sentence from the corpus (not a toy
example), before committing to the full pipeline architecture.

DESIGN DECISIONS BEING TESTED HERE (confirm both before scaling up):
  1. GPT-2 uses its FULL available left-context within a sentence,
     not a forced 2-word window -- unlike the trigram model, which
     is limited to 2 words purely by its own technical constraints,
     not by design choice. Testing GPT-2 at its full natural
     capability is the fair comparison.
  2. Context RESETS at each sentence boundary, matching how every
     other feature in the pipeline (lexical surprisal, DLT costs,
     syntactic surprisal) already resets per sentence -- for
     consistency across the whole feature set, not because GPT-2
     itself requires it.

WHY THIS AVOIDS THE ORIGINAL PILOT'S FAILURE MODE:
  The earlier LLM surprisal attempt used Llama 3.2 via Ollama, an
  INSTRUCTION-TUNED chat model -- which responds conversationally
  rather than performing genuine next-token completion, making
  surprisal values meaningless. GPT-2 via Hugging Face Transformers
  is a genuine base/completion model, built specifically for the
  next-token-prediction task surprisal computation actually needs.

Author: Nitin Kumar
Date: August 2026 (GPT-2 surprisal, validation stage)
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math

print("Loading GPT-2 tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()  # inference mode -- disables dropout etc.
print("Loaded successfully.\n")


def compute_word_surprisal(sentence_words):
    """
    Given a list of words (already tokenized at the WORD level, e.g.
    from clean_word_index.csv), computes surprisal for each word
    using GPT-2's own BPE tokenizer internally, then re-aggregates
    back to word-level surprisal values.

    Returns a list of (word, surprisal_bits) tuples, one per input
    word. The first word's surprisal is computed using GPT-2's
    special beginning-of-sequence handling (no real left-context
    available, same situation as the trigram model's "<s> <s>"
    padding).
    """
    # Prepend GPT-2's own beginning-of-sequence token, so the FIRST
    # real word also has something to condition its prediction on --
    # directly analogous to the trigram model's "<s> <s>" padding at
    # the start of every sentence. Without this, the first word would
    # have no surprisal value at all (exactly what happened in the
    # first test run: 18 surprisals for 19 words).
    bos = tokenizer.bos_token  # GPT-2's special start-of-text token
    text = bos + " " + " ".join(sentence_words)

    # Tokenize the whole sentence at once, keeping track of which
    # token(s) came from which original word via the tokenizer's
    # own offset mapping.
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"][0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"  Original words ({len(sentence_words)}): {sentence_words}")
    print(f"  GPT-2 subword tokens, including BOS ({input_ids.shape[1]}): {tokens}")
    print(f"  --> Expect token count = word count + 1 now (the +1 is the BOS")
    print(f"      token), UNLESS a word splits into multiple subword pieces --")
    print(f"      watch for any real word appearing broken into fragments.\n")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)

    log_probs = torch.log_softmax(logits, dim=-1)

    # Now start from i=1 as before, but since position 0 is BOS, this
    # loop's first iteration (i=1) computes the surprisal of the
    # FIRST REAL WORD, using BOS as its context -- closing the gap
    # from the previous run.
    token_surprisals = []
    for i in range(1, input_ids.shape[1]):
        actual_token_id = input_ids[0, i].item()
        predicted_log_prob = log_probs[0, i - 1, actual_token_id].item()
        surprisal_bits = -predicted_log_prob / math.log(2)  # convert nats -> bits
        token_surprisals.append(surprisal_bits)

    return token_surprisals, offsets, tokens


if __name__ == "__main__":
    # Test 1 -- the familiar Integration Cost worked-example sentence,
    # now with the BOS-token fix applied.
    real_sentence = [
        "first", "i", "need", "to", "know", "how", "do", "you",
        "feel", "about", "sending", "an", "elderly", "family",
        "member", "to", "a", "nursing", "home"
    ]

    print("=" * 60)
    print("TEST 1: GPT-2 surprisal, with BOS-token fix")
    print("=" * 60)
    surprisals, offsets, tokens = compute_word_surprisal(real_sentence)

    print("Word-aligned surprisal values (bits):")
    for word, s in zip(real_sentence, surprisals):
        print(f"    {word:12} {s:.3f}")
    print(f"\nTotal words: {len(real_sentence)}, total surprisal values: {len(surprisals)}")
    print("These should now match exactly -- every real word has a value.\n")

    # Test 2 -- IMPORTANT CORRECTION: GPT-2 surprisal must be computed
    # on the CLEAN, disfluency-free corpus, exactly like every other
    # feature in this pipeline (lexical surprisal, DLT costs,
    # syntactic surprisal all run on clean_word_index.csv, never on
    # raw text with fillers still present -- this was an explicit
    # instruction from the supervisor, and matches the pipeline's
    # existing design principle throughout).
    #
    # An EARLIER version of this test mistakenly fed GPT-2 a sentence
    # WITH fillers still in it ("okay uh first um i need..."), which
    # would have made GPT-2 condition its predictions on "uh"/"um" as
    # if they were real context words -- inconsistent with how every
    # other feature in the pipeline is computed, and producing
    # surprisal values that could not be fairly compared against them.
    #
    # This corrected version uses the CLEAN version of the same
    # sentence (fillers removed, exactly as 03a_clean_corpus.py would
    # produce) -- this is what the real pipeline will actually do.
    clean_sentence = [
        "okay", "first", "i", "need", "to", "how", "do", "you",
        "feel", "about", "about", "sending", "an", "elderly",
        "family", "member"
    ]

    print("=" * 60)
    print("TEST 2 (CORRECTED): GPT-2 surprisal on the CLEAN version")
    print("(fillers removed, matching clean_word_index.csv)")
    print("=" * 60)
    surprisals2, offsets2, tokens2 = compute_word_surprisal(clean_sentence)

    print("Word-aligned surprisal values (bits):")
    for word, s in zip(clean_sentence, surprisals2):
        print(f"    {word:12} {s:.3f}")

    print(f"\nTotal words: {len(clean_sentence)}, total surprisal values: {len(surprisals2)}")
    print("\nThese are the values that should actually be used -- computed the")
    print("same way every other feature in the pipeline is computed: clean")
    print("text only. The filler words themselves will get their features via")
    print("PROPAGATION afterward, reusing the existing propagate_feature()")
    print("logic -- never by computing GPT-2 surprisal directly on a filler.")