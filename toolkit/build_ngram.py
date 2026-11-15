"""
build_ngram.py

Builds the trigram counts once and saves them, so the toolkit does not have to
read OANC every time it starts. Same corpus, same tokenisation and same counting
as src/02_lexical_surprisal.py (it imports that file's loader), just stored as
plain dicts so they can be pickled.

run from the project folder:  python toolkit/build_ngram.py
out: toolkit/models/ngram_counts.pkl   (takes a few minutes, file is large)
"""

import os
import sys
import pickle
import importlib.util
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# 02_lexical_surprisal.py starts with a digit so it can't be imported normally
spec = importlib.util.spec_from_file_location("lexsurp", os.path.join(ROOT, "src", "02_lexical_surprisal.py"))
lexsurp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lexsurp)

sentences = lexsurp.load_oanc_text(lexsurp.OANC_DIR)

print("counting trigrams...")
tri = Counter()
bi = Counter()
vocab = set()
for sent in sentences:
    padded = ["<s>", "<s>"] + [sys.intern(t) for t in sent] + ["</s>"]
    for i in range(2, len(padded)):
        w0, w1, w2 = padded[i - 2], padded[i - 1], padded[i]
        tri[(w0, w1, w2)] += 1
        bi[(w0, w1)] += 1
        vocab.add(w2)

V = len(vocab)
print(f"trigrams: {len(tri):,} | bigram contexts: {len(bi):,} | vocabulary: {V:,}")

out_path = os.path.join(OUT_DIR, "ngram_counts.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"tri": dict(tri), "bi": dict(bi), "V": V}, f, protocol=4)
print("saved ->", out_path)