"""
disfluency_toolkit.py

Type a sentence, get the chance of a filler / reparandum after every word.

For each sentence the toolkit builds the same features as the thesis pipeline:
  1. trigram surprisal  (add-one smoothed trigram trained on OANC, as in src/02)
  2. GPT-2 surprisal    (per-word tokenisation, as in src/14)
  3. DLT integration cost and storage cost from spaCy (as in src/04)
Then it standardises them with the training mean / sd and applies the logistic
model coefficients exported by src/24_export_toolkit_models.R.

A "gap" is the slot between word i and word i+1:
  prev_* features = word i, foll_* features = word i+1.

usage:
  python toolkit/disfluency_toolkit.py "so i went to the store and then we left"
  python toolkit/disfluency_toolkit.py --check     (compares with the pipeline files)
"""

import os
import re
import sys
import math
import pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(HERE, "models")
PROCESSED = os.path.join(ROOT, "data", "processed")

FEATURES = ["prev_lex_surp", "foll_lex_surp", "prev_gpt2_surp", "foll_gpt2_surp",
            "prev_IC", "foll_IC", "prev_SC", "foll_SC"]
TASKS = ["filler", "reparandum"]

# which word carries the random intercept (same as the thesis GLMM):
#   filler     -> word_for_model of the filler dataset
#   reparandum -> the reparandum word itself, i.e. the word before the gap
# for the filler the word is taken from the side given here ("after" or "before" the gap)
FILLER_EFFECT_ON = "after"


def split_sentences(text):
    """lowercase, drop punctuation, split into sentences on . ? ! ;"""
    sents = []
    for chunk in re.split(r"[.?!;]+", text):
        words = re.findall(r"[a-z0-9']+", chunk.lower())
        if words:
            sents.append(words)
    return sents


class Toolkit:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self._ngram = None
        self._gpt2 = None
        self._nlp = None
        self._load_models()

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---------------- loading ----------------
    def _load_models(self):
        self.models = {}
        for task in TASKS:
            path = os.path.join(MODELS, f"model_{task}.csv")
            m = pd.read_csv(path).set_index("term")
            self.models[task] = m
        self.meta = pd.read_csv(os.path.join(MODELS, "model_meta.csv")).set_index("task")
        # random intercept of every training word (new words get 0, like in the GLMM CV)
        self.word_eff = {}
        for task in TASKS:
            p = os.path.join(MODELS, f"wordeffects_{task}.csv")
            if os.path.exists(p):
                w = pd.read_csv(p, keep_default_na=False)
                self.word_eff[task] = dict(zip(w["word"].astype(str), w["effect"]))
            else:
                self.word_eff[task] = {}

    def _load_ngram(self):
        if self._ngram is None:
            self._say("loading trigram counts (this takes a minute)...")
            with open(os.path.join(MODELS, "ngram_counts.pkl"), "rb") as f:
                self._ngram = pickle.load(f)
        return self._ngram

    def _load_gpt2(self):
        if self._gpt2 is None:
            self._say("loading GPT-2...")
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            tok = GPT2Tokenizer.from_pretrained("gpt2")
            mdl = GPT2LMHeadModel.from_pretrained("gpt2")
            mdl.eval()
            self._gpt2 = (tok, mdl, torch)
        return self._gpt2

    def _load_spacy(self):
        if self._nlp is None:
            self._say("loading spaCy...")
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    # ---------------- features ----------------
    def ngram_surprisal(self, word, c2, c1):
        ng = self._load_ngram()
        num = ng["tri"].get((c2, c1, word), 0) + 1
        den = ng["bi"].get((c2, c1), 0) + ng["V"]
        return -math.log2(num / den)

    def ngram_features(self, words):
        """prev = surprisal of word i, foll = surprisal of word i+1 (same as src/02)"""
        n = len(words)
        prev_s, foll_s = [], []
        for i in range(n):
            w_km2 = words[i - 2] if i >= 2 else "<s>"
            w_km1 = words[i - 1] if i >= 1 else "<s>"
            prev_s.append(self.ngram_surprisal(words[i], w_km2, w_km1))
            foll_s.append(self.ngram_surprisal(words[i + 1], w_km1, words[i]) if i < n - 1 else np.nan)
        return prev_s, foll_s

    def gpt2_surprisals(self, words):
        """per-word GPT-2 surprisal in bits, subword pieces summed (same as src/14)"""
        tok, mdl, torch = self._load_gpt2()
        ids = [tok.bos_token_id]
        counts = []
        for w in words:
            piece = tok.encode(" " + w, add_special_tokens=False)
            if len(piece) == 0:
                piece = tok.encode(w, add_special_tokens=False)
            counts.append(len(piece))
            ids.extend(piece)
        x = torch.tensor([ids])
        with torch.no_grad():
            logits = mdl(x).logits
        logp = torch.log_softmax(logits, dim=-1)
        tok_s = [-logp[0, i - 1, ids[i]].item() / math.log(2) for i in range(1, len(ids))]
        out, cur = [], 0
        for c in counts:
            out.append(sum(tok_s[cur:cur + c]))
            cur += c
        return out

    def dlt_costs(self, words):
        """integration and storage cost per word (same as src/04)"""
        nlp = self._load_spacy()
        n = len(words)
        text = " ".join(str(w) for w in words if w and str(w).strip())
        try:
            doc = nlp(text)
            IC, SC = [], []
            for i, token in enumerate(doc):
                ic = 1
                for dep_tok in doc:
                    if dep_tok.head.i == i and dep_tok.i != i:
                        ic += abs(dep_tok.i - i)
                IC.append(float(ic))
                sc = 1
                for dep_tok in doc[:i + 1]:
                    if dep_tok.head.i > i:
                        sc += 1
                SC.append(float(sc))
        except Exception:
            IC, SC = [1.0] * n, [1.0] * n
        while len(IC) < n:
            IC.append(1.0)
            SC.append(1.0)
        return IC[:n], SC[:n]

    def gap_features(self, words):
        """one row per gap between neighbouring words of one sentence"""
        lex_prev, lex_foll = self.ngram_features(words)
        g = self.gpt2_surprisals(words)
        ic, sc = self.dlt_costs(words)
        rows = []
        for i in range(len(words) - 1):
            rows.append({
                "prev_lex_surp": lex_prev[i], "foll_lex_surp": lex_foll[i],
                "prev_gpt2_surp": g[i], "foll_gpt2_surp": g[i + 1],
                "prev_IC": ic[i], "foll_IC": ic[i + 1],
                "prev_SC": sc[i], "foll_SC": sc[i + 1],
            })
        return pd.DataFrame(rows, columns=FEATURES)

    # ---------------- model ----------------
    def word_effect(self, words, task):
        """random intercept of the word that belongs to each gap (0 if the word is new)"""
        eff = self.word_eff.get(task, {})
        if task == "filler" and FILLER_EFFECT_ON == "after":
            keys = words[1:]
        else:
            keys = words[:-1]
        return np.array([eff.get(k, 0.0) for k in keys], dtype=float)

    def score(self, feats, task, words=None):
        """logistic model: standardise, weighted sum (+ word effect), sigmoid"""
        m = self.models[task]
        z = m["estimate"]["(Intercept)"] * np.ones(len(feats))
        for f in FEATURES:
            z = z + m["estimate"][f] * (feats[f].values - m["mean"][f]) / m["sd"][f]
        if words is not None:
            z = z + self.word_effect(words, task)
        return 1.0 / (1.0 + np.exp(-z))

    def accuracy(self, task):
        """cross-validated accuracy to show next to the predictions"""
        r = self.meta.loc[task]
        return float(r["cv_accuracy_word"]) if "cv_accuracy_word" in r.index else float(r["cv_accuracy"])

    def analyze(self, text):
        """returns one row per gap with the probability of a filler / reparandum"""
        out = []
        for s_no, words in enumerate(split_sentences(text), 1):
            if len(words) < 2:
                continue
            feats = self.gap_features(words)
            p_fill = self.score(feats, "filler", words)
            p_rep = self.score(feats, "reparandum", words)
            for i in range(len(words) - 1):
                row = {"sentence": s_no, "gap": i + 1,
                       "word_before": words[i], "word_after": words[i + 1],
                       "P_filler": float(p_fill[i]), "P_reparandum": float(p_rep[i])}
                row.update({f: float(feats[f].iloc[i]) for f in FEATURES})
                out.append(row)
        return pd.DataFrame(out)

    # ---------------- check against the pipeline files ----------------
    def check(self, n_sent=3):
        """recompute the features for real sentences and compare with the pipeline CSVs"""
        idx = pd.read_csv(os.path.join(PROCESSED, "clean_word_index.csv"))
        lex = pd.read_csv(os.path.join(PROCESSED, "lexical_surprisal.csv")).set_index("word_id")
        dlt = pd.read_csv(os.path.join(PROCESSED, "dlt_costs.csv")).set_index("word_id")
        gpt_path = os.path.join(PROCESSED, "gpt2_surprisal.csv")
        gpt = pd.read_csv(gpt_path).set_index("word_id") if os.path.exists(gpt_path) else None

        def maxdiff(mine, ref):
            a, b = np.array(mine, dtype=float), np.array(ref, dtype=float)
            ok = ~(np.isnan(a) | np.isnan(b))
            return float(np.abs(a[ok] - b[ok]).max()) if ok.any() else float("nan")

        shown = 0
        for sent_idx, grp in idx.groupby("sent_idx", sort=True):
            grp = grp.sort_values("word_pos")
            raw = grp["word"].astype(str).tolist()
            if not 6 <= len(raw) <= 15:
                continue
            ids = grp["word_id"].tolist()
            words = [w.lower() for w in raw]
            pl, fl = self.ngram_features(words)
            ic, sc = self.dlt_costs(raw)
            g = self.gpt2_surprisals(raw)
            print(f"\nsentence {sent_idx}: {' '.join(raw)}")
            print("  prev_lex_surp  max diff:", maxdiff(pl, lex.loc[ids, "prev_lex_surp"]))
            print("  foll_lex_surp  max diff:", maxdiff(fl, lex.loc[ids, "foll_lex_surp"]))
            print("  prev_IC        max diff:", maxdiff(ic, dlt.loc[ids, "prev_IC"]))
            print("  prev_SC        max diff:", maxdiff(sc, dlt.loc[ids, "prev_SC"]))
            if gpt is not None:
                print("  prev_gpt2_surp max diff:", maxdiff(g, gpt.loc[ids, "prev_gpt2_surp"]))
            shown += 1
            if shown >= n_sent:
                break
        print("\n(differences near 0 mean the toolkit reproduces the pipeline features)")


if __name__ == "__main__":
    tk = Toolkit()
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        tk.check()
    else:
        text = " ".join(sys.argv[1:]) or "so i went to the store and then we got the car"
        res = tk.analyze(text)
        pd.set_option("display.width", 160)
        print(res[["sentence", "gap", "word_before", "word_after", "P_filler", "P_reparandum"]]
              .round(3).to_string(index=False))
        for t in TASKS:
            r = tk.meta.loc[t]
            print(f"{t}: held-out accuracy of this text-only model {tk.accuracy(t):.2f}% "
                  f"(always-guess {r.always_guess:.2f}%, {int(r.n_rows):,} rows)")