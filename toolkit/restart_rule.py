"""
restart_rule.py -- repetition rule for restarts (separate from the statistical model).

A restart often repeats the words that came just before the break:
    "i went to the right to the left"   ->  reparandum "to the right", repair "to the left"
The rule looks for a short phrase (1 to 3 words) that comes back soon after,
with at most one other word in between (editing terms like "sorry" do not count).
It only looks at the typed words, so it can mark a restart that is already in the text.
It does not touch the model probabilities.
"""

EDIT_WORDS = {"sorry", "uh", "um", "er", "erm", "oops", "no", "rather", "well"}
# an explicit editing term makes a restart clear, so a single repeated word may come back a little later
STRONG_EDIT = {"sorry", "oops", "rather", "uh", "um", "er", "erm"}


def find_restarts(words, max_len=3, max_extra=1):
    """words: list of lowercase tokens.
    returns a list of dicts: reparandum (start, end), edit (list of idx), repair (start, end)
    with end exclusive."""
    w = [x.lower() for x in words]
    out = []
    i = 0
    while i < len(w):
        found = None
        if w[i] in EDIT_WORDS:
            i += 1
            continue
        for n in range(max_len, 0, -1):
            phrase = w[i:i + n]
            if len(phrase) < n:
                continue
            extra = 0
            edits = []
            p = i + n
            limit = 0 if n == 1 else max_extra     # a single word must repeat straight away
            while p < len(w):
                if w[p] in EDIT_WORDS:
                    edits.append(p)
                    p += 1
                    continue
                if w[p:p + n] == phrase and w[p] not in EDIT_WORDS:
                    if n > 1 or extra == 0 or any(w[e] in STRONG_EDIT for e in edits):
                        found = (n, extra, edits, p)
                    break
                extra += 1
                if extra > (max_extra + 1 if n == 1 else limit):
                    break
                p += 1
            if found:
                break
        if found:
            n, extra, edits, p = found
            rep_end = edits[0] if edits else p
            out.append({"reparandum": (i, rep_end), "edit": edits,
                        "repair": (p, min(len(w), p + n + extra))})
            i = p + n
        else:
            i += 1
    return out


if __name__ == "__main__":
    tests = [
        "i went to the right to the left",
        "i went to the righ sorry to the left",
        "i went to the i went to the store",
        "we saw the the dog",
        "i went to the store and to the bank",
        "we saw the dog and the cat",
        "so i went to the store and then we left",
        "and then she moved to a small town near the coast",
        "she gave it to him to her brother",
        "we drove to boston sorry to new york for the weekend",
        "she gave the book to john sorry to mary yesterday",
        "he bought a red a blue car last week",
        "i went to him and to her",
        "we saw the dog sorry the cat",
    ]
    for t in tests:
        ws = t.split()
        r = find_restarts(ws)
        shown = []
        for k, x in enumerate(r):
            a, b = x["reparandum"]; c, d = x["repair"]
            shown.append(f"reparandum=[{' '.join(ws[a:b])}] edit={[ws[e] for e in x['edit']]} repair=[{' '.join(ws[c:d])}]")
        print(f"{t!r:55} -> {shown or 'none'}")