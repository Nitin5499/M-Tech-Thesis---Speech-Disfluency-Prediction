"""
01_parse_xml.py
Parses the Switchboard NXT corpus XML files.

Produces:
  data/processed/words_labeled.csv
    columns: conv_id, speaker, word_id, local_id, word, pos,
             start, end, duration, label

Labels:
  filler     - words with pos=UH and word in filler vocabulary
  reparandum - words inside a <reparandum> element
  fluent     - everything else
"""

import os
import re
import pandas as pd
from lxml import etree

# ─────────────────────────────────────────────────────────────
# PATHS  — update XML_ROOT if your dataset is in a different location
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

XML_ROOT = os.path.join(
    ROOT_DIR, "data", "switchboard-nxt-release-1.0", "xml"
)

TERMINALS_DIR  = os.path.join(XML_ROOT, "terminals")
DISFLUENCY_DIR = os.path.join(XML_ROOT, "disfluency")
OUTPUT_DIR     = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_CSV     = os.path.join(OUTPUT_DIR, "words_labeled.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# NITE XML namespace
NS = "http://nite.sourceforge.net/"

# Filler vocabulary — words counted as disfluent fillers
FILLER_WORDS = {
    "uh", "um", "uh-huh", "uhhuh", "mm", "mmm",
    "hm", "hmm", "huh", "ah", "oh", "er", "erm",
    "ugh", "mhm", "uhm",
}


# ─────────────────────────────────────────────────────────────
# Helper: extract word id from href
# 'sw2005.A.terminals.xml#id(s2_21)' -> 'sw2005.A.s2_21'
# ─────────────────────────────────────────────────────────────
def href_to_global_id(href):
    if not href:
        return None
    m = re.search(r"#id\(([^)]+)\)", href)
    if not m:
        return None
    local_id  = m.group(1)
    file_part = href.split("#")[0]
    prefix    = file_part.replace(".terminals.xml", "")
    return f"{prefix}.{local_id}"


# ─────────────────────────────────────────────────────────────
# STEP 1  Parse terminal files → every word with timing
# ─────────────────────────────────────────────────────────────
def parse_terminals(terminals_dir):
    records = []
    files   = sorted(f for f in os.listdir(terminals_dir)
                     if f.endswith(".xml"))
    total   = len(files)
    print(f"Found {total} terminal files. Parsing...")

    for idx, fname in enumerate(files):
        base  = fname.replace(".terminals.xml", "")
        parts = base.split(".")
        if len(parts) < 2:
            continue
        conv_id     = parts[0]
        speaker     = parts[1]
        file_prefix = f"{conv_id}.{speaker}"

        fpath = os.path.join(terminals_dir, fname)
        try:
            tree = etree.parse(fpath)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")
            continue

        root = tree.getroot()

        for elem in root:
            local_tag = elem.tag.split("}")[-1]
            if local_tag != "word":
                continue

            local_id  = elem.get(f"{{{NS}}}id", "")
            global_id = f"{file_prefix}.{local_id}"
            orth = elem.get("orth", "").strip()
            word_norm = orth.lower()
            pos       = elem.get("pos",  "UNK")
            start     = elem.get(f"{{{NS}}}start")
            end       = elem.get(f"{{{NS}}}end")

            if not orth or orth == "*":
                continue

            try:
                start = float(start) if start is not None else None
                end   = float(end)   if end   is not None else None
            except (ValueError, TypeError):
                start, end = None, None

            duration = (
                round(end - start, 6)
                if start is not None and end is not None and end > start
                else None
            )

            records.append({
                "conv_id":  conv_id,
                "speaker":  speaker,
                "word_id":  global_id,
                "local_id": local_id,
                "word":     word_norm,
                "pos":      pos,
                "start":    start,
                "end":      end,
                "duration": duration,
            })

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{total} done...")

    df = pd.DataFrame(records)
    print(f"\nTotal word tokens: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2  Parse disfluency files → reparandum word ids
# ─────────────────────────────────────────────────────────────
def parse_reparandums(disfluency_dir):
    reparandum_ids = set()
    files = sorted(f for f in os.listdir(disfluency_dir)
                   if f.endswith(".xml"))
    total = len(files)
    print(f"\nFound {total} disfluency files. Parsing...")

    for idx, fname in enumerate(files):
        fpath = os.path.join(disfluency_dir, fname)
        try:
            tree = etree.parse(fpath)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")
            continue

        root = tree.getroot()

        for rep in root.iter("reparandum"):
            for child in rep:
                local_tag = child.tag.split("}")[-1]
                if local_tag == "child":
                    href = child.get("href", "")
                    gid  = href_to_global_id(href)
                    if gid:
                        reparandum_ids.add(gid)

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{total} done...")

    print(f"  Reparandum word IDs: {len(reparandum_ids):,}")
    return reparandum_ids


# ─────────────────────────────────────────────────────────────
# STEP 3  Assign labels
# ─────────────────────────────────────────────────────────────
def assign_labels(df, reparandum_ids):
    def label(row):
        if row["word_id"] in reparandum_ids:
            return "reparandum"
        if (row["pos"] == "UH" and
                row["word"].lower() in FILLER_WORDS):
            return "filler"
        return "fluent"

    df["label"] = df.apply(label, axis=1)
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("Step 1: Parse Switchboard NXT XML")
    print("=" * 55)

    print("\n--- Parsing terminals ---")
    df = parse_terminals(TERMINALS_DIR)

    print("\n--- Parsing disfluencies ---")
    reparandum_ids = parse_reparandums(DISFLUENCY_DIR)

    print("\n--- Assigning labels ---")
    df = assign_labels(df, reparandum_ids)

    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nPaper targets:")
    print("  fluent:     ~558,361")
    print("  reparandum:  ~14,520")
    print("  filler:      ~12,050")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")

    print("\n" + "=" * 55)
    print("Step 1 COMPLETE")
    print("=" * 55)