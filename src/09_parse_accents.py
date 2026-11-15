"""
09_parse_accents.py
====================
PROSODIC EXTENSION -- Stage 2 of 7.

Parses the Switchboard NXT accents XML files into a single flat CSV.
Mirrors 07_parse_phonwords.py's structure exactly.

INPUT:
  data/switchboard-nxt-release-1.0/xml/accent/*.accents.xml
  (one file per conversation-speaker pair, e.g. sw2018.A.accents.xml)

OUTPUT:
  data/processed/accents_index.csv
    columns: conv_id, speaker, accent_id, phonword_id, strength, type,
             accent_start

NOTES ON THE XML STRUCTURE (from inspected samples):
  - Each <accent> has nite:id, nite:start/nite:end (identical values
    in every sample seen -- the accent peak is a single instant, not
    a span), strength ("full" or "weak"), and type ("nuclear",
    "plain", "pre-nuclear" -- seen values so far).
  - The link to the actual word is via a <nite:pointer role="at"
    href="....phonwords.xml#id(...)"/> child element, NOT an
    attribute on <accent> itself -- the phonword_id has to be
    extracted from that child pointer's href.
  - Not every phonword has an accent -- this file only contains
    entries for phonwords that WERE accented, so absence from this
    table (once merged) means "not accented", not "missing data".

Author: Nitin Kumar
Date: August 2026 (prosodic extension, stage 2)
"""

import os
import re
import pandas as pd
from lxml import etree
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR

ACCENTS_DIR = os.path.join(
    ROOT_DIR, "data", "switchboard-nxt-release-1.0", "xml", "accent"
)

OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "accents_index.csv")
LOG_FILE   = os.path.join(OUTPUT_DIR, "pipeline_log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

NS = "http://nite.sourceforge.net/"


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


def extract_phonword_id(href):
    """
    Extracts the phonword id from an href like:
    'sw2018.A.phonwords.xml#id(ms2B_pw1)' -> 'ms2B_pw1'
    Returns None if the pattern doesn't match.
    """
    if not href:
        return None
    m = re.search(r"#id\(([^)]+)\)", href)
    return m.group(1) if m else None


def parse_accents(accents_dir):
    records = []
    n_no_pointer = 0
    n_unparsed_href = 0

    if not os.path.isdir(accents_dir):
        raise FileNotFoundError(
            f"Accents directory not found: {accents_dir}\n"
            f"Check the ACCENTS_DIR path against your actual folder layout."
        )

    files = sorted(f for f in os.listdir(accents_dir)
                   if f.endswith(".accents.xml"))
    total = len(files)
    log(f"Found {total} accent files. Parsing...")

    if total == 0:
        log("  WARNING: zero files found -- check ACCENTS_DIR and file naming pattern.")

    for idx, fname in enumerate(files):
        base  = fname.replace(".accents.xml", "")
        parts = base.split(".")
        if len(parts) < 2:
            log(f"  [SKIP] {fname}: unexpected filename pattern")
            continue
        conv_id = parts[0]
        speaker = parts[1]

        fpath = os.path.join(accents_dir, fname)
        try:
            tree = etree.parse(fpath)
        except Exception as e:
            log(f"  [SKIP] {fname}: {e}")
            continue

        root = tree.getroot()

        for elem in root:
            # Skip comment nodes (elem.tag is a callable, not a string,
            # for XML comments) -- seen in some accent files' headers,
            # e.g. "<!-- File generated from the EDINBURGH CONVERTED
            # files. -->".
            if not isinstance(elem.tag, str):
                continue

            local_tag = elem.tag.split("}")[-1]
            if local_tag != "accent":
                continue

            accent_id = elem.get(f"{{{NS}}}id", "")
            strength  = elem.get("strength", None)
            acc_type  = elem.get("type", None)
            start     = elem.get(f"{{{NS}}}start")

            # Find the <nite:pointer role="at" href="..."/> child
            phonword_id = None
            found_pointer = False
            for child in elem:
                child_tag = child.tag.split("}")[-1]
                if child_tag == "pointer" and child.get("role") == "at":
                    found_pointer = True
                    href = child.get(f"{{{NS}}}href", child.get("href", ""))
                    phonword_id = extract_phonword_id(href)
                    if phonword_id is None:
                        n_unparsed_href += 1
                    break

            if not found_pointer:
                n_no_pointer += 1
                continue

            try:
                start_f = float(start) if start is not None else None
            except (ValueError, TypeError):
                start_f = None

            records.append({
                "conv_id":     conv_id,
                "speaker":     speaker,
                "accent_id":   accent_id,
                "phonword_id": phonword_id,
                "strength":    strength,
                "type":        acc_type,
                "accent_start": start_f,
            })

        if (idx + 1) % 100 == 0:
            log(f"  {idx+1}/{total} files done...")

    log(f"  Done -- all {total} files processed")
    log(f"  Accents with no <nite:pointer> child at all: {n_no_pointer:,}")
    log(f"  Accents whose href didn't match expected pattern: {n_unparsed_href:,}")

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    log_section("Prosodic Extension -- Stage 2: Parse Accents")
    log(f"Reading from: {ACCENTS_DIR}")

    df = parse_accents(ACCENTS_DIR)

    log(f"\nTotal accent tokens parsed: {len(df):,}")
    log(f"Conversations covered: {df['conv_id'].nunique():,}")

    n_null_phonword = df['phonword_id'].isna().sum()
    log(f"Rows with no resolved phonword_id: {n_null_phonword:,}")

    log("\nStrength value distribution:")
    log(df['strength'].value_counts(dropna=False).to_string())
    log("\nType value distribution:")
    log(df['type'].value_counts(dropna=False).to_string())

    dup_check = df.duplicated(subset=['conv_id', 'speaker', 'phonword_id']).sum()
    log(f"\nRows sharing the same (conv_id, speaker, phonword_id) -- i.e. a phonword "
        f"accented more than once in this file: {dup_check:,}")
    if dup_check > 0:
        log("  NOTE: not necessarily an error -- investigate a few examples before "
            "assuming this is safe to average/collapse in the bridging stage.")

    df.to_csv(OUTPUT_CSV, index=False)

    log_section("Stage 2 COMPLETE")
    log(f"Saved -> {OUTPUT_CSV}")
    log("\nNext: run 10_parse_breaks.py, then 08_bridge_terminals_phonwords.py")