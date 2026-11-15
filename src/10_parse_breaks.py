"""
10_parse_breaks.py
===================
PROSODIC EXTENSION -- Stage 3 of 7.

Parses the Switchboard NXT breaks XML files into a single flat CSV.
Mirrors 07_parse_phonwords.py and 09_parse_accents.py's structure.

INPUT:
  data/switchboard-nxt-release-1.0/xml/breaks/*.breaks.xml
  (one file per conversation-speaker pair, e.g. sw2018.A.breaks.xml)

OUTPUT:
  data/processed/breaks_index.csv
    columns: conv_id, speaker, break_id, phonword_id, break_index,
             phrase_tone, boundary_tone, break_start

NOTES ON THE XML STRUCTURE (from inspected samples):
  - Each <break> has nite:id, nite:start/nite:end (identical in every
    sample seen), an "index" attribute (the ToBI break strength --
    values seen: "0","1","1p","2","2p","3","3-","4","4-","X"), and
    optional "phraseTone" / "boundaryTone" attributes (H/L/!H, only
    present on stronger breaks).
  - Also has a "UWtime" attribute (a second timestamp from the UW
    prosody annotation effort) -- NOT used here; nite:start is kept
    as the canonical timestamp to stay consistent with every other
    script in this pipeline.
  - Same pointer-based link structure as accents: phonword_id comes
    from a <nite:pointer role="at" href="..."/> child, not an
    attribute.
  - IMPORTANT, confirmed by inspection: multiple <break> entries can
    point to the SAME phonword_id (seen in the pasted samples, e.g.
    two separate break entries both pointing to ms3A_pw14 at
    slightly different UWtime values). This is because breaks.xml
    stores TWO parallel annotation sources (UW and Edinburgh/
    Stanford) in some files. This script does NOT attempt to resolve
    or deduplicate this -- it is flagged and counted, and left for
    the bridging stage to decide how to handle explicitly, since
    silently averaging or picking one could hide a real modeling
    decision that deserves to be visible.

Author: Nitin Kumar
Date: August 2026 (prosodic extension, stage 3)
"""

import os
import re
import pandas as pd
from lxml import etree
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR

BREAKS_DIR = os.path.join(
    ROOT_DIR, "data", "switchboard-nxt-release-1.0", "xml", "breaks"
)

OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "breaks_index.csv")
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
    if not href:
        return None
    m = re.search(r"#id\(([^)]+)\)", href)
    return m.group(1) if m else None


def parse_breaks(breaks_dir):
    records = []
    n_no_pointer = 0
    n_unparsed_href = 0

    if not os.path.isdir(breaks_dir):
        raise FileNotFoundError(
            f"Breaks directory not found: {breaks_dir}\n"
            f"Check the BREAKS_DIR path against your actual folder layout."
        )

    files = sorted(f for f in os.listdir(breaks_dir)
                   if f.endswith(".breaks.xml"))
    total = len(files)
    log(f"Found {total} break files. Parsing...")

    if total == 0:
        log("  WARNING: zero files found -- check BREAKS_DIR and file naming pattern.")

    for idx, fname in enumerate(files):
        base  = fname.replace(".breaks.xml", "")
        parts = base.split(".")
        if len(parts) < 2:
            log(f"  [SKIP] {fname}: unexpected filename pattern")
            continue
        conv_id = parts[0]
        speaker = parts[1]

        fpath = os.path.join(breaks_dir, fname)
        try:
            tree = etree.parse(fpath)
        except Exception as e:
            log(f"  [SKIP] {fname}: {e}")
            continue

        root = tree.getroot()

        for elem in root:
            # Skip comment nodes (elem.tag is a callable, not a string,
            # for XML comments -- e.g. "<!-- File generated from the UW
            # files. -->" appears at the top of some breaks.xml files).
            if not isinstance(elem.tag, str):
                continue

            local_tag = elem.tag.split("}")[-1]
            if local_tag != "break":
                continue

            break_id      = elem.get(f"{{{NS}}}id", "")
            break_index   = elem.get("index", None)
            phrase_tone   = elem.get("phraseTone", None)
            boundary_tone = elem.get("boundaryTone", None)
            start         = elem.get(f"{{{NS}}}start")

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
                "conv_id":       conv_id,
                "speaker":       speaker,
                "break_id":      break_id,
                "phonword_id":   phonword_id,
                "break_index":   break_index,
                "phrase_tone":   phrase_tone,
                "boundary_tone": boundary_tone,
                "break_start":   start_f,
            })

        if (idx + 1) % 100 == 0:
            log(f"  {idx+1}/{total} files done...")

    log(f"  Done -- all {total} files processed")
    log(f"  Breaks with no <nite:pointer> child at all: {n_no_pointer:,}")
    log(f"  Breaks whose href didn't match expected pattern: {n_unparsed_href:,}")

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    log_section("Prosodic Extension -- Stage 3: Parse Breaks")
    log(f"Reading from: {BREAKS_DIR}")

    df = parse_breaks(BREAKS_DIR)

    log(f"\nTotal break tokens parsed: {len(df):,}")
    log(f"Conversations covered: {df['conv_id'].nunique():,}")

    n_null_phonword = df['phonword_id'].isna().sum()
    log(f"Rows with no resolved phonword_id: {n_null_phonword:,}")

    log("\nbreak_index value distribution (top 15):")
    log(df['break_index'].value_counts(dropna=False).head(15).to_string())

    # Flag, not silently resolve, the multiple-breaks-per-phonword case
    dup_counts = df.groupby(['conv_id', 'speaker', 'phonword_id']).size()
    n_multi = (dup_counts > 1).sum()
    log(f"\nphonword_ids with MORE THAN ONE break entry pointing to them: {n_multi:,}")
    if n_multi > 0:
        log("  This is expected -- breaks.xml carries both UW and Edinburgh/Stanford")
        log("  annotation sources in some files. The bridging stage (08) must make")
        log("  an explicit, documented choice about which source to prefer, rather")
        log("  than silently averaging or picking one arbitrarily here.")
        example_ids = dup_counts[dup_counts > 1].head(3)
        log(f"  Example duplicated keys: {list(example_ids.index)}")

    df.to_csv(OUTPUT_CSV, index=False)

    log_section("Stage 3 COMPLETE")
    log(f"Saved -> {OUTPUT_CSV}")
    log("\nNext: 08_bridge_terminals_phonwords.py -- links phonword_id back to")
    log("word_id, and will need to incorporate the accents_index.csv and")
    log("breaks_index.csv outputs from stages 2 and 3.")