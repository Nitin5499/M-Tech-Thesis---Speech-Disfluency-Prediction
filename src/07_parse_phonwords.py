"""
07_parse_phonwords.py
======================
PROSODIC EXTENSION -- Stage 1 of 7.

Parses the Switchboard NXT phonwords XML files (MS-State transcript,
time-aligned to audio) into a single flat CSV index.

This is a pure extraction step -- no bridging, no feature computation.
Mirrors the structure of 01_parse_xml.py exactly, so the two pipelines
(Penn Treebank / terminals side, and MS-State / phonwords side) stay
easy to compare side by side.

INPUT:
  data/switchboard-nxt-release-1.0/xml/phonwords/*.phonwords.xml
  (one file per conversation-speaker pair, e.g. sw2005.A.phonwords.xml)

OUTPUT:
  data/processed/phonwords_index.csv
    columns: conv_id, speaker, phonword_id, orth, start, end,
             msstate, stress_profile

NOTES ON THE XML STRUCTURE (from inspected samples):
  - Each <phonword> has nite:id (e.g. ms1A_pw1), orth (word text,
    already lowercased by MS-State), nite:start / nite:end (float
    seconds), and optionally stressProfile.
  - Some phonwords carry no stressProfile attribute at all (seen on
    words like "really", "actually" mid-sentence) -- treated as
    missing, not an error.
  - <noise> and <laughter> elements also appear in this stream,
    interspersed with <phonword> elements. These are NOT words and
    are explicitly skipped -- they have no orth attribute we want to
    treat as a spoken word, and no corresponding terminals.xml entry
    to bridge to later.

Author: Nitin Kumar
Date: August 2026 (prosodic extension, stage 1)
"""

import os
import pandas as pd
from lxml import etree
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR

# Dataset root confirmed identical to existing pipeline's convention:
# data/switchboard-nxt-release-1.0/xml/... -- same corpus copy already
# used by 01_parse_xml.py and 03a_clean_corpus.py. The phonwords,
# accent, and breaks folders were confirmed present here alongside
# terminals/ and disfluency/, so no separate corpus copy is needed.
PHONWORDS_DIR = os.path.join(
    ROOT_DIR, "data", "switchboard-nxt-release-1.0", "xml", "phonwords"
)

OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "phonwords_index.csv")
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


def parse_phonwords(phonwords_dir):
    records = []
    n_skipped_noise = 0
    n_skipped_laughter = 0
    n_missing_stress = 0

    if not os.path.isdir(phonwords_dir):
        raise FileNotFoundError(
            f"Phonwords directory not found: {phonwords_dir}\n"
            f"Check the PHONWORDS_DIR path against your actual folder layout."
        )

    files = sorted(f for f in os.listdir(phonwords_dir)
                   if f.endswith(".phonwords.xml"))
    total = len(files)
    log(f"Found {total} phonwords files. Parsing...")

    if total == 0:
        log("  WARNING: zero files found -- check PHONWORDS_DIR and file naming pattern.")

    for idx, fname in enumerate(files):
        # e.g. "sw2005.A.phonwords.xml" -> conv_id="sw2005", speaker="A"
        base  = fname.replace(".phonwords.xml", "")
        parts = base.split(".")
        if len(parts) < 2:
            log(f"  [SKIP] {fname}: unexpected filename pattern")
            continue
        conv_id = parts[0]
        speaker = parts[1]

        fpath = os.path.join(phonwords_dir, fname)
        try:
            tree = etree.parse(fpath)
        except Exception as e:
            log(f"  [SKIP] {fname}: {e}")
            continue

        root = tree.getroot()

        for elem in root:
            # Skip comment nodes (elem.tag is a callable, not a string,
            # for XML comments) -- seen in some phonwords files' headers.
            if not isinstance(elem.tag, str):
                continue

            local_tag = elem.tag.split("}")[-1]

            if local_tag == "noise":
                n_skipped_noise += 1
                continue
            if local_tag == "laughter":
                n_skipped_laughter += 1
                continue
            if local_tag != "phonword":
                continue

            phonword_id = elem.get(f"{{{NS}}}id", "")
            orth        = elem.get("orth", "").strip()
            start       = elem.get(f"{{{NS}}}start")
            end         = elem.get(f"{{{NS}}}end")
            msstate     = elem.get("msstate", "")
            stress      = elem.get("stressProfile", None)

            if not orth:
                continue

            try:
                start_f = float(start) if start is not None else None
                end_f   = float(end)   if end   is not None else None
            except (ValueError, TypeError):
                start_f, end_f = None, None

            if stress is None:
                n_missing_stress += 1

            records.append({
                "conv_id":       conv_id,
                "speaker":       speaker,
                "phonword_id":   phonword_id,
                "orth":          orth,
                "start":         start_f,
                "end":           end_f,
                "msstate":       msstate,
                "stress_profile": stress,
            })

        if (idx + 1) % 100 == 0:
            log(f"  {idx+1}/{total} files done...")

    log(f"  Done -- all {total} files processed")
    log(f"  Skipped <noise> elements:    {n_skipped_noise:,}")
    log(f"  Skipped <laughter> elements: {n_skipped_laughter:,}")
    log(f"  Phonwords with no stress profile attribute: {n_missing_stress:,}")

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    log_section("Prosodic Extension -- Stage 1: Parse Phonwords")
    log(f"Reading from: {PHONWORDS_DIR}")

    df = parse_phonwords(PHONWORDS_DIR)

    log(f"\nTotal phonword tokens parsed: {len(df):,}")
    log(f"Conversations covered: {df['conv_id'].nunique():,}")
    log(f"Speaker sides covered: {df.groupby(['conv_id','speaker']).ngroups:,}")

    # Basic sanity checks before saving -- same discipline as the
    # existing pipeline's verification steps.
    n_null_start = df['start'].isna().sum()
    n_null_end   = df['end'].isna().sum()
    log(f"Rows with missing start time: {n_null_start:,}")
    log(f"Rows with missing end time:   {n_null_end:,}")

    # Duplicate phonword_id check -- IDs should be unique per file,
    # but since IDs are only unique WITHIN a conversation-speaker
    # side (not globally), we check uniqueness within that scope.
    dup_check = df.duplicated(subset=['conv_id', 'speaker', 'phonword_id']).sum()
    log(f"Duplicate (conv_id, speaker, phonword_id) rows: {dup_check:,}")
    if dup_check > 0:
        log("  WARNING: duplicates found -- investigate before proceeding to bridging stage.")

    df.to_csv(OUTPUT_CSV, index=False)

    log_section("Stage 1 COMPLETE")
    log(f"Saved -> {OUTPUT_CSV}")
    log("\nNext: run 09_parse_accents.py and 10_parse_breaks.py, then")
    log("08_bridge_terminals_phonwords.py to link this back to word_id.")