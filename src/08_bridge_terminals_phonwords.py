"""
08_bridge_terminals_phonwords.py

Prosodic extension, stage 4 of 7 -- the bridging script.

words_labeled.csv is keyed on the Penn-Treebank-style word_id from
01_parse_xml.py, but that script never pulled the <nite:pointer role="phon">
child that links each terminal word to its MS-State phonword(s) -- it only
grabbed text/POS/timing. So this script goes back and re-reads the raw
terminals XML just for those pointers, then uses them to attach accent
(accents_index.csv) and break (breaks_index.csv) data onto each word_id.

Four mapping cases show up (checked these by hand against real corpus
samples before writing this):
  1. 1:1        -- normal case, one terminal word -> one phonword
  2. many:1     -- contractions. "it" + "'s" in terminals both point to
                   the single phonword "it's". Both word_ids legitimately
                   get the same prosodic values here.
  3. 1:many     -- multi-word spans, e.g. "eighty-three" points to a range
                   of phonwords ("eighty".."three"). Using the FIRST
                   phonword in the range since it's closest to the word's
                   own start time.
  4. no mapping -- msstateID="non-aligned" words have no phon pointer at
                   all. NaN for every prosodic feature -- genuinely missing
                   data, handled later by propagate_feature(), not here.

Duplicate resolution (decided this deliberately, not just averaging):
  - accents: if a phonword has multiple accent rows, prefer the one with
    non-null `type` (the manual Edinburgh/Stanford annotation seems to
    populate this more reliably than the UW automatic one), else first row
  - breaks: same idea, prefer the row with non-null phrase_tone or
    boundary_tone, else first row
  - break_index: strip trailing '-'/'p' to get the core 0-4 ToBI scale
    ('2p' -> 2, '3-' -> 3). 'X' (annotator couldn't judge) -> NaN, not a value

in:  data/processed/words_labeled.csv
     data/switchboard-nxt-release-1.0/xml/terminals/*.xml  (re-read, for pointers)
     data/processed/phonwords_index.csv   (stage 1)
     data/processed/accents_index.csv     (stage 2)
     data/processed/breaks_index.csv      (stage 3)
out: data/processed/word_prosody_bridge.csv
     columns: word_id, phonword_id_used, mapping_case, accent_strength,
     accent_type, break_index_raw, break_index_clean, phrase_tone, boundary_tone

Nitin, Aug 2026 (prosodic extension, stage 4)
"""

import os
import re
import pandas as pd
import numpy as np
from lxml import etree
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

TERMINALS_DIR = os.path.join(ROOT_DIR, "data", "switchboard-nxt-release-1.0", "xml", "terminals")
LABELED_CSV   = os.path.join(PROCESSED, "words_labeled.csv")
PHONWORDS_CSV = os.path.join(PROCESSED, "phonwords_index.csv")
ACCENTS_CSV   = os.path.join(PROCESSED, "accents_index.csv")
BREAKS_CSV    = os.path.join(PROCESSED, "breaks_index.csv")
OUTPUT_CSV    = os.path.join(PROCESSED, "word_prosody_bridge.csv")
LOG_FILE      = os.path.join(PROCESSED, "pipeline_log.txt")

NS = "http://nite.sourceforge.net/"


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def log_section(title):
    line = "\n" + "-" * 50 + f"\n{title}\n" + "-" * 50
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def extract_phonword_ids(href):
    """
    href can point to a single phonword or a range:
      'sw2018.A.phonwords.xml#id(ms2B_pw1)' -> ['ms2B_pw1']
      'sw2018.A.phonwords.xml#id(ms64B_pw441)..id(ms64B_pw442)'
        -> ['ms64B_pw441', 'ms64B_pw442']
    empty/no match -> []
    """
    if not href:
        return []
    return re.findall(r"id\(([^)]+)\)", href)


def parse_terminals_pointers(terminals_dir):
    """
    Re-reads the raw terminals XML (not words_labeled.csv) to grab the
    <nite:pointer role="phon" href="..."/> child of each <word> -- this
    is the thing 01_parse_xml.py skipped.

    Returns conv_id, speaker, local_id, phonword_ids (list col, empty
    for non-aligned words).
    """
    records = []
    n_no_pointer = 0

    files = sorted(f for f in os.listdir(terminals_dir) if f.endswith(".xml"))
    total = len(files)
    log(f"re-reading {total} terminals files for phon pointers...")

    for idx, fname in enumerate(files):
        base = fname.replace(".terminals.xml", "")
        parts = base.split(".")
        if len(parts) < 2:
            continue
        conv_id = parts[0]
        speaker = parts[1]

        fpath = os.path.join(terminals_dir, fname)
        try:
            tree = etree.parse(fpath)
        except Exception as e:
            log(f"  skipping {fname}: {e}")
            continue

        root = tree.getroot()

        for elem in root:
            if not isinstance(elem.tag, str):
                continue
            if elem.tag.split("}")[-1] != "word":
                continue

            local_id = elem.get(f"{{{NS}}}id", "")
            msstate_id = elem.get("msstateID", "")

            phonword_ids = []
            for child in elem:
                if not isinstance(child.tag, str):
                    continue
                if child.tag.split("}")[-1] == "pointer" and child.get("role") == "phon":
                    href = child.get(f"{{{NS}}}href", child.get("href", ""))
                    phonword_ids = extract_phonword_ids(href)
                    break

            if not phonword_ids:
                n_no_pointer += 1

            records.append({
                "conv_id": conv_id,
                "speaker": speaker,
                "local_id": local_id,
                "msstateID": msstate_id,
                "phonword_ids": phonword_ids,
            })

        if (idx + 1) % 100 == 0:
            log(f"  {idx+1}/{total} files done")

    log(f"  done, {total} files read")
    log(f"  terminal words with no phon pointer at all: {n_no_pointer:,}")

    return pd.DataFrame(records)


def resolve_accent(df):
    """
    Tie-break rule across the whole accents table: within each
    (conv_id, speaker, phonword_id) group, prefer the row with non-null
    'type', else keep whichever row came first.

    Doing this with sort + drop_duplicates instead of groupby().apply()
    -- apply() was silently folding the grouping columns into the
    result's index, which threw a KeyError later when set_index() ran
    and conv_id/speaker/phonword_id weren't columns anymore. Took a
    while to track down.
    """
    df = df.copy()
    # has_type=True sorts first with ascending=False + stable sort,
    # so the row we want ends up first within each group
    df['_has_type'] = df['type'].notna()
    df = df.sort_values('_has_type', ascending=False, kind='stable')
    resolved = df.drop_duplicates(subset=['conv_id', 'speaker', 'phonword_id'], keep='first')
    return resolved.drop(columns=['_has_type']).reset_index(drop=True)


def resolve_break(df):
    """Same idea as resolve_accent() -- prefer non-null phrase_tone/boundary_tone."""
    df = df.copy()
    df['_has_tone'] = df['phrase_tone'].notna() | df['boundary_tone'].notna()
    df = df.sort_values('_has_tone', ascending=False, kind='stable')
    resolved = df.drop_duplicates(subset=['conv_id', 'speaker', 'phonword_id'], keep='first')
    return resolved.drop(columns=['_has_tone']).reset_index(drop=True)


def clean_break_index(raw, drop_p=True):
    """
    '2p' etc. are the corpus's disfluent-pause codes. The corpus README says these
    duplicate the disfluency layer, so keeping them as plain 1-3 values leaks the label.
    Treating them as missing here.
    """
    if pd.isna(raw):
        return np.nan
    raw = str(raw).strip()
    if raw.upper() == 'X':
        return np.nan
    if drop_p and 'p' in raw:
        return np.nan
    try:
        return int(raw.rstrip('-'))
    except ValueError:
        return np.nan


if __name__ == "__main__":
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- new run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    log_section("prosodic extension, stage 4: bridge terminals <-> phonwords")

    log("loading words_labeled.csv...")
    labeled_df = pd.read_csv(LABELED_CSV)
    log(f"  {len(labeled_df):,} words")

    log("re-reading terminals XML for phon pointers (01_parse_xml.py didn't capture these)...")
    pointers_df = parse_terminals_pointers(TERMINALS_DIR)

    log("loading phonwords_index.csv...")
    phonwords_df = pd.read_csv(PHONWORDS_CSV)
    log(f"  {len(phonwords_df):,} phonwords")

    log("loading accents_index.csv and breaks_index.csv...")
    accents_df = pd.read_csv(ACCENTS_CSV)
    breaks_df = pd.read_csv(BREAKS_CSV)
    log(f"  accents: {len(accents_df):,} rows, {accents_df['phonword_id'].nunique():,} unique phonwords")
    log(f"  breaks: {len(breaks_df):,} rows, {breaks_df['phonword_id'].nunique():,} unique phonwords")

    log("resolving duplicate accent entries (prefer non-null type)...")
    accent_resolved = resolve_accent(accents_df)
    log(f"  down to {len(accent_resolved):,} rows, one per phonword")

    log("resolving duplicate break entries (prefer non-null tone info)...")
    break_resolved = resolve_break(breaks_df)
    log(f"  down to {len(break_resolved):,} rows, one per phonword")

    break_resolved['break_index_clean'] = break_resolved['break_index'].apply(clean_break_index)
    n_x = (break_resolved['break_index'].astype(str).str.upper() == 'X').sum()
    log(f"  break_index values that were 'X' (treated as missing): {n_x:,}")

    log("building word_id -> phonword_id bridge, classifying mapping case...")

    # local_id in words_labeled.csv lines up with the terminals nite:id --
    # rebuild the same join key 01_parse_xml.py used for word_id
    pointers_df['word_id'] = pointers_df['conv_id'] + "." + pointers_df['speaker'] + "." + pointers_df['local_id']

    bridge_rows = []
    n_case_1_to_many = 0
    n_case_none = 0

    accent_lookup = accent_resolved.set_index(['conv_id', 'speaker', 'phonword_id'])
    break_lookup = break_resolved.set_index(['conv_id', 'speaker', 'phonword_id'])

    for _, row in pointers_df.iterrows():
        word_id = row['word_id']
        conv_id = row['conv_id']
        speaker = row['speaker']
        pw_ids = row['phonword_ids']

        if len(pw_ids) == 0:
            n_case_none += 1
            bridge_rows.append({
                "word_id": word_id, "conv_id": conv_id, "speaker": speaker,
                "phonword_id_used": None,
                "mapping_case": "no_mapping",
                "accent_strength": np.nan, "accent_type": np.nan,
                "break_index_raw": np.nan, "break_index_clean": np.nan,
                "phrase_tone": np.nan, "boundary_tone": np.nan,
            })
            continue

        if len(pw_ids) > 1:
            n_case_1_to_many += 1

        pw_id_used = pw_ids[0]

        acc_strength, acc_type = np.nan, np.nan
        key = (conv_id, speaker, pw_id_used)
        if key in accent_lookup.index:
            acc_row = accent_lookup.loc[key]
            if isinstance(acc_row, pd.DataFrame):
                acc_row = acc_row.iloc[0]
            acc_strength = acc_row['strength']
            acc_type = acc_row['type']

        brk_raw, brk_clean, ph_tone, bd_tone = np.nan, np.nan, np.nan, np.nan
        if key in break_lookup.index:
            brk_row = break_lookup.loc[key]
            if isinstance(brk_row, pd.DataFrame):
                brk_row = brk_row.iloc[0]
            brk_raw = brk_row['break_index']
            brk_clean = brk_row['break_index_clean']
            ph_tone = brk_row['phrase_tone']
            bd_tone = brk_row['boundary_tone']

        mapping_case = "1_to_many" if len(pw_ids) > 1 else "unresolved_1to1_or_many_to_1"

        bridge_rows.append({
            "word_id": word_id, "conv_id": conv_id, "speaker": speaker,
            "phonword_id_used": pw_id_used,
            "mapping_case": mapping_case,
            "accent_strength": acc_strength, "accent_type": acc_type,
            "break_index_raw": brk_raw, "break_index_clean": brk_clean,
            "phrase_tone": ph_tone, "boundary_tone": bd_tone,
        })

    bridge_df = pd.DataFrame(bridge_rows)

    # phonword_id strings aren't globally unique -- "ms1A_pw1" shows up in
    # almost every conversation -- so the many:1 check has to be scoped to
    # (conv_id, speaker, phonword_id_used) together, never phonword_id_used
    # alone. First version of this checked it alone and misclassified ~92%
    # of the corpus as many_to_1 -- caught it via the word-text mismatch
    # check below, which was throwing up totally unrelated pairs like
    # word="know" matched against orth="[laughter-it's]"
    pw_usage_counts = bridge_df.groupby(['conv_id', 'speaker', 'phonword_id_used']).size()
    many_to_1_keys = set(pw_usage_counts[pw_usage_counts > 1].index)

    def finalize_case(row):
        if row['mapping_case'] == 'no_mapping':
            return 'no_mapping'
        if row['mapping_case'] == '1_to_many':
            return '1_to_many'
        key = (row['conv_id'], row['speaker'], row['phonword_id_used'])
        return 'many_to_1' if key in many_to_1_keys else '1_to_1'

    bridge_df['mapping_case'] = bridge_df.apply(finalize_case, axis=1)

    case_counts = bridge_df['mapping_case'].value_counts()
    log("mapping case distribution:")
    log(case_counts.to_string())

    # word text sanity check, same idea as the alignment check in
    # 03_syntactic_surprisal_modelblocks.py
    log("checking word text agreement between terminals and phonwords...")
    merged_check = bridge_df.merge(labeled_df[['word_id', 'word']], on='word_id', how='left')

    # phonword_id is only unique WITHIN a (conv_id, speaker) pair, so this
    # merge has to include conv_id and speaker too -- joining on
    # phonword_id_used alone blows up into a cartesian product (every
    # "ms1A_pw1" across all 642 conversations matches every other one),
    # same bug class as the many_to_1 thing above. Caught it this time
    # because the row count after merging (13.3M) was way bigger than the
    # actual 1:1 row count (742K) -- worth checking row counts after any
    # merge in this pipeline, not just trusting it ran without error
    phonwords_renamed = phonwords_df[['conv_id', 'speaker', 'phonword_id', 'orth']].rename(
        columns={'phonword_id': 'phonword_id_used'}
    )
    merged_check = merged_check.merge(
        phonwords_renamed, on=['conv_id', 'speaker', 'phonword_id_used'], how='left'
    )
    comparable = merged_check[merged_check['phonword_id_used'].notna()]

    # only checking 1:1 rows here -- many:1 and 1:many cases are expected
    # to show text differences ("it" vs "it's") so they're excluded
    strict_1to1 = comparable[comparable['mapping_case'] == '1_to_1']
    mismatches = strict_1to1[
        strict_1to1['word'].astype(str).str.lower() != strict_1to1['orth'].astype(str).str.lower()
    ]
    log(f"  1:1 rows checked: {len(strict_1to1):,}")
    log(f"  text mismatches (1:1 only): {len(mismatches):,}")

    if len(mismatches) > 0:
        log("  sample mismatches:")
        log(mismatches[['word_id', 'word', 'phonword_id_used', 'orth']].head(10).to_string())

        # rather than just eyeballing these and calling it "mostly benign",
        # sorting mismatches into known patterns from an earlier sample run
        # (filler spelling, fragment notation, hyphenation differences
        # between Penn Treebank and MS-State transcripts). first matching
        # rule wins.
        def categorize_mismatch(row):
            w = str(row['word']).lower().strip()
            o = str(row['orth']).lower().strip()

            # filler spelling variants (uh/um family)
            filler_pairs = {
                ('uh', 'um'), ('um', 'uh'),
                ('uh-huh', 'um-hum'), ('um-hum', 'uh-huh'),
                ('huh', 'hum'), ('hum', 'huh'),
            }
            if (w, o) in filler_pairs:
                return "filler_spelling_variant"

            # laughter spans tend to drift out of position by a word or two
            # relative to the Penn Treebank side (word="whether" paired with
            # orth="[laughter-would]") -- the drift during laughter is the
            # pattern, not any specific word pair
            if o.startswith('[laughter-') and o.endswith(']'):
                return "laughter_span_drift"

            # fragment notation -- cut-off words can be marked at either end
            # ("tr-" for a word cut off at the end, or "-[becau]se" for a
            # word picked back up mid-utterance) with the other side showing
            # the inferred completion in brackets
            if w.endswith('-') or w.startswith('-') or o.endswith('-') or o.startswith('-'):
                w_stem = w.strip('-')
                o_stem = re.sub(r'\[.*?\]', '', o).strip('-')
                if w_stem and o_stem and (w_stem in o_stem or o_stem in w_stem):
                    return "fragment_notation_difference"

            # same letters, just a hyphen difference
            w_nohyphen = w.replace('-', '')
            o_nohyphen = o.replace('-', '')
            if w_nohyphen == o_nohyphen and w != o:
                return "hyphenation_difference"

            # shared stem, different suffix (e.g. strengthening/strengthened)
            shorter, longer = (w, o) if len(w) <= len(o) else (o, w)
            if len(shorter) >= 4 and shorter[:len(shorter) - 2] == longer[:len(shorter) - 2]:
                return "shared_stem_different_suffix"

            # short backchannel words (oh, yeah, right, um-hum, etc.) seem to
            # have looser alignment in general, beyond just uh/um above --
            # both sides being short common backchannel words is the signal.
            # real content differences, but plausibly just loose alignment
            # during overlapping speech rather than an actual tracking bug
            backchannel_words = {
                'oh', 'yeah', 'yep', 'right', 'that', 'it', "that's", "it's",
                'um-hum', 'uh-huh', 'ok', 'okay', 'sure', 'well',
            }
            if w in backchannel_words and o in backchannel_words:
                return "backchannel_word_confusion"

            return "unexplained_other"

        mismatches = mismatches.copy()
        mismatches['pattern'] = mismatches.apply(categorize_mismatch, axis=1)
        pattern_counts = mismatches['pattern'].value_counts()
        log("mismatch breakdown by pattern:")
        log(pattern_counts.to_string())
        log(f"  unexplained: {pattern_counts.get('unexplained_other', 0):,} of {len(mismatches):,} "
            f"({100 * pattern_counts.get('unexplained_other', 0) / len(mismatches):.1f}%) -- worth a look")

        unexplained = mismatches[mismatches['pattern'] == 'unexplained_other']
        if len(unexplained) > 0:
            log("  sample of unexplained mismatches:")
            log(unexplained[['word_id', 'word', 'phonword_id_used', 'orth']].head(15).to_string())

    bridge_df.to_csv(OUTPUT_CSV, index=False)

    log_section("stage 4 done")
    log(f"saved -> {OUTPUT_CSV}")
    log(f"  word_ids covered: {len(bridge_df):,}")
    log(f"  words with resolved accent info: {bridge_df['accent_strength'].notna().sum():,}")
    log(f"  words with resolved break info: {bridge_df['break_index_clean'].notna().sum():,}")
    log("next: 11_build_prosodic_features.py -- builds prev_*/foll_* shifted features "
        "from this table, using the existing propagate_feature() logic for words with "
        "no prosodic data (non-aligned or outside the 150-file annotated subset)")