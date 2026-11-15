"""
03a_clean_corpus.py
===================
Creates a clean corpus by:
  1. Removing all disfluent words (fillers + reparandums) from each conversation
  2. Extracting proper sentence boundaries using Switchboard sentence markers
     (s1_1, s2_1 etc — the number before underscore is the sentence ID)
  3. Saving clean sentences and a word index for feature alignment

INPUT:
  data/switchboard-nxt-release-1.0/xml/terminals/  — word XML files
  data/switchboard-nxt-release-1.0/xml/disfluency/ — disfluency XML files
  data/processed/words_labeled.csv                 — labeled words

OUTPUT:
  data/processed/clean_sentences.txt    — one clean sentence per line
  data/processed/clean_word_index.csv   — word_id to sentence/position mapping
  data/processed/pipeline_log.txt       — pipeline log with timestamps

Author: Nitin Kumar
Date: May 2026
"""

import os
import re
import pandas as pd
from lxml import etree
from datetime import datetime
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
XML_DIR     = os.path.join(ROOT_DIR, 'data', 'switchboard-nxt-release-1.0', 'xml')
TERM_DIR    = os.path.join(XML_DIR, 'terminals')
DISF_DIR    = os.path.join(XML_DIR, 'disfluency')
LABELED_CSV = os.path.join(ROOT_DIR, 'data', 'processed', 'words_labeled.csv')
OUT_SENTS   = os.path.join(ROOT_DIR, 'data', 'processed', 'clean_sentences.txt')
OUT_INDEX   = os.path.join(ROOT_DIR, 'data', 'processed', 'clean_word_index.csv')
LOG_FILE    = os.path.join(ROOT_DIR, 'data', 'processed', 'pipeline_log.txt')

# ── Filler POS tags and words ──────────────────────────────────
FILLER_POS   = {'UH'}
FILLER_WORDS = {'uh', 'um', 'mm', 'hm', 'huh', 'oh', 'ah', 'er',
                'uhhuh', 'uh-huh', 'mmhm', 'mmhmm'}

# ── Logger ─────────────────────────────────────────────────────
def log(msg, logfile=LOG_FILE):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(logfile, 'a') as f:
        f.write(line + '\n')

def log_section(title):
    line = "\n" + "=" * 60 + f"\n{title}\n" + "=" * 60
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

# ── Get sentence number from nite:id ──────────────────────────
def get_sent_num(nite_id):
    """
    Extract sentence number from nite:id like s2_3 → 2
    The number before the underscore is the sentence ID.
    """
    match = re.match(r's(\d+)_', nite_id)
    if match:
        return int(match.group(1))
    return None

# ── Load disfluent word IDs ────────────────────────────────────
def load_disfluent_ids(conv_id, speaker):
    """
    Load all reparandum word IDs from the disfluency XML file.
    Returns a set of word IDs that are disfluent.
    """
    disf_ids = set()
    disf_file = os.path.join(DISF_DIR, f"{conv_id}.{speaker}.disfluency.xml")
    if not os.path.exists(disf_file):
        return disf_ids
    try:
        tree = etree.parse(disf_file)
        root = tree.getroot()
        ns = {'nite': 'http://nite.sourceforge.net/'}
        for elem in root.iter():
            role = elem.get('role', '')
            if role in ('reparandum', 'rm', 'repair'):
                href = elem.get('{http://nite.sourceforge.net/}href', '')
                if not href:
                    href = elem.get('href', '')
                ids = re.findall(r'id\(([^)]+)\)', href)
                disf_ids.update(ids)
    except Exception as e:
        pass
    return disf_ids

# ── Main ───────────────────────────────────────────────────────
def main():
    # Clear log
    with open(LOG_FILE, 'w') as f:
        f.write(f"Pipeline Log — Clean Corpus Creation\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    log_section("Step 03a: Clean Corpus Creation")
    log("Approach: DC correction (remove fillers + reparandums)")
    log("Sentence boundaries: Switchboard NXT sentence markers (s1_1, s2_1 etc)")

    # Load labeled words
    log("\nLoading words_labeled.csv...")
    df = pd.read_csv(LABELED_CSV)
    df = df.sort_values(['conv_id', 'speaker', 'start']).reset_index(drop=True)
    log(f"  Total words: {len(df):,}")
    log(f"  Conversations: {df['conv_id'].nunique():,}")
    log(f"  Fillers: {(df['label']=='filler').sum():,}" if 'label' in df.columns else "  Labels loaded")

    # Get unique conversations
    conversations = df[['conv_id', 'speaker']].drop_duplicates().values.tolist()
    log(f"  Conversation sides: {len(conversations):,}")

    clean_sentences = []
    index_rows      = []
    sent_global_idx = 0
    stats = {
        'total_words': 0,
        'fillers_removed': 0,
        'reparandums_removed': 0,
        'clean_words': 0,
        'sentences': 0,
        'empty_sentences': 0,
    }

    log("\nProcessing conversations...")

    for ci, (conv_id, speaker) in enumerate(conversations):

        # Get words for this conversation side
        grp = df[(df['conv_id'] == conv_id) & (df['speaker'] == speaker)]
        grp = grp.sort_values('start').reset_index(drop=True)

        # Load disfluent word IDs
        disf_ids = load_disfluent_ids(conv_id, speaker)

        # Group words by sentence number using nite:id
        sentence_groups = defaultdict(list)
        for _, row in grp.iterrows():
            word_id  = row['word_id']
            # Extract local id from word_id like sw2005.A.s2_3 → s2_3
            local_id = word_id.split('.')[-1]
            sent_num = get_sent_num(local_id)
            if sent_num is None:
                sent_num = 0

            # Determine if word is disfluent
            label    = row.get('label', 'fluent')
            pos      = str(row.get('pos', ''))
            word     = str(row.get('word', '')).strip()
            is_filler = (
                label == 'filler' or
                pos in FILLER_POS or
                word.lower() in FILLER_WORDS
            )
            is_reparandum = (
                label == 'reparandum' or
                local_id in disf_ids
            )

            stats['total_words'] += 1
            if is_filler:
                stats['fillers_removed'] += 1
                continue
            if is_reparandum:
                stats['reparandums_removed'] += 1
                continue

            # Clean word — keep it
            sentence_groups[sent_num].append({
                'word_id':  word_id,
                'word':     word,
                'word_pos': len(sentence_groups[sent_num])
            })

        # Process each sentence
        for sent_num in sorted(sentence_groups.keys()):
            words_in_sent = sentence_groups[sent_num]

            # Filter empty or very short sentences
            clean_words = [w for w in words_in_sent
                          if w['word'] and len(w['word'].strip()) > 0]
            if len(clean_words) < 2:
                stats['empty_sentences'] += 1
                continue

            # Build sentence text
            sent_text = ' '.join(w['word'].lower().strip()
                                 for w in clean_words)
            clean_sentences.append(sent_text)

            # Build index
            for pos, w in enumerate(clean_words):
                index_rows.append({
                    'sent_idx':  sent_global_idx,
                    'word_pos':  pos,
                    'word_id':   w['word_id'],
                    'word':      w['word'].lower().strip(),
                    'conv_id':   conv_id,
                    'speaker':   speaker,
                    'sent_num':  sent_num,
                })
                stats['clean_words'] += 1

            stats['sentences'] += 1
            sent_global_idx += 1

        if (ci + 1) % 100 == 0:
            log(f"  {ci+1}/{len(conversations)} conversation sides processed...")

    log(f"  Done — all {len(conversations)} conversation sides processed")

    # Save outputs
    log("\nSaving outputs...")

    # Save clean sentences
    with open(OUT_SENTS, 'w', encoding='utf-8', newline='\n') as f:
        for sent in clean_sentences:
            f.write(sent + '\n')
    log(f"  Saved clean_sentences.txt — {len(clean_sentences):,} sentences")

    # Save word index
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(OUT_INDEX, index=False)
    log(f"  Saved clean_word_index.csv — {len(index_df):,} words")

    # Print statistics
    log_section("Statistics")
    log(f"  Total words in corpus:    {stats['total_words']:,}")
    log(f"  Fillers removed:          {stats['fillers_removed']:,}")
    log(f"  Reparandums removed:      {stats['reparandums_removed']:,}")
    log(f"  Clean words kept:         {stats['clean_words']:,}")
    log(f"  Empty sentences skipped:  {stats['empty_sentences']:,}")
    log(f"  Clean sentences:          {stats['sentences']:,}")
    log(f"  Average sentence length:  {stats['clean_words']/max(stats['sentences'],1):.1f} words")

    # Sample output
    log("\nSample clean sentences:")
    for i, sent in enumerate(clean_sentences[:5]):
        log(f"  {i+1}: {sent[:80]}")

    log_section("Step 03a COMPLETE")
    log(f"Output files:")
    log(f"  {OUT_SENTS}")
    log(f"  {OUT_INDEX}")
    log(f"  {LOG_FILE}")
    log("\nNext step: Run 03b_prepare_modelblocks.py")

if __name__ == '__main__':
    main()