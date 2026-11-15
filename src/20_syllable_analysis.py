"""
20_syllable_analysis.py

Sir's item 5: how are syllables handled in the prosodic annotation, and how do
the prosodic features (accent, break) vary across words with more than one syllable.

What the annotation gives us:
  - every phonword has a stressProfile with one letter per syllable
    (p = primary stress, s = secondary, n = none), e.g. "about" = np, "elderly" = pnn
  - every phonword also has a child href pointing to a syllable range
    (ms1A_sy1..ms1A_sy2 means 2 syllables)
  - accents and breaks are attached to the WHOLE phonword, not to a single syllable,
    so "variation across syllables" has to be looked at by syllable count and by
    where the primary stress sits

Step 1: read all phonwords xml files -> data/processed/phonword_syllables.csv
Step 2: join onto word_prosody_bridge.csv (by speaker side + phonword id)
Step 3: tables of accent / break by syllable count and by stress pattern

Only words from the annotated speaker-sides and with a clean 1-to-1 terminal-phonword
mapping are used (contractions like i'm map many words to one phonword, so they would
muddy the syllable counts).

run from repo root:  python src/20_syllable_analysis.py
"""

import os
import re
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'src' else BASE_DIR
PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
RESULTS   = os.path.join(ROOT_DIR, "results")
PHON_DIR  = os.path.join(ROOT_DIR, "data", "switchboard-nxt-release-1.0", "xml", "phonwords")

BRIDGE_CSV = os.path.join(PROCESSED, "word_prosody_bridge.csv")
SYL_CSV    = os.path.join(PROCESSED, "phonword_syllables.csv")

NS = '{http://nite.sourceforge.net/}'

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
os.makedirs(RESULTS, exist_ok=True)


def count_from_href(href):
    # ms1A_sy1..ms1A_sy2 -> 2 ; a single id -> 1 ; nothing -> 0
    nums = re.findall(r'_sy(\d+)', href or '')
    if len(nums) >= 2:
        return int(nums[-1]) - int(nums[0]) + 1
    if len(nums) == 1:
        return 1
    return 0


# ---------------------------------------------------------------
# step 1: parse the phonwords files
# ---------------------------------------------------------------
files = sorted(glob.glob(os.path.join(PHON_DIR, '*.phonwords.xml')))
print(f"phonwords files found: {len(files)}")
if not files:
    raise SystemExit("no phonwords files found, check PHON_DIR: " + PHON_DIR)

rows = []
for path in files:
    side = os.path.basename(path).replace('.phonwords.xml', '')   # e.g. sw2005.A
    root = ET.parse(path).getroot()
    for pw in root.iter('phonword'):
        child = pw.find(NS + 'child')
        href = child.get('href') if child is not None else None
        sp = pw.get('stressProfile')
        n_href = count_from_href(href)
        rows.append({
            'side': side,
            'phonword_id': pw.get(NS + 'id'),
            'orth': pw.get('orth'),
            'stress_profile': sp,
            'n_syl_href': n_href,
        })

syl = pd.DataFrame(rows)
# syllable count: from the stress profile when there is one, else from the href range
has_sp = syl['stress_profile'].notna() & (syl['stress_profile'].str.len() > 0)
syl['n_syl'] = np.where(has_sp, syl['stress_profile'].str.len(), syl['n_syl_href'])
syl.to_csv(SYL_CSV, index=False)

print(f"phonwords parsed: {len(syl):,}  (phonwords_index.csv had 801,512)")
print(f"no stressProfile: {(~has_sp).sum():,}  (counted from the href range instead)")
both = has_sp & (syl['n_syl_href'] > 0)
print(f"stressProfile length vs href range disagree: {(syl.loc[both,'stress_profile'].str.len() != syl.loc[both,'n_syl_href']).sum():,} of {both.sum():,}")
print("\nsyllables per phonword, whole corpus:")
print(syl['n_syl'].clip(upper=6).value_counts().sort_index().to_string())

# ---------------------------------------------------------------
# step 2: join onto the bridge, annotated sides only, 1-to-1 only
# ---------------------------------------------------------------
b = pd.read_csv(BRIDGE_CSV, low_memory=False)
b['side'] = b['word_id'].str.split('.').str[:2].str.join('.')

annotated_sides = set(b.loc[b['accent_strength'].notna(), 'side'])
d = b[b['side'].isin(annotated_sides)].copy()
print(f"\nwords in the {len(annotated_sides)} annotated speaker-sides: {len(d):,}")

if 'mapping_case' in d.columns:
    d = d[d['mapping_case'] == '1_to_1']
    print(f"  of which clean 1-to-1 terminal-phonword mapping: {len(d):,}")

d = d.merge(syl[['side', 'phonword_id', 'n_syl', 'stress_profile']],
            left_on=['side', 'phonword_id_used'], right_on=['side', 'phonword_id'], how='left')
print(f"  words with no syllable info after the join: {d['n_syl'].isna().sum():,}")
d = d[d['n_syl'].notna() & (d['n_syl'] > 0)].copy()

d['n_bin'] = d['n_syl'].clip(upper=4).astype(int).astype(str).replace({'4': '4+'})
d['accented'] = d['accent_strength'].notna()
d['has_break'] = d['break_index_clean'].notna()
d['b4'] = np.where(d['has_break'], (d['break_index_clean'] == 4).astype(float), np.nan)
order = ['1', '2', '3', '4+']

# ---------------------------------------------------------------
# step 3: tables
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("TABLE 1: accent by syllable count (annotated words only)")
print("=" * 70)
t1 = d.groupby('n_bin').agg(
    words=('word_id', 'size'),
    accent_rate=('accented', 'mean'),
    full_share_of_accents=('accent_strength', lambda s: (s == 'full').sum() / max(s.notna().sum(), 1)),
).reindex(order)
t1 = t1.round(3)
print(t1.to_string())
t1.to_csv(os.path.join(RESULTS, 'table_syllable_accent.csv'))

print("\n" + "=" * 70)
print("TABLE 2: accent TYPE by syllable count (only accents that have a type)")
print("=" * 70)
typed = d[d['accent_type'].notna()]
t2c = pd.crosstab(typed['n_bin'], typed['accent_type']).reindex(order)
t2p = pd.crosstab(typed['n_bin'], typed['accent_type'], normalize='index').reindex(order).round(3)
print("counts:")
print(t2c.to_string())
print("row shares:")
print(t2p.to_string())
t2c.to_csv(os.path.join(RESULTS, 'table_syllable_accent_type.csv'))

print("\n" + "=" * 70)
print("TABLE 3: break after the word, by syllable count")
print("=" * 70)
t3 = d.groupby('n_bin').agg(
    words=('word_id', 'size'),
    has_break=('has_break', 'mean'),
    share_break4_among_breaks=('b4', 'mean'),
).reindex(order).round(3)
print(t3.to_string())
b3 = d[d['has_break']]
t3b = pd.crosstab(b3['n_bin'], b3['break_index_clean'], normalize='index').reindex(order).round(3)
print("break value shares (only words that have a break value):")
print(t3b.to_string())
t3.to_csv(os.path.join(RESULTS, 'table_syllable_break.csv'))

print("\n" + "=" * 70)
print("TABLE 4: stress pattern (most common patterns, at least 100 words)")
print("=" * 70)
d['profile'] = d['stress_profile'].fillna('none')
t4 = d.groupby('profile').agg(
    words=('word_id', 'size'),
    accent_rate=('accented', 'mean'),
    share_break4_among_breaks=('b4', 'mean'),
)
t4 = t4[t4['words'] >= 100].sort_values('words', ascending=False).head(15).round(3)
print(t4.to_string())
t4.to_csv(os.path.join(RESULTS, 'table_syllable_stress_pattern.csv'))

# where does the primary stress sit in multi-syllable words, and does that matter for accent
multi = d[(d['n_syl'] >= 2) & d['stress_profile'].notna()].copy()
multi['primary_pos'] = multi['stress_profile'].str.find('p') + 1       # 1 = first syllable, 0 = none
print("\nmulti-syllable words: position of the primary stress")
t5 = multi.groupby('primary_pos').agg(words=('word_id', 'size'), accent_rate=('accented', 'mean')).round(3)
print(t5.to_string())

print("\ndone. tables saved in results/ (table_syllable_*.csv); syllable list in data/processed/phonword_syllables.csv")