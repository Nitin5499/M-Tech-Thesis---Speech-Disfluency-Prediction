"""
21_prosody_by_source.py

Robustness check for the prosody results: does the break (and accent) effect look the
same in the two kinds of annotated files?

The README.PROSODY.txt says the annotated files come from different sources:
  - UW / Edinburgh Converted : every word has prosody annotation, breaks have the full 0-4 scale
  - Edinburgh Original       : only words in sentences annotated for kontrast carry prosody,
                               and breaks only exist at phrase ends (2p, 3, 4, X)
The first line of each breaks xml says which one it is
("File generated from the UW files" / "... EDINBURGH ORIGINAL files").

For each of the four prosody datasets this prints, per source:
  rows, share disfluent, and for the break sets the AUC of prev_break_index alone and
  the share disfluent at each break value; for the accent sets the share of rows with accent 0.0

run from repo root:  python src/21_prosody_by_source.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_BREAKS = os.path.join(ROOT, 'data', 'switchboard-nxt-release-1.0', 'xml', 'breaks')
PROC = os.path.join(ROOT, 'data', 'processed')

pd.set_option('display.width', 200)

# which source each speaker-side (e.g. sw2005.A) comes from
src = {}
for f in glob.glob(os.path.join(XML_BREAKS, '*.breaks.xml')):
    head = open(f, encoding='utf-8', errors='ignore').read(800)
    m = re.search(r'generated from the ([^.]*)', head)
    side = os.path.basename(f).replace('.breaks.xml', '')
    is_orig = bool(m) and 'EDINBURGH ORIGINAL' in m.group(1).upper()
    src[side] = 'edinburgh_original' if is_orig else 'uw_or_converted'

print('speaker-sides by source:', pd.Series(src).value_counts().to_dict())


def auc(y, s):
    # rank-based AUC, same idea as in the GLMM script
    y = np.asarray(y)
    r = pd.Series(s).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


names = ['dataset_filler_break', 'dataset_reparandum_break',
         'dataset_filler_accent', 'dataset_reparandum_accent']

for name in names:
    path = os.path.join(PROC, name + '.csv')
    if not os.path.exists(path):
        print('\nmissing:', path)
        continue
    df = pd.read_csv(path, low_memory=False)
    df['side'] = df['word_id'].str.split('.').str[:2].str.join('.')
    df['source'] = df['side'].map(src)

    print('\n' + '=' * 70)
    print(name, '  rows:', f'{len(df):,}', '  rows with no source found:', int(df['source'].isna().sum()))
    print('=' * 70)

    g = df.groupby('source').agg(rows=('word_id', 'size'), share_disfluent=('disfluency', 'mean')).round(3)
    print(g.to_string())

    if name.endswith('break'):
        for s, d in df.groupby('source'):
            print(f'\n[{s}] AUC of prev_break_index alone: {auc(d["disfluency"], d["prev_break_index"]):.3f}')
            t = d.groupby(d['prev_break_index'].round()).agg(rows=('word_id', 'size'),
                                                            share_disfluent=('disfluency', 'mean')).round(3)
            print(t.to_string())
    else:
        print('\nshare of rows where prev_accent_strength is 0.0, and AUC of prev_accent_strength alone:')
        for s, d in df.groupby('source'):
            print(f'[{s}] zero share: {(d["prev_accent_strength"] == 0).mean():.3f}   '
                  f'AUC: {auc(d["disfluency"], d["prev_accent_strength"]):.3f}   rows: {len(d):,}')

print('\ndone.')