"""
import_modelblocks.py
Imports ModelBlocks surprisal from swbd_5000.tokdecs
and merges into pipeline.
Run from project root: python src/import_modelblocks.py
"""
import os, math
import numpy as np
import pandas as pd

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB_DIR    = os.path.join(ROOT, 'data', 'modelblocks')
PROCESSED = os.path.join(ROOT, 'data', 'processed')

TOKDECS   = os.path.join(
    ROOT,
    'Modelbock release master',
    'modelblocks-release-master',
    'workspace', 'genmodel', 'testing',
    'swbd_5000.tokdecs'
)
INDEX_CSV = os.path.join(MB_DIR, 'swbd_word_index.csv')
OUT_MB    = os.path.join(MB_DIR, 'swbd_5000_surprisal.csv')
OUT_CSV   = os.path.join(PROCESSED, 'syntactic_surprisal_modelblocks_final.csv')

print("=" * 60)
print("Importing ModelBlocks surprisal from swbd_5000.tokdecs")
print("=" * 60)

# ── Load index ────────────────────────────────────────────────
print("\nLoading word index...")
index_df = pd.read_csv(INDEX_CSV)
print(f"  {len(index_df):,} words, {index_df['sent_idx'].nunique():,} sentences")

# ── Parse tokdecs ─────────────────────────────────────────────
print("\nParsing tokdecs file...")
mb_rows  = []
sent_idx = -1
word_pos = 0
header_done = False

with open(TOKDECS, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if not header_done:
            if line.lower().startswith('word'):
                header_done = True
                continue
        if line.startswith('!ARTICLE'):
            sent_idx += 1
            word_pos  = 0
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = parts[0]
        try:
            surp = float(parts[-1])
        except:
            surp = np.nan
        mb_rows.append({
            'sent_idx': sent_idx,
            'word_pos': word_pos,
            'mb_word':  word,
            'totsurp':  surp
        })
        word_pos += 1

mb_df = pd.DataFrame(mb_rows)
print(f"  {len(mb_df):,} words read")
print(f"  {mb_df['sent_idx'].nunique():,} sentences")
print(f"  Valid surprisal: {mb_df['totsurp'].notna().sum():,}")
print(f"  FAIL/nan: {mb_df['totsurp'].isna().sum():,}")

# ── Merge with index ──────────────────────────────────────────
print("\nMerging with word index...")
merged = index_df.merge(mb_df, on=['sent_idx','word_pos'], how='left')

# Check alignment
mismatches = merged[
    merged['mb_word'].notna() &
    (merged['word'].str.lower() != merged['mb_word'].str.lower())
]
print(f"  Matched words: {merged['totsurp'].notna().sum():,}")
print(f"  Mismatches: {len(mismatches):,}")
if len(mismatches) == 0:
    print("  All words matched correctly!")

# ── Build prev and foll surprisal ─────────────────────────────
print("\nBuilding prev_syn_surp and foll_syn_surp...")
results = []
for sid, group in merged.groupby('sent_idx'):
    group = group.sort_values('word_pos').reset_index(drop=True)
    n     = len(group)
    surps = group['totsurp'].tolist()
    ids   = group['word_id'].tolist()
    for i in range(n):
        s = surps[i]
        sn = surps[i+1] if i+1 < n else np.nan
        results.append({
            'word_id':       ids[i],
            'prev_syn_surp': round(s, 6)  if not (s != s or s == float('inf')) else np.nan,
            'foll_syn_surp': round(sn, 6) if not (sn != sn or sn == float('inf')) else np.nan,
        })

result_df = pd.DataFrame(results)
result_df.to_csv(OUT_MB,  index=False)
result_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved -> {OUT_MB}")
print(f"Saved -> {OUT_CSV}")
print(f"  Total rows:       {len(result_df):,}")
print(f"  prev_syn_surp ok: {result_df['prev_syn_surp'].notna().sum():,}")
print(f"  foll_syn_surp ok: {result_df['foll_syn_surp'].notna().sum():,}")
print("\nSample (valid rows):")
print(result_df[result_df['prev_syn_surp'].notna()].head(8).to_string())
print("\n" + "=" * 60)
print("COMPLETE — Next: python src/compare_pcfg_modelblocks.py")
print("=" * 60)