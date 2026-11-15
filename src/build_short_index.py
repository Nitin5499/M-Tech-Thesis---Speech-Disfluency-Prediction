"""
build_short_index.py
Builds word index matching the 15-word chunk format used for ModelBlocks.
Run from project root: python build_short_index.py
OR from src: python src/build_short_index.py
"""
import pandas as pd
import os

# Always find project root correctly regardless of where script is
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# If script is in src/ go up one level, otherwise use current dir
if os.path.basename(SCRIPT_DIR) == 'src':
    ROOT = os.path.dirname(SCRIPT_DIR)
else:
    ROOT = SCRIPT_DIR

LABELED   = os.path.join(ROOT, 'data', 'processed', 'words_labeled.csv')
OUT_INDEX = os.path.join(ROOT, 'data', 'modelblocks', 'swbd_word_index.csv')

print(f"Project root: {ROOT}")
print("Loading words_labeled.csv...")
df = pd.read_csv(LABELED)
df = df.sort_values(['conv_id','speaker','start']).reset_index(drop=True)
df = df[df['label'] == 'fluent']
df['word'] = df['word'].fillna('').astype(str)
df['word'] = df['word'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)
df = df[df['word'].str.len() > 0].reset_index(drop=True)
print(f"  {len(df):,} fluent words")

index_rows = []
sent_idx = 0

for (conv_id, speaker), group in df.groupby(['conv_id','speaker'], sort=False):
    group = group.reset_index(drop=True)
    words = group['word'].str.lower().tolist()
    wids  = group['word_id'].tolist()

    for i in range(0, len(words), 15):
        chunk_words = words[i:i+15]
        chunk_ids   = wids[i:i+15]
        if len(chunk_words) >= 3:
            for pos, (w, wid) in enumerate(zip(chunk_words, chunk_ids)):
                index_rows.append({
                    'sent_idx': sent_idx,
                    'word_pos': pos,
                    'word_id':  wid,
                    'word':     w
                })
            sent_idx += 1

index_df = pd.DataFrame(index_rows)
os.makedirs(os.path.dirname(OUT_INDEX), exist_ok=True)
index_df.to_csv(OUT_INDEX, index=False)

print(f"Done - {sent_idx:,} sentences, {len(index_rows):,} words in index")
print(f"Saved -> {OUT_INDEX}")
print("\nFirst 10 rows:")
print(index_df.head(10).to_string())