import pandas as pd

df = pd.read_csv('data/processed/words_labeled.csv')
df = df.sort_values(['conv_id','speaker','start'])
df = df[df['label'] == 'fluent']
df['word'] = df['word'].fillna('').astype(str)
df['word'] = df['word'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)
df = df[df['word'].str.len() > 0]

sentences = (df.groupby(['conv_id','speaker'])['word']
               .apply(lambda x: list(x.str.lower()))
               .reset_index())

short_sents = []
for _, row in sentences.iterrows():
    words = row['word']
    for i in range(0, len(words), 15):
        chunk = words[i:i+15]
        if len(chunk) >= 3:
            short_sents.append(' '.join(chunk))

with open('swbd_500.linetoks', 'w', newline='\n') as f:
    for s in short_sents[:500]:
        f.write(s + '\n')

print(f'Done - {min(500, len(short_sents))} sentences written')
print('Line 1:', short_sents[0])
print('Line 2:', short_sents[1])
print('Line 3:', short_sents[2])