import numpy as np
import pandas as pd

df = pd.read_csv('data/fb/dataset-fb-valence-arousal-anon.csv')
df = df[df['Anonymized Message'].notna()]
df.index=np.arange(len(df))
col = df.loc[:, ['Valence1', 'Valence2']]
df['valence'] = col.mean(axis=1)
col = df.loc[:, ['Arousal1', 'Arousal2']]
df['arousal'] = col.mean(axis=1)
df = df[['Anonymized Message', 'arousal', 'valence']]
df.columns = ['text', 'arousal', 'valence']
df.to_csv('metadata_fb.csv', index=False)