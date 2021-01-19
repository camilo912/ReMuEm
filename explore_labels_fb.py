import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('data/fb/dataset-fb-valence-arousal-anon.csv')
df = df[df['Anonymized Message'].notna()]
df.index=np.arange(len(df))
col = df.loc[:, ['Valence1', 'Valence2']]
df['valence'] = col.mean(axis=1)
col = df.loc[:, ['Arousal1', 'Arousal2']]
df['arousal'] = col.mean(axis=1)
df = df[['Anonymized Message', 'arousal', 'valence']]
df.columns = ['text', 'arousal', 'valence']

print(df['arousal'].describe())
print(df['valence'].describe())

fig, ax = plt.subplots()
plt.plot(-1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, -1, 'wo', marker=".", markersize=0)
plt.plot(1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, 1, 'wo', marker=".", markersize=0)
plt.plot(0, 0, 'ko', marker="+")
plt.plot((df['valence'].values-1)/4-1, (df['arousal'].values-1)/4-1, 'bo', label='labels', markersize=2)
ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()