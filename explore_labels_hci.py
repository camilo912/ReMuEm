import os
import pandas as pd
from matplotlib import pyplot as plt

data_dir = 'data/hci_tagging/Sessions/'

df = pd.DataFrame(columns = ['sess', 'arousal', 'valence'])

cont = 0
for sess in os.listdir(data_dir):
    with open(data_dir + sess + '/session.xml', 'r') as f:
        f.readline()
        l = f.readline()
        idx = l.index('feltArsl="')
        aro = int(l[idx+10:idx+11])
        assert 1 <= aro and aro <= 9
        idx = l.index('feltVlnc="')
        val = int(l[idx+10:idx+11])
        assert 1 <= val and val <= 9
        idx = l.index('sessionId="')
        idx_2 = l.index('" date="')
        sess_id = l[idx+11:idx_2]
        assert sess_id == sess

        df.loc[cont] = [sess_id, aro, val]
        cont +=1

df = df.astype({'arousal': float, 'valence':float})
print(df[['arousal', 'valence']].describe())

print(df.loc[(df['arousal']<=3) & (df['valence']<=3), ['arousal', 'valence']].describe())
print(df.loc[(df['arousal']<=3) & (df['valence'].between(3,6,inclusive=False)), ['arousal', 'valence']].describe())
print(df.loc[(df['arousal']<=3) & (df['valence']>=6), ['arousal', 'valence']].describe())
print(df.loc[(df['arousal'].between(3,6,inclusive=False)) & (df['valence']<=3), ['arousal', 'valence']].describe())
print(df.loc[(df['arousal'].between(3,6,inclusive=False)) & (df['valence'].between(3,6,inclusive=False)), ['arousal', 'valence']].describe())
print(df.loc[(df['arousal'].between(3,6,inclusive=False)) & (df['valence']>=6), ['arousal', 'valence']].describe())
print(df.loc[(df['arousal']>=6) & (df['valence']<=3), ['arousal', 'valence']].describe())
print(df.loc[(df['arousal']>=6) & (df['valence'].between(3,6,inclusive=False)), ['arousal', 'valence']].describe())
print(df.loc[(df['arousal']>=6) & (df['valence']>=6), ['arousal', 'valence']].describe())

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
