import numpy as np
import pandas as pd
import pickle

metadata = pd.read_csv('../metadata_mosei.csv')
data = np.load('data/mosei/data.npy', allow_pickle=True)
selected_data = np.load('data/mosei/data_selected.npy', allow_pickle=True)
labels = np.load('data/mosei/labels.npy', allow_pickle=True)
print(len(labels))

cont = 0
for i in range(len(metadata)):
    path,vid,clip,emotion,arousal,valence = metadata.loc[i, :]
    aro,val = labels[i, 1:3].astype(np.float64)

    if(aro != arousal and valence != val):
        selected_data[0] = np.insert(selected_data[0], i, np.array([np.nan for _ in range(selected_data[0].shape[1])]), 0)
        selected_data[1] = np.insert(selected_data[1], i, np.array([np.nan for _ in range(selected_data[1].shape[1])]), 0)
        selected_data[2] = np.insert(selected_data[2], i, np.array([np.nan for _ in range(selected_data[2].shape[1])]), 0)
        data = np.insert(data, i, np.array([np.nan for _ in range(data.shape[1])]), 0)
        labels = np.insert(labels, i, np.array([emotion,arousal,valence]), 0)

        cont += 1

print(cont)

cont = 0
for i in range(len(metadata)):
    path,vid,clip,emotion,arousal,valence = metadata.loc[i, :]
    aro,val = labels[i, 1:3].astype(np.float64)
    if(aro != arousal and valence != val):
        cont += 1

print(cont)
print(labels.shape)
print(data.shape)
print(selected_data[0].shape)
print(selected_data[1].shape)
print(selected_data[2].shape)

assert cont == 0
assert data.shape[0] == metadata.shape[0]
assert selected_data[0].shape[0] == metadata.shape[0]
assert selected_data[1].shape[0] == metadata.shape[0]
assert selected_data[2].shape[0] == metadata.shape[0]
assert labels.shape[0] == metadata.shape[0]

np.save('data/mosei/data.npy', data)
pickle.dump(selected_data, open('data/mosei/data_selected.npy', 'wb'))
np.save('data/mosei/labels.npy', labels)
